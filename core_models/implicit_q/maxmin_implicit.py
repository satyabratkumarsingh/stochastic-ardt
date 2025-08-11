import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from copy import deepcopy
from data_class.trajectory import Trajectory
from core_models.base_models.base_model import RtgFFN, RtgLSTM
from core_models.implicit_q.value_net import ValueNet
from core_models.dataset.ardt_dataset import ARDTDataset

# Loss weights from your stable implementation
IQL_LOSS_WEIGHT = 0.2
Q_ONLY_LOSS_WEIGHT = 0.2

def _iql_loss_v(q_values, v_values, acts_mask, tau=0.7):
    """Simplified V loss for debugging."""
    q_flat = q_values.squeeze(-1)
    v_flat = v_values.squeeze(-1) 
    valid_mask = (~acts_mask).float()
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=q_values.device, requires_grad=True)
    
    diff = q_flat.detach() - v_flat
    weight = torch.where(diff < 0, 1 - tau, tau)
    loss = ((weight * diff**2 * valid_mask).sum() / valid_mask.sum())
    
    return loss.clamp(min=1e-8)


def _iql_loss_q(
    q_values: torch.Tensor,
    rewards: torch.Tensor,
    v_values: torch.Tensor,
    acts_mask: torch.Tensor,
    gamma: float
) -> torch.Tensor:
    """
    IQL Q-loss: L_Q(θ) = E[(r + γV_ψ(s') - Q_θ(s,a))^2]
    """
    batch_size, seq_len, _ = q_values.shape
    
    q_values = q_values.squeeze(-1)  # [batch_size, seq_len]
    rewards = rewards.squeeze(-1)    # [batch_size, seq_len-1]
    v_values = v_values.squeeze(-1)  # [batch_size, seq_len]
    
    # Q-learning target: r + γV(s')
    q_target = rewards + gamma * v_values[:, 1:].detach()  # [batch_size, seq_len-1]
    q_target = torch.clamp(q_target, -1e6, 1e6)
    
    # Current Q-values for transitions
    q_pred = q_values[:, :-1]  # [batch_size, seq_len-1]
    
    # Valid transitions mask
    valid_mask = ~acts_mask[:, :-1]  # [batch_size, seq_len-1]
    
    if valid_mask.sum() > 0:
        loss = ((q_pred - q_target) ** 2 * valid_mask).sum() / valid_mask.sum()
    else:
        loss = torch.tensor(0.0, device=q_values.device)
    
    return torch.clamp(loss, 0.0, 1e6)


def _q_only_loss_max(
    q_values: torch.Tensor,
    rewards: torch.Tensor,
    acts_mask: torch.Tensor,
    gamma: float
) -> torch.Tensor:
    """
    Q-only loss with max operator: L_Q_only(θ) = E[(r + γ max Q_θ(s',a') - Q_θ(s,a))^2]
    """
    batch_size, seq_len, action_size = q_values.shape
    
    # Current Q-values
    q_pred = q_values[:, :-1]  # [batch_size, seq_len-1, action_size]
    
    # Next Q-values with max operator
    if action_size == 1:
        q_next = q_values[:, 1:].detach()  # [batch_size, seq_len-1, 1]
        q_pred = q_pred  # [batch_size, seq_len-1, 1]
    else:
        q_next = q_values[:, 1:].detach().max(dim=-1, keepdim=True)[0]  # [batch_size, seq_len-1, 1]
        q_pred = q_pred.max(dim=-1, keepdim=True)[0]  # [batch_size, seq_len-1, 1]
    
    # Compute target
    rewards = rewards.squeeze(-1) if len(rewards.shape) > 2 else rewards  # [batch_size, seq_len-1]
    if len(rewards.shape) == 1:
        rewards = rewards.unsqueeze(0)  # Handle single batch case
    
    q_target = rewards + gamma * q_next.squeeze(-1)  # [batch_size, seq_len-1]
    q_target = torch.clamp(q_target, -1e6, 1e6)
    
    # Valid transitions mask
    valid_mask = ~acts_mask[:, :-1]  # [batch_size, seq_len-1]
    
    if valid_mask.sum() > 0:
        loss = ((q_pred.squeeze(-1) - q_target) ** 2 * valid_mask).sum() / valid_mask.sum()
    else:
        loss = torch.tensor(0.0, device=q_values.device)
    
    return torch.clamp(loss, 0.0, 1e6)


def _q_only_loss_min(
    q_values: torch.Tensor,
    rewards: torch.Tensor,
    acts_mask: torch.Tensor,
    gamma: float
) -> torch.Tensor:
    """
    Q-only loss with min operator for adversary: L_Q_only(θ) = E[(r + γ min Q_θ(s',a') - Q_θ(s,a))^2]
    """
    batch_size, seq_len, action_size = q_values.shape
    
    # Current Q-values
    q_pred = q_values[:, :-1]  # [batch_size, seq_len-1, action_size]
    
    # Next Q-values with min operator for adversary
    if action_size == 1:
        q_next = q_values[:, 1:].detach()  # [batch_size, seq_len-1, 1]
        q_pred = q_pred  # [batch_size, seq_len-1, 1]
    else:
        q_next = q_values[:, 1:].detach().min(dim=-1, keepdim=True)[0]  # [batch_size, seq_len-1, 1]
        q_pred = q_pred.min(dim=-1, keepdim=True)[0]  # [batch_size, seq_len-1, 1]
    
    # Compute target
    rewards = rewards.squeeze(-1) if len(rewards.shape) > 2 else rewards
    if len(rewards.shape) == 1:
        rewards = rewards.unsqueeze(0)
    
    q_target = rewards + gamma * q_next.squeeze(-1)
    q_target = torch.clamp(q_target, -1e6, 1e6)
    
    # Valid transitions mask
    valid_mask = ~acts_mask[:, :-1]
    
    if valid_mask.sum() > 0:
        loss = ((q_pred.squeeze(-1) - q_target) ** 2 * valid_mask).sum() / valid_mask.sum()
    else:
        loss = torch.tensor(0.0, device=q_values.device)
    
    return torch.clamp(loss, 0.0, 1e6)


def evaluate_models(qsa_pr_model, qsa_adv_model, v_model, dataloader, device):
    """Evaluate Q_max, Q_min, and V on one batch and print mean predictions + gaps."""
    with torch.no_grad():
        obs, acts, adv_acts, ret, seq_len = next(iter(dataloader))
        batch_size = obs.shape[0]
        obs_len = obs.shape[1]
        
        obs = obs.view(batch_size, obs_len, -1).to(device)
        acts = acts.to(device)
        adv_acts = adv_acts.to(device)
        ret = ret.to(device)
        
        # Create mask for valid timesteps
        timestep_indices = torch.arange(obs_len, device=device).unsqueeze(0).expand(batch_size, -1)
        acts_mask = timestep_indices >= seq_len.unsqueeze(1).to(device)
        valid_mask = ~acts_mask
        
        # Get predictions
        pred_pr = qsa_pr_model(obs, acts)
        pred_adv = qsa_adv_model(obs, acts, adv_acts)  
        pred_v = v_model(obs)
        
        # Compute means only for valid timesteps
        if valid_mask.sum() > 0:
            pred_pr_mean = (pred_pr.squeeze(-1) * valid_mask).sum() / valid_mask.sum()
            pred_adv_mean = (pred_adv.squeeze(-1) * valid_mask).sum() / valid_mask.sum()
            pred_v_mean = (pred_v.squeeze(-1) * valid_mask).sum() / valid_mask.sum()
            true_mean = (ret * valid_mask).sum() / valid_mask.sum()
        else:
            pred_pr_mean = pred_pr.mean()
            pred_adv_mean = pred_adv.mean() 
            pred_v_mean = pred_v.mean()
            true_mean = ret.mean()

    gap_pr_adv = pred_pr_mean.item() - pred_adv_mean.item()
    gap_pr_v = pred_pr_mean.item() - pred_v_mean.item()
    gap_v_adv = pred_v_mean.item() - pred_adv_mean.item()
    
    symbol_pr_adv = "✅" if gap_pr_adv >= 0 else "❌"
    symbol_pr_v = "✅" if gap_pr_v >= 0 else "❌"
    symbol_v_adv = "✅" if gap_v_adv >= 0 else "❌"
    
    print(f"   Eval -> True mean: {true_mean.item():.4f}")
    print(f"         Q_pr mean: {pred_pr_mean.item():.4f}, Q_adv mean: {pred_adv_mean.item():.4f}, V mean: {pred_v_mean.item():.4f}")
    print(f"         Q_pr - Q_adv: {gap_pr_adv:.4f} {symbol_pr_adv}, Q_pr - V: {gap_pr_v:.4f} {symbol_pr_v}, V - Q_adv: {gap_v_adv:.4f} {symbol_v_adv}")
    
    return pred_pr_mean.item(), pred_adv_mean.item(), pred_v_mean.item(), gap_pr_adv


def maxmin(
    trajs: list[Trajectory],
    action_space: gym.spaces,
    adv_action_space: gym.spaces,
    train_args: dict,
    device: str,
    n_cpu: int,
    is_simple_model: bool = False,
    is_toy: bool = False,
    is_discretize: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Train a max-min adversarial RL model with value functions and IQL losses.
    Updated to handle the new JSON dataset format and **without scaling**.
    """
    print("=== IQL Minimax Training with JSON Dataset (No Scaling) ===")
    
    # Collect rewards for statistics
    all_rewards = []
    for traj in trajs:
        episode_reward = np.sum(traj.rewards)
        all_rewards.append(episode_reward)
    print(f"Dataset size: {len(trajs)} episodes")
    print(f"Reward stats: mean={np.mean(all_rewards):.3f}, std={np.std(all_rewards):.3f}")
    
    # Setup action spaces and sizes
    if isinstance(action_space, gym.spaces.Discrete):
        obs_size = np.prod(trajs[0].obs[0].shape)
        action_size = action_space.n
        adv_action_size = adv_action_space.n
        action_type = 'discrete'
    else:
        obs_size = np.prod(np.array(trajs[0].obs[0]).shape)
        action_size = action_space.shape[0]
        adv_action_size = adv_action_space.shape[0]
        action_type = 'continuous'

    print(f"Dataset info: obs_size={obs_size}, action_size={action_size}, adv_action_size={adv_action_size}")

    # Build dataset and dataloader for training
    max_len = max([len(traj.obs) for traj in trajs]) + 1
    dataset = ARDTDataset(
        trajs, 
        max_len, 
        gamma=train_args['gamma'], 
        act_type=action_type
    )
    print(f"Dataset: {len(dataset)} samples, max_len={max_len}")
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_args['batch_size'], num_workers=n_cpu
    )

    # Set up the models
    print(f'Creating models... (simple={is_simple_model})')
    if is_simple_model:
        qsa_pr_model = RtgFFN(obs_size, action_size, include_adv=False).to(device)
        qsa_adv_model = RtgFFN(obs_size, action_size, adv_action_size, include_adv=True).to(device)
        v_model = ValueNet(obs_size, is_lstm=False).to(device)
    else:
        qsa_pr_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=False).to(device)
        qsa_adv_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)
        v_model = ValueNet(obs_size, train_args, is_lstm=True).to(device)

    # Optimizers with weight decay
    qsa_pr_optimizer = torch.optim.AdamW(
        qsa_pr_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args.get('model_wd', 1e-4)
    )
    qsa_adv_optimizer = torch.optim.AdamW(
        qsa_adv_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args.get('model_wd', 1e-4)
    )
    value_optimizer = torch.optim.AdamW(
        v_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args.get('model_wd', 1e-4)
    )

    # Training parameters
    mse_epochs = train_args.get('mse_epochs', 10)
    maxmin_epochs = train_args.get('maxmin_epochs', 20)
    total_epochs = mse_epochs + maxmin_epochs
    gamma = train_args['gamma']
    
    print(f'Training for {total_epochs} epochs (MSE: {mse_epochs}, MaxMin: {maxmin_epochs})')
    
    # Ensure maxmin_epochs is even for alternating updates
    if maxmin_epochs % 2 != 0:
        maxmin_epochs += 1
        total_epochs += 1
        print(f"Adjusted maxmin_epochs to {maxmin_epochs} for alternating updates")
    
    print("=== Training Configuration ===")
    print(f"MSE epochs: {mse_epochs}")
    print(f"Minimax epochs: {maxmin_epochs}")
    print(f"Using ALTERNATING updates during minimax phase")
    print("No scaling applied.")
    print("Phase order: MSE -> MAX (protagonist) -> MIN (adversary)")

    # Training loop
    qsa_pr_model.train()
    qsa_adv_model.train()
    v_model.train()
    
    for epoch in range(total_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        total_loss = 0
        total_pr_loss = 0
        total_adv_loss = 0
        total_v_loss = 0
        total_q_loss = 0
        total_iql_loss = 0
        total_batches = 0

        for obs, acts, adv_acts, ret, seq_len in pbar:
            total_batches += 1
            
            # Adjust for toy environment
            if is_toy:
                obs, acts, adv_acts, ret = (
                    obs[:, :-1], acts[:, :-1], adv_acts[:, :-1], ret[:, :-1]
                )
                seq_len = torch.clamp(seq_len - 1, min=1)

            # Clamp sequence lengths to valid range
            seq_len = torch.clamp(seq_len, max=obs.shape[1])
            
            # Set up variables
            batch_size = obs.shape[0]
            obs_len = obs.shape[1]
            
            obs = obs.view(batch_size, obs_len, -1).to(device)
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            ret = ret.to(device)
            
            # Create mask for padded timesteps (True for padded positions)
            timestep_indices = torch.arange(obs_len, device=device).unsqueeze(0).expand(batch_size, -1)
            acts_mask = timestep_indices >= seq_len.unsqueeze(1)
            
            # Compute single-step rewards: r_t = R_t - γ * R_{t+1}
            if obs_len > 1:
                rewards = ret[:, :-1] - gamma * ret[:, 1:]  # [batch_size, seq_len-1]
                rewards = rewards.unsqueeze(-1)  # [batch_size, seq_len-1, 1]
                rewards = torch.clamp(rewards, -1e6, 1e6)
            else:
                rewards = torch.zeros(batch_size, 0, 1, device=device)

            # Zero gradients
            qsa_pr_optimizer.zero_grad()
            qsa_adv_optimizer.zero_grad()
            value_optimizer.zero_grad()

            # Model predictions
            ret_pr_pred = qsa_pr_model(obs, acts)  # [batch_size, obs_len, 1]
            ret_adv_pred = qsa_adv_model(obs, acts, adv_acts)  # [batch_size, obs_len, 1]
            v_pred = v_model(obs)  # [batch_size, obs_len, 1]
            
            # Clamp predictions to avoid numerical issues
            ret_pr_pred = torch.clamp(ret_pr_pred, -1e6, 1e6)
            ret_adv_pred = torch.clamp(ret_adv_pred, -1e6, 1e6)
            v_pred = torch.clamp(v_pred, -1e6, 1e6)

            # Training phases
            if epoch < mse_epochs:
                # Phase 1: MSE pre-training
                ret_target = ret.unsqueeze(-1)  # [batch_size, obs_len, 1]
                
                # MSE losses for Q-functions
                valid_mask_expanded = (~acts_mask).unsqueeze(-1).float()
                ret_pr_loss = ((ret_pr_pred - ret_target) ** 2 * valid_mask_expanded).sum() / valid_mask_expanded.sum()
                ret_adv_loss = ((ret_adv_pred - ret_target) ** 2 * valid_mask_expanded).sum() / valid_mask_expanded.sum()
                
                # IQL Value losses using proper expectile formulation
                # L_V(ψ) for protagonist (τ=0.7, conservative)
                v_loss_pr = _iql_loss_v(ret_pr_pred, v_pred, acts_mask, tau=0.7)
                
                # L_V(ψ) for adversary (τ=0.3, optimistic for minimization)  
                v_loss_adv = _iql_loss_v(ret_adv_pred, v_pred, acts_mask, tau=0.3)
                
                # Weighted combination as per the proposal
                v_loss = 0.7 * v_loss_pr + 0.3 * v_loss_adv
                
                # IQL Q-losses: L_Q(θ) = E[(r + γV_ψ(s') - Q_θ(s,a))^2]
                iql_loss_pr = _iql_loss_q(ret_pr_pred, rewards, v_pred, acts_mask, gamma)
                iql_loss_adv = _iql_loss_q(ret_adv_pred, rewards, v_pred, acts_mask, gamma)
                
                # Q-only losses with max/min operators
                q_only_loss_pr = _q_only_loss_max(ret_pr_pred, rewards, acts_mask, gamma)
                q_only_loss_adv = _q_only_loss_min(ret_adv_pred, rewards, acts_mask, gamma)
                
                # Combined loss
                total_loss_epoch = (
                    ret_pr_loss + ret_adv_loss + v_loss + 
                    IQL_LOSS_WEIGHT * (iql_loss_pr + iql_loss_adv) + 
                    Q_ONLY_LOSS_WEIGHT * (q_only_loss_pr + q_only_loss_adv)
                )
                
                total_loss_epoch = torch.clamp(total_loss_epoch, 0.0, 1e6)
                total_loss_epoch.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(qsa_pr_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(qsa_adv_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(v_model.parameters(), max_norm=1.0)
                
                qsa_pr_optimizer.step()
                qsa_adv_optimizer.step()
                value_optimizer.step()
                
                # Update statistics
                total_loss += total_loss_epoch.item()
                total_pr_loss += ret_pr_loss.item()
                total_adv_loss += ret_adv_loss.item()
                total_v_loss += v_loss.item()
                total_q_loss += (q_only_loss_pr.item() + q_only_loss_adv.item())
                total_iql_loss += (iql_loss_pr.item() + iql_loss_adv.item())
                
            elif (epoch - mse_epochs) % 2 == 0:
                # Phase 2a: Max step - protagonist maximizes
                alpha = train_args.get('alpha', 0.8)  # Higher alpha for maximization
                
                # Core minimax loss: protagonist maximizes using expectile loss
                if obs_len > 1 and (~acts_mask[:, :-1]).sum() > 0:
                    # Expectile loss on protagonist predictions vs adversary targets
                    pr_pred_transitions = ret_pr_pred[:, :-1].squeeze(-1)  # [batch_size, seq_len-1]
                    
                    # Protagonist maximizes against adversary's future predictions (detached)
                    with torch.no_grad():
                        adv_next = ret_adv_pred[:, 1:].squeeze(-1)  # [batch_size, seq_len-1]
                        targets = rewards.squeeze(-1) + gamma * adv_next  # [batch_size, seq_len-1]
                        targets = torch.clamp(targets, -1e6, 1e6)
                    
                    # Expectile loss for maximization (protagonist wants to be above target)
                    diff = pr_pred_transitions - targets
                    valid_transitions = ~acts_mask[:, :-1]
                    
                    if valid_transitions.sum() > 0:
                        weight = torch.where(diff >= 0, alpha, 1 - alpha)
                        ret_pr_loss = (weight * (diff ** 2) * valid_transitions).sum() / valid_transitions.sum()
                    else:
                        ret_pr_loss = torch.tensor(0.0, device=device)
                else:
                    ret_pr_loss = torch.tensor(0.0, device=device)
                
                # Value function learns from protagonist using IQL formulation
                v_loss = _iql_loss_v(ret_pr_pred, v_pred, acts_mask, tau=0.7)
                
                # Q-learning consistency for protagonist using IQL Q-loss
                iql_loss_pr = _iql_loss_q(ret_pr_pred, rewards, v_pred, acts_mask, gamma)
                
                # Q-only loss with max operator
                q_only_loss_pr = _q_only_loss_max(ret_pr_pred, rewards, acts_mask, gamma)
                
                total_loss_epoch = (
                    ret_pr_loss + 
                    0.1 * v_loss + 
                    IQL_LOSS_WEIGHT * iql_loss_pr +
                    Q_ONLY_LOSS_WEIGHT * q_only_loss_pr
                )
                total_loss_epoch = torch.clamp(total_loss_epoch, 0.0, 1e6)
                total_loss_epoch.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(qsa_pr_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(v_model.parameters(), max_norm=1.0)
                
                qsa_pr_optimizer.step()
                value_optimizer.step()
                
                # Update statistics
                total_loss += total_loss_epoch.item()
                total_pr_loss += ret_pr_loss.item()
                total_v_loss += v_loss.item()
                total_q_loss += q_only_loss_pr.item()
                total_iql_loss += iql_loss_pr.item()
                
            else:
                # Phase 2b: Min step - adversary minimizes
                alpha = train_args.get('alpha', 0.3)  # Lower alpha for minimization
                leaf_weight = train_args.get('leaf_weight', 0.1)
                
                # Tree loss: adversary tries to minimize Q_adv against protagonist's future
                if obs_len > 1:
                    # Corrected expectile for minimization
                    pr_future = ret_pr_pred[:, 1:].detach().squeeze(-1)  # [batch_size, seq_len-1]
                    adv_current = ret_adv_pred[:, :-1].squeeze(-1)  # [batch_size, seq_len-1]
                    
                    # Target that adversary wants to be below
                    targets = rewards.squeeze(-1) + gamma * pr_future
                    targets = torch.clamp(targets, -1e6, 1e6)
                    
                    # Difference for expectile loss (adversary wants to be below target)
                    diff = adv_current - targets  # [batch_size, seq_len-1]
                    valid_transitions = ~acts_mask[:, :-1]
                    
                    if valid_transitions.sum() > 0:
                        # For minimization: higher weight when above target (diff > 0)
                        weight = torch.where(diff >= 0, 1 - alpha, alpha)  # Penalize being above target
                        ret_tree_loss = (weight * (diff ** 2) * valid_transitions).sum() / valid_transitions.sum()
                    else:
                        ret_tree_loss = torch.tensor(0.0, device=device)
                else:
                    ret_tree_loss = torch.tensor(0.0, device=device)
                
                # Leaf loss: terminal state consistency
                if (~acts_mask).sum() > 0:
                    # Use last valid timestep for each sequence
                    last_valid_timesteps = []
                    for b in range(batch_size):
                        valid_t = torch.where(~acts_mask[b])[0]
                        if len(valid_t) > 0:
                            last_valid_timesteps.append((b, valid_t[-1].item()))
                    
                    if last_valid_timesteps:
                        batch_indices, time_indices = zip(*last_valid_timesteps)
                        batch_indices = torch.tensor(batch_indices, device=device)
                        time_indices = torch.tensor(time_indices, device=device)
                        
                        ret_leaf_loss = ((ret_adv_pred[batch_indices, time_indices, 0] - 
                                        ret[batch_indices, time_indices]) ** 2).mean()
                    else:
                        ret_leaf_loss = torch.tensor(0.0, device=device)
                else:
                    ret_leaf_loss = torch.tensor(0.0, device=device)
                
                ret_adv_loss = (1 - leaf_weight) * ret_tree_loss + leaf_weight * ret_leaf_loss
                
                # Value function learns from adversary using conservative expectile (tau=0.3)
                v_loss = _iql_loss_v(ret_adv_pred, v_pred, acts_mask, tau=0.3)
                
                # Q-only loss for adversary (minimization)
                q_only_loss_adv = _q_only_loss_min(ret_adv_pred, rewards, acts_mask, gamma)
                
                # IQL loss for adversary
                iql_loss_adv = _iql_loss_q(ret_adv_pred, rewards, v_pred, acts_mask, gamma)
                
                total_loss_epoch = (
                    ret_adv_loss + 
                    0.1 * v_loss + 
                    Q_ONLY_LOSS_WEIGHT * q_only_loss_adv + 
                    IQL_LOSS_WEIGHT * iql_loss_adv
                )
                total_loss_epoch = torch.clamp(total_loss_epoch, 0.0, 1e6)
                total_loss_epoch.backward()
                
                # Gradient clipping
                torch.nn.utils.clip_grad_norm_(qsa_adv_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(v_model.parameters(), max_norm=1.0)
                
                qsa_adv_optimizer.step()
                value_optimizer.step()
                
                # Update statistics  
                total_loss += total_loss_epoch.item()
                total_adv_loss += ret_adv_loss.item()
                total_v_loss += v_loss.item()
                total_q_loss += q_only_loss_adv.item()
                total_iql_loss += iql_loss_adv.item()

            # Update progress bar
            phase = "MSE" if epoch < mse_epochs else ("MAX" if (epoch - mse_epochs) % 2 == 0 else "MIN")
            pbar.set_description(f"Epoch {epoch} ({phase}) | "
                               f"Loss: {total_loss / total_batches:.4f} | "
                               f"Pr: {total_pr_loss / total_batches:.4f} | "
                               f"Adv: {total_adv_loss / total_batches:.4f} | "
                               f"V: {total_v_loss / total_batches:.4f} | "
                               f"Q: {total_q_loss / total_batches:.4f} | "
                               f"IQL: {total_iql_loss / total_batches:.4f}")

        print(f"Epoch {epoch} completed - Average Loss: {total_loss / total_batches:.6f}")
        
        # Evaluate models at the end of each epoch
        if epoch < mse_epochs:
            print(f"MSE Epoch {epoch}:")
        elif (epoch - mse_epochs) % 2 == 0:
            print(f"Minimax Epoch {epoch - mse_epochs} (MAX step):")
        else:
            print(f"Minimax Epoch {epoch - mse_epochs} (MIN step):")
            
        evaluate_models(qsa_pr_model, qsa_adv_model, v_model, dataloader, device)

    # ---- Relabeling Phase ----
    print("\n=== Trajectory Relabeling ===")
    qsa_pr_model.eval()
    with torch.no_grad():
        relabeled_trajs = []
        learned_returns = []
        prompt_value = -np.inf
        
        for traj in tqdm(trajs, desc="Relabeling trajectories"):
            obs = torch.from_numpy(np.array(traj.obs)).float().to(device).view(1, -1, obs_size)
            acts = torch.from_numpy(np.array(traj.actions)).to(device)
            if action_type == "discrete" and not is_discretize:
                acts = acts.float().view(1, -1, action_size)
            else:
                acts = acts.view(1, -1, action_size)
                
            returns = qsa_pr_model(obs, acts).cpu().flatten().numpy()
            if len(returns) > 0 and prompt_value < returns[0]:
                prompt_value = returns[0]
            learned_returns.append(np.round(returns, decimals=3))
            
            relabeled_traj = deepcopy(traj)
            relabeled_traj.minimax_returns_to_go = returns.tolist()
            relabeled_trajs.append(relabeled_traj)

    print(f"Relabeled {len(relabeled_trajs)} trajectories with minimax returns-to-go")
    print(f"Training complete. Prompt value: {prompt_value:.3f}")
    return relabeled_trajs, np.round(prompt_value, decimals=3)