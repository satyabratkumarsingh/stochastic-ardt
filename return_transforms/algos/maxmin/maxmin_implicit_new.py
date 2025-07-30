import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from copy import deepcopy

# Assuming these are available from your project structure
from data_loading.load_mujoco import Trajectory
from return_transforms.models.ardt.maxmin_model import RtgFFN, RtgLSTM
from return_transforms.datasets.ardt_dataset import ARDTDataset
from return_transforms.models.ardt.value_net import ValueNet

IQL_LOSS_WEIGHT = 0.2
Q_ONLY_LOSS_WEIGHT = 0.2

def expectile_loss(diff: torch.Tensor, alpha: float) -> torch.Tensor:
    """
    Expectile loss function as used in ARDT paper.
    
    Args:
        diff: Prediction - target differences
        alpha: Expectile level
               - alpha < 0.5: Approximates maximum (optimistic)
               - alpha > 0.5: Approximates minimum (pessimistic)
    """
    weight = torch.where(diff >= 0, alpha, 1 - alpha)
    return weight * (diff ** 2)

def _iql_loss(
    v_values: torch.Tensor,
    q_values: torch.Tensor,
    rewards: torch.Tensor,
    acts_mask: torch.Tensor,
    gamma: float,
    device: str
) -> torch.Tensor:
    """
    Compute IQL loss to align Q-values with TD targets, ensuring conservative Q-learning.
    """
    batch_size, seq_len, _ = v_values.shape
    v_values = v_values.squeeze(-1)
    rewards = rewards.squeeze(-1)
    q_values = q_values.squeeze(-1) if q_values.shape[-1] == 1 else q_values
    
    q_target = rewards + gamma * v_values[:, 1:].detach()
    q_target = torch.nan_to_num(q_target, nan=0.0, posinf=1e6, neginf=-1e6)
    
    if len(q_values.shape) == 3:
        q_pred = q_values[:, :-1].max(dim=-1)[0]
    else:
        q_pred = q_values[:, :-1]
    
    loss = (((q_pred - q_target) ** 2) * ~acts_mask[:, :-1]).mean()
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    
    return loss


def _q_only_loss(
    q_values: torch.Tensor,
    rewards: torch.Tensor,
    acts_mask: torch.Tensor,
    gamma: float,
    device: str
) -> torch.Tensor:
    """
    Compute Q-only loss (Bellman loss) to align Q-values with TD targets.
    """
    batch_size, seq_len, action_size = q_values.shape
    
    if action_size == 1:
        q_next = q_values[:, 1:].detach()
    else:
        q_next = q_values[:, 1:].detach().max(dim=-1, keepdim=True)[0]
    
    q_target = rewards + gamma * q_next
    q_target = torch.nan_to_num(q_target, nan=0.0, posinf=1e6, neginf=-1e6)
    
    q_pred = q_values[:, :-1]
    if action_size > 1:
        q_pred = q_pred.max(dim=-1, keepdim=True)[0]
    
    loss = (((q_pred - q_target) ** 2) * ~acts_mask[:, :-1].unsqueeze(-1)).mean()
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    
    return loss


def evaluate_models(R_max_model, R_min_model, v_model, dataloader, device, scale):
    """Evaluate R_max and R_min on one batch and print mean predictions + gap."""
    with torch.no_grad():
        obs, acts, adv_acts, ret, seq_len = next(iter(dataloader))
        batch_size, obs_len = obs.shape[0], obs.shape[1]
        
        obs = obs.view(batch_size, obs_len, -1).to(device)
        acts = acts.to(device)
        adv_acts = adv_acts.to(device)
        ret_scaled = (ret / scale).to(device)
        
        pred_max = R_max_model(obs, acts).mean().item()
        pred_min = R_min_model(obs, acts, adv_acts).mean().item()
        pred_v = v_model(obs).mean().item()
        true_mean = ret_scaled.mean().item()

    gap = pred_max - pred_min
    symbol = "✅" if gap >= 0 else "❌"
    print(f"   Eval -> True mean: {true_mean:.4f}, "
          f"R_max mean: {pred_max:.4f}, R_min mean: {pred_min:.4f}, V mean: {pred_v:.4f}, "
          f"Gap: {gap:.4f} {symbol}")
    return pred_max, pred_min, gap


def ardt_minimax_expectile_regression(
    trajs: list[Trajectory],
    obs_size: int,
    action_size: int,
    adv_action_size: int,
    train_args: dict,
    device: str,
    scale: float = 1.0,
    is_simple_model: bool = False
) -> tuple[torch.nn.Module, torch.nn.Module]:
    
    # Models
    if is_simple_model:
        R_max_model = RtgFFN(obs_size, action_size, include_adv=False).to(device)
        R_min_model = RtgFFN(obs_size, action_size, adv_action_size, include_adv=True).to(device)
        v_model = ValueNet(obs_size, is_lstm=False).to(device)
    else:
        R_max_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=False).to(device)
        R_min_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)
        v_model = ValueNet(obs_size, train_args, is_lstm=True).to(device)

    max_optimizer = torch.optim.AdamW(R_max_model.parameters(), lr=train_args['model_lr'])
    min_optimizer = torch.optim.AdamW(R_min_model.parameters(), lr=train_args['model_lr'])
    value_optimizer = torch.optim.AdamW(v_model.parameters(), lr=train_args['model_lr'])

    # Dataset
    max_len = max([len(traj.obs) for traj in trajs]) + 1
    dataset = ARDTDataset(trajs, max_len, gamma=train_args['gamma'], act_type='continuous')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_args['batch_size'])

    alpha_max = 0.1
    alpha_min = 0.9
    warmup_epochs = train_args.get('warmup_epochs', 10)
    minimax_epochs = train_args.get('minimax_epochs', 20)
    gamma = train_args.get('gamma', 0.99)
    leaf_weight = train_args.get('leaf_weight', 0.1)
    
    IQL_LOSS_WEIGHT = 0.2
    Q_ONLY_LOSS_WEIGHT = 0.2

    print("=== ARDT Minimax Expectile Regression ===")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Minimax epochs: {minimax_epochs}")
    print(f"Alpha_max (optimistic): {alpha_max}")
    print(f"Alpha_min (pessimistic): {alpha_min}")

    # ---- Warmup (MSE + IQL/Bellman) ----
    for epoch in range(warmup_epochs):
        total_max_loss, total_min_loss, total_v_loss = 0, 0, 0
        total_q_loss, total_iql_loss = 0, 0
        num_batches = 0
        for obs, acts, adv_acts, ret, seq_len in tqdm(dataloader, desc=f"Warmup Epoch {epoch}"):
            batch_size, obs_len = obs.shape[0], obs.shape[1]
            obs = obs.view(batch_size, obs_len, -1).to(device)
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            ret_scaled = (ret / scale).to(device)
            
            timestep_mask = torch.arange(obs_len, device=device)[None, :] < seq_len[:, None]
            rewards = ret_scaled[:, :-1] - gamma * ret_scaled[:, 1:]
            rewards = rewards.unsqueeze(-1)
            
            # R_max
            max_pred = R_max_model(obs, acts)
            min_pred = R_min_model(obs, acts, adv_acts)
            v_pred = v_model(obs)

            # MSE losses for both Q-models
            max_loss_mse = (((max_pred - ret_scaled.unsqueeze(-1)) ** 2) * timestep_mask.unsqueeze(-1)).mean()
            min_loss_mse = (((min_pred - ret_scaled.unsqueeze(-1)) ** 2) * timestep_mask.unsqueeze(-1)).mean()
            
            # V-loss: V learns from Q
            v_loss_pr = expectile_loss(max_pred.detach().squeeze(-1) - v_pred.squeeze(-1), 0.7)
            v_loss_adv = expectile_loss(min_pred.detach().squeeze(-1) - v_pred.squeeze(-1), 0.5)
            v_loss = v_loss_pr.mean() + v_loss_adv.mean()
            
            # Q-losses
            iql_loss_pr = _iql_loss(v_pred, max_pred, rewards, timestep_mask, gamma, device)
            iql_loss_adv = _iql_loss(v_pred, min_pred, rewards, timestep_mask, gamma, device)
            q_only_loss_pr = _q_only_loss(max_pred, rewards, timestep_mask, gamma, device)
            q_only_loss_adv = _q_only_loss(min_pred, rewards, timestep_mask, gamma, device)
            
            # Combined Loss
            total_loss_epoch = (max_loss_mse + min_loss_mse + v_loss + 
                                IQL_LOSS_WEIGHT * (iql_loss_pr + iql_loss_adv) + 
                                Q_ONLY_LOSS_WEIGHT * (q_only_loss_pr + q_only_loss_adv))
            
            total_loss_epoch = torch.nan_to_num(total_loss_epoch, nan=0.0, posinf=0.0, neginf=0.0)
            total_loss_epoch.backward()
            
            max_optimizer.step()
            min_optimizer.step()
            value_optimizer.step()
            
            total_max_loss += max_loss_mse.item()
            total_min_loss += min_loss_mse.item()
            total_v_loss += v_loss.item()
            total_q_loss += q_only_loss_pr.item() + q_only_loss_adv.item()
            total_iql_loss += iql_loss_pr.item() + iql_loss_adv.item()
            num_batches += 1
        
        print(f"Warmup Epoch {epoch}: Max Loss = {total_max_loss/num_batches:.6f}, Min Loss = {total_min_loss/num_batches:.6f}, V Loss = {total_v_loss/num_batches:.6f}")
        evaluate_models(R_max_model, R_min_model, v_model, dataloader, device, scale)

    # ---- Minimax Expectile Regression ----
    for epoch in range(minimax_epochs):
        total_max_loss, total_min_loss, total_v_loss = 0, 0, 0
        total_q_loss, total_iql_loss = 0, 0
        num_batches = 0

        for obs, acts, adv_acts, ret, seq_len in tqdm(dataloader, desc=f"Minimax Epoch {epoch}"):
            batch_size, obs_len = obs.shape[0], obs.shape[1]
            obs = obs.view(batch_size, obs_len, -1).to(device)
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            ret_scaled = (ret / scale).to(device)
            
            timestep_mask = torch.arange(obs_len, device=device)[None, :] < seq_len[:, None]
            transition_mask = timestep_mask[:, :-1] & timestep_mask[:, 1:]
            
            rewards = ret_scaled[:, :-1] - gamma * ret_scaled[:, 1:]
            rewards = rewards.unsqueeze(-1)
            
            # --- Get predictions for all models ---
            max_pred = R_max_model(obs, acts)
            min_pred = R_min_model(obs, acts, adv_acts)
            v_pred = v_model(obs)

            # --- R_max update (Protagonist) ---
            max_optimizer.zero_grad()
            value_optimizer.zero_grad()
            
            # ARDT-style loss
            with torch.no_grad():
                min_next = R_min_model(obs[:, 1:], acts[:, 1:], adv_acts[:, 1:]).view(batch_size, obs_len - 1)
                targets = rewards[:, :-1] + gamma * min_next.unsqueeze(-1)
            
            max_loss_ardt = (expectile_loss(max_pred[:, :-1] - targets, alpha_max) * transition_mask.unsqueeze(-1)).mean()
            
            # V-loss for R_max
            v_loss_pr = expectile_loss(max_pred.detach().squeeze(-1) - v_pred.squeeze(-1), 0.7).mean()
            
            # Q-loss for R_max
            q_only_loss_pr = _q_only_loss(max_pred, rewards, timestep_mask, gamma, device)
            
            max_total_loss = max_loss_ardt + v_loss_pr + Q_ONLY_LOSS_WEIGHT * q_only_loss_pr
            max_total_loss = torch.nan_to_num(max_total_loss, nan=0.0, posinf=0.0, neginf=0.0)
            max_total_loss.backward()
            max_optimizer.step()
            value_optimizer.step()


            # --- R_min update (Adversary) ---
            min_optimizer.zero_grad()
            value_optimizer.zero_grad()

            # ARDT-style loss
            with torch.no_grad():
                max_next = R_max_model(obs[:, 1:], acts[:, 1:]).view(batch_size, obs_len - 1)
                targets = rewards[:, :-1] + gamma * max_next.unsqueeze(-1)

            min_loss_ardt = (expectile_loss(min_pred[:, :-1] - targets, alpha_min) * transition_mask.unsqueeze(-1)).mean()
            
            # Leaf loss
            terminal_indices = torch.arange(batch_size, device=device)
            valid_terminals = (seq_len > 0) & (seq_len <= obs_len)
            if valid_terminals.sum() > 0:
                terminal_seq_len = (seq_len[valid_terminals] - 1).clamp(0, obs_len - 1)
                leaf_loss = ((min_pred[valid_terminals, terminal_seq_len].squeeze(-1) - 
                              ret_scaled[valid_terminals, terminal_seq_len]) ** 2).mean()
            else:
                leaf_loss = torch.tensor(0.0, device=device)

            min_loss_ardt = (1 - leaf_weight) * min_loss_ardt + leaf_weight * leaf_loss

            # V-loss for R_min
            v_loss_adv = expectile_loss(min_pred.detach().squeeze(-1) - v_pred.squeeze(-1), 0.5).mean()
            
            # Q-loss for R_min
            q_only_loss_adv = _q_only_loss(min_pred, rewards, timestep_mask, gamma, device)
            iql_loss_adv = _iql_loss(v_pred, min_pred, rewards, timestep_mask, gamma, device)
            
            min_total_loss = min_loss_ardt + v_loss_adv + Q_ONLY_LOSS_WEIGHT * q_only_loss_adv + IQL_LOSS_WEIGHT * iql_loss_adv
            min_total_loss = torch.nan_to_num(min_total_loss, nan=0.0, posinf=0.0, neginf=0.0)
            min_total_loss.backward()
            min_optimizer.step()
            value_optimizer.step()

            total_max_loss += max_total_loss.item()
            total_min_loss += min_total_loss.item()
            total_v_loss += (v_loss_pr.item() + v_loss_adv.item())
            total_q_loss += (q_only_loss_pr.item() + q_only_loss_adv.item())
            total_iql_loss += iql_loss_adv.item()
            num_batches += 1

        print(f"Minimax Epoch {epoch}: Max Loss = {total_max_loss/num_batches:.6f}, Min Loss = {total_min_loss/num_batches:.6f}, V Loss = {total_v_loss/num_batches:.6f}")
        evaluate_models(R_max_model, R_min_model, v_model, dataloader, device, scale)

    return R_max_model, R_min_model


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
    
    print("=== ARDT Training ===")
    all_rewards = []
    for traj in trajs:
        episode_reward = np.sum(traj.rewards)
        all_rewards.append(episode_reward)
    
    print(f"Dataset size: {len(trajs)} episodes")
    print(f"Reward stats: mean={np.mean(all_rewards):.3f}, std={np.std(all_rewards):.3f}")
    
    # Setup scaling
    scale = train_args.get('scale', 1.0)
    if np.std(all_rewards) > 0:
        scale = max(1.0, np.std(all_rewards) / 1.0)
    print(f"Using scale: {scale:.2f}")
    
    # Setup action spaces
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
    
    # STEP 1: Minimax Expectile Regression (Core ARDT contribution)
    print("\n=== Step 1: Minimax Expectile Regression ===")
    R_max_model, R_min_model = ardt_minimax_expectile_regression(
        trajs, obs_size, action_size, adv_action_size, train_args, device, scale, is_simple_model
    )
    
    # STEP 2: Trajectory Relabeling with Minimax Returns-to-Go
    print("\n=== Step 2: Trajectory Relabeling ===")
    relabeled_trajs = []
    
    R_max_model.eval()
    R_min_model.eval()
    
    with torch.no_grad():
        for i, traj in enumerate(tqdm(trajs, desc="Relabeling trajectories")):
            # Prepare trajectory data
            obs = torch.from_numpy(np.array(traj.obs)).float().to(device).view(1, -1, obs_size)
            acts = torch.from_numpy(np.array(traj.actions)).to(device).view(1, -1)
            adv_acts = torch.from_numpy(np.array(traj.adv_actions)).to(device).view(1, -1)
            
            if action_type == "discrete" and not is_discretize:
                acts = acts.float().view(1, -1, action_size)
                adv_acts = adv_acts.float().view(1, -1, adv_action_size)
            else:
                acts = acts.view(1, -1, action_size)
                adv_acts = adv_acts.view(1, -1, adv_action_size)
            
            # Get minimax return predictions
            minimax_returns = R_max_model(obs, acts.float()).squeeze().cpu().numpy()
            
            # Scale back to original scale
            minimax_returns_scaled = minimax_returns * scale
            
            # Create new trajectory with relabeled returns-to-go
            relabeled_traj = deepcopy(traj)
            
            new_returns_to_go = []
            for t in range(len(traj.obs)):
                new_returns_to_go.append(minimax_returns_scaled[t])
            
            relabeled_traj.minimax_returns_to_go = new_returns_to_go
            relabeled_trajs.append(relabeled_traj)
    
    print(f"Relabeled {len(relabeled_trajs)} trajectories with minimax returns-to-go")
    
    # STEP 3: Train Decision Transformer on Relabeled Data
    print("\n=== Step 3: Decision Transformer Training ===")
    
    if is_simple_model:
        dt_model = RtgFFN(obs_size, action_size, include_adv=False).to(device)
    else:
        dt_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=False).to(device)
    
    dt_optimizer = torch.optim.AdamW(
        dt_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )
    
    max_len = max([len(traj.obs) for traj in relabeled_trajs]) + 1
    
    relabeled_dataset = ARDTDataset(
        relabeled_trajs, max_len, gamma=train_args['gamma'], act_type=action_type,
        use_minimax_returns=True
    )
    
    relabeled_dataloader = torch.utils.data.DataLoader(
        relabeled_dataset, batch_size=train_args['batch_size'], num_workers=n_cpu, shuffle=True
    )
    
    dt_epochs = train_args.get('dt_epochs', 100)
    dt_model.train()
    
    for epoch in range(dt_epochs):
        total_loss = 0
        num_batches = 0
        
        for obs, acts, _, minimax_ret, seq_len in tqdm(relabeled_dataloader, desc=f"DT Epoch {epoch}"):
            batch_size = obs.shape[0]
            obs_len = obs.shape[1]
            obs = obs.view(batch_size, obs_len, -1).to(device)
            acts = acts.to(device)
            minimax_ret_scaled = (minimax_ret / scale).to(device)
            
            timestep_mask = torch.arange(obs_len, device=device)[None, :] < seq_len[:, None]
            
            dt_optimizer.zero_grad()
            
            if action_type == 'discrete':
                action_preds = dt_model(obs, acts).view(batch_size, obs_len, -1)
                loss = F.cross_entropy(
                    action_preds[timestep_mask].view(-1, action_size),
                    acts[timestep_mask].view(-1).long(),
                    reduction='mean'
                )
            else:
                action_preds = dt_model(obs, acts).view(batch_size, obs_len, -1)
                loss = ((action_preds - acts) ** 2 * timestep_mask.unsqueeze(-1)).sum() / timestep_mask.sum()
            
            loss.backward()
            dt_optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        print(f"DT Epoch {epoch}: Loss = {total_loss/num_batches:.6f}")
    
    print("\n=== ARDT Evaluation ===")
    dt_model.eval()
    
    learned_returns = []
    prompt_value = -np.inf
    
    with torch.no_grad():
        for i, traj in enumerate(tqdm(trajs, desc="Evaluating")):
            obs = torch.from_numpy(np.array(traj.obs)).float().to(device).view(1, -1, obs_size)
            
            initial_minimax_return = relabeled_trajs[i].minimax_returns_to_go[0]
            
            if prompt_value < initial_minimax_return:
                prompt_value = initial_minimax_return
            
            # The returned `learned_returns` will be the minimax returns, not DT predictions.
            # The DT model is then used during inference, conditioned on these returns.
            learned_returns.append(relabeled_trajs[i].minimax_returns_to_go)
    
    print(f"ARDT training complete. Prompt value: {prompt_value:.3f}")
    
    return learned_returns, prompt_value