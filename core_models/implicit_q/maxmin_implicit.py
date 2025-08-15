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

# Loss weights - more balanced for effective training
IQL_LOSS_WEIGHT = 0.3
Q_ONLY_LOSS_WEIGHT = 0.05
V_LOSS_WEIGHT = 0.1

def compute_value_from_q(q_values, value_type="stochastic", temperature=1.0):
    """
    Prevents V_adv explosion by deriving V from Q directly.
    """
    if len(q_values.shape) == 2:
        return q_values
    
    if len(q_values.shape) == 3 and q_values.shape[-1] > 1:
        if value_type == "deterministic":
            v_values, _ = q_values.max(dim=-1)
            return v_values
        else:
            weights = F.softmax(q_values / temperature, dim=-1)
            v_values = (q_values * weights).sum(dim=-1)
            return v_values
    else:
        return q_values.squeeze(-1) if len(q_values.shape) == 3 else q_values

def _iql_loss_v_expectile(q_values, v_values, acts, timestep_mask, expectile=0.7):
    """
    V loss using expectile regression.
    """
    batch_size, obs_len = q_values.shape[0], q_values.shape[1]
    
    if len(q_values.shape) == 3:
        q_flat = (q_values * acts).sum(dim=-1)
    else:
        q_flat = q_values
        
    if len(v_values.shape) == 3:
        v_flat = (v_values * acts).sum(dim=-1)
    else:
        v_flat = v_values
    
    valid_mask = timestep_mask.float()
    
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=q_values.device, requires_grad=True)
    
    diff = q_flat.detach() - v_flat
    
    loss = torch.where(
        diff >= 0,
        expectile * (diff ** 2),
        (1 - expectile) * (diff ** 2)
    )
    
    masked_loss = loss * valid_mask
    final_loss = masked_loss.sum() / valid_mask.sum()
    clamped_loss = torch.clamp(final_loss, min=1e-8, max=100.0)
    
    return clamped_loss

def _iql_loss_q(q_values, immediate_rewards, v_values, acts, transition_mask, gamma):
    """Standard IQL Q loss for protagonist Q-function"""
    if transition_mask is None or transition_mask.sum() == 0:
        return torch.tensor(0.0, device=q_values.device, requires_grad=True)
    
    batch_size, obs_len = q_values.shape[0], q_values.shape[1]
    
    if len(q_values.shape) == 3:
        action_indices = torch.argmax(acts, dim=-1, keepdim=True)
        q_flat = torch.gather(q_values, dim=-1, index=action_indices).squeeze(-1)
    else:
        q_flat = q_values

    # FIX: Define v_flat before it's used
    if len(v_values.shape) == 3:
        # Use the same logic as q_flat for consistency and correctness
        action_indices = torch.argmax(acts, dim=-1, keepdim=True)
        v_flat = torch.gather(v_values, dim=-1, index=action_indices).squeeze(-1)
    else:
        v_flat = v_values

    q_pred = q_flat[:, :-1]
    v_next = v_flat[:, 1:].detach()
    q_target_raw = immediate_rewards + gamma * v_next
    q_target = torch.clamp(q_target_raw, -1e6, 1e6)
    loss_raw = ((q_pred - q_target) ** 2) * transition_mask
    loss = loss_raw.sum() / transition_mask.sum()
    clamped_loss = torch.clamp(loss, 0.0, 1e6)
    
    return clamped_loss

def _iql_loss_adv_q(q_values, immediate_rewards, v_values, adv_acts, transition_mask, gamma):
    if transition_mask is None or transition_mask.sum() == 0:
        return torch.tensor(0.0, device=q_values.device, requires_grad=True)

    batch_size, obs_len = q_values.shape[0], q_values.shape[1]

    if len(q_values.shape) == 3:
        # Get the Q-value corresponding to the adversary's action
        action_indices = torch.argmax(adv_acts, dim=-1, keepdim=True)
        q_flat = torch.gather(q_values, dim=-1, index=action_indices).squeeze(-1)
    else:
        q_flat = q_values


    if len(v_values.shape) == 3:
        # Get the V-value corresponding to the adversary's action
        action_indices = torch.argmax(adv_acts, dim=-1, keepdim=True)
        v_flat = torch.gather(v_values, dim=-1, index=action_indices).squeeze(-1)
    else:
        v_flat = v_values
        
    q_pred = q_flat[:, :-1]
    v_next = v_flat[:, 1:].detach()
    q_target_raw = immediate_rewards + gamma * v_next
    q_target = torch.clamp(q_target_raw, -1e6, 1e6)
    loss_raw = ((q_pred - q_target) ** 2) * transition_mask
    loss = loss_raw.sum() / transition_mask.sum()

    return torch.clamp(loss, 0.0, 1e6)

def _q_only_loss(q_values, immediate_rewards, transition_mask, gamma, use_max=True):
    if transition_mask is None or transition_mask.sum() == 0:
        return torch.tensor(0.0, device=q_values.device, requires_grad=True)
    
    batch_size, obs_len = q_values.shape[0], q_values.shape[1]
    
    if len(q_values.shape) == 3 and q_values.shape[-1] > 1:
        if use_max:
            q_pred = q_values[:, :-1].max(dim=-1)[0]
            q_next = q_values[:, 1:].detach().max(dim=-1)[0]
        else:
            q_pred = q_values[:, :-1].min(dim=-1)[0]
            q_next = q_values[:, 1:].detach().min(dim=-1)[0]
    else:
        q_flat = q_values.view(batch_size, obs_len)
        q_pred = q_flat[:, :-1]
        q_next = q_flat[:, 1:].detach()
    
    q_target = immediate_rewards + gamma * q_next
    q_target = torch.clamp(q_target, -1e6, 1e6)
    
    loss = (((q_pred - q_target) ** 2) * transition_mask).sum() / transition_mask.sum()
    
    return torch.clamp(loss, 0.0, 1e6)

def evaluate_models(qsa_pr_model, qsa_adv_model, dataloader, device):
    """Evaluate models and check proper minimax ordering, using ValueNet for both V_pr and V_adv."""
    with torch.no_grad():
        obs, acts, adv_acts, ret, seq_len = next(iter(dataloader))
        batch_size, obs_len = obs.shape[0], obs.shape[1]
        
        obs = obs.view(batch_size, obs_len, -1).to(device)
        acts = acts.to(device)
        adv_acts = adv_acts.to(device)
        ret = ret.to(device)
        
        timestep_mask = torch.arange(obs_len, device=device)[None, :] < seq_len[:, None]
        valid_mask = timestep_mask.float()
        
        pred_pr_full = qsa_pr_model(obs, acts)
        pred_adv_full = qsa_adv_model(obs, acts, adv_acts)
        
        pred_pr = pred_pr_full.max(dim=-1)[0] if len(pred_pr_full.shape) == 3 else pred_pr_full
        pred_adv = pred_adv_full.min(dim=-1)[0] if len(pred_adv_full.shape) == 3 else pred_adv_full

        pred_v_pr = compute_value_from_q(pred_pr_full, value_type="deterministic") 
        pred_v_adv = compute_value_from_q(pred_adv_full, value_type="stochastic") 
        
        print("   Using computed V from Q for both protagonist and adversary (ValueNet)")
        
        if valid_mask.sum() > 0:
            pred_pr_mean = (pred_pr * valid_mask).sum() / valid_mask.sum()
            pred_adv_mean = (pred_adv * valid_mask).sum() / valid_mask.sum()
            pred_v_pr_mean = (pred_v_pr * valid_mask).sum() / valid_mask.sum()
            pred_v_adv_mean = (pred_v_adv * valid_mask).sum() / valid_mask.sum()
            true_mean = (ret * valid_mask).sum() / valid_mask.sum()
        else:
            pred_pr_mean = pred_pr.mean()
            pred_adv_mean = pred_adv.mean()
            pred_v_pr_mean = pred_v_pr.mean()
            pred_v_adv_mean = pred_v_adv.mean()
            true_mean = ret.mean()

    gap_pr_v_pr = pred_pr_mean.item() - pred_v_pr_mean.item()
    gap_v_adv_adv = pred_v_adv_mean.item() - pred_adv_mean.item()
    gap_pr_adv = pred_pr_mean.item() - pred_adv_mean.item()
    
    symbol_pr_v_pr = "✅" if gap_pr_v_pr >= -0.01 else "❌"
    symbol_v_adv_adv = "✅" if gap_v_adv_adv >= -0.01 else "❌"
    symbol_pr_adv = "✅" if gap_pr_adv >= -0.01 else "❌"
    
    print(f"   Eval -> True: {true_mean.item():.4f}")
    print(f"         Q_pr: {pred_pr_mean.item():.4f}, V_pr: {pred_v_pr_mean.item():.4f}")
    print(f"         Q_adv: {pred_adv_mean.item():.4f}, V_adv: {pred_v_adv_mean.item():.4f}")
    print(f"         Q_pr-V_pr: {gap_pr_v_pr:.4f} {symbol_pr_v_pr}, V_adv-Q_adv: {gap_v_adv_adv:.4f} {symbol_v_adv_adv}, Q_pr-Q_adv: {gap_pr_adv:.4f} {symbol_pr_adv}")
    
    return pred_pr_mean.item(), pred_adv_mean.item(), pred_v_pr_mean.item(), pred_v_adv_mean.item()

def maxmin(
    trajs: list[Trajectory],
    action_space: gym.spaces,
    adv_action_space: gym.spaces,
    train_args: dict,
    device: str,
    n_cpu: int,
    is_simple_model: bool = True,
    is_toy: bool = False,
    is_discretize: bool = False,
) -> tuple[np.ndarray, float]:
    """
    Fixed Minimax training with ValueNet for both V_pr and V_adv
    """
    print("=== Fixed Minimax Training: ValueNet for V_pr and V_adv ===")
    
    value_type = train_args.get('value_type', 'stochastic')
    temperature = train_args.get('temperature', 1.0)
    print(f"Using ValueNet for V: type={value_type}, temp={temperature}")

    mse_epochs = train_args.get('mse_epochs', 5)
    maxmin_epochs = train_args.get('maxmin_epochs', 10)
    total_epochs = mse_epochs + maxmin_epochs
    gamma=train_args['gamma']

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

    print(f"Model dimensions: obs={obs_size}, act={action_size}, adv_act={adv_action_size}")

    max_len = max([len(traj.obs) for traj in trajs]) + 1
    
    dataset = ARDTDataset(trajs, max_len, gamma=gamma, act_type=action_type)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_args['batch_size'], num_workers=n_cpu
    )
    
    # Get a batch to determine the scaling factor
    obs, acts, adv_acts, ret, seq_len = next(iter(dataloader))
    
    # Calculate scale factor from the initial returns-to-go
    max_return = torch.max(torch.abs(ret)).item()
    
    # Avoid division by zero
    scale_factor = max_return if max_return > 0 else 1.0

    ret = ret / scale_factor
    
    print(f"Computed de-scaling factor from dataset: {scale_factor:.4f}")

    print(f'Creating models (simple={is_simple_model})...')
    if is_simple_model:
        qsa_pr_model = RtgFFN(obs_size, action_size, adv_action_size, include_adv=False).to(device)
        qsa_adv_model = RtgFFN(obs_size, action_size, adv_action_size, include_adv=True).to(device)
    else:
        qsa_pr_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=False).to(device)
        qsa_adv_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)

    base_lr = train_args['model_lr'] * 0.2
    qsa_pr_optimizer = torch.optim.AdamW(
        qsa_pr_model.parameters(), lr=base_lr, weight_decay=train_args.get('model_wd', 1e-4)
    )
    qsa_adv_optimizer = torch.optim.AdamW(
        qsa_adv_model.parameters(), lr=base_lr, weight_decay=train_args.get('model_wd', 1e-4)
    )
    
    pr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(qsa_pr_optimizer, T_max=total_epochs)
    adv_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(qsa_adv_optimizer, T_max=total_epochs)

    print(f"Training schedule: {mse_epochs} MSE + {maxmin_epochs} Fixed IQL = {total_epochs} epochs")
    print("Using ValueNet for both V_pr and V_adv!")

    for epoch in range(total_epochs):
        qsa_pr_model.train()
        qsa_adv_model.train()
        
        pbar = tqdm(dataloader, total=len(dataloader))
        epoch_loss = 0
        epoch_pr_loss = 0
        epoch_adv_loss = 0
        epoch_v_pr_loss = 0
        epoch_v_adv_loss = 0
        epoch_iql_loss = 0
        epoch_q_only_loss = 0
        n_batches = 0

        for batch_idx, (obs, acts, adv_acts, ret, seq_len) in enumerate(pbar):
            n_batches += 1
            
            if is_toy:
                obs, acts, adv_acts, ret = (
                    obs[:, :-1], acts[:, :-1], adv_acts[:, :-1], ret[:, :-1]
                )
                seq_len = torch.clamp(seq_len - 1, min=1)

            seq_len = torch.clamp(seq_len, max=obs.shape[1])
            batch_size, obs_len = obs.shape[0], obs.shape[1]
            
            obs = obs.view(batch_size, obs_len, -1).to(device)
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            ret = ret.to(device)
            
            timestep_mask = torch.arange(obs_len, device=device)[None, :] < seq_len[:, None]
            transition_mask = timestep_mask[:, :-1] & timestep_mask[:, 1:] if obs_len > 1 else None
            
            if obs_len > 1:
                immediate_rewards = ret[:, :-1] - gamma * ret[:, 1:]
                immediate_rewards = torch.clamp(immediate_rewards, -1e6, 1e6)
            else:
                immediate_rewards = torch.zeros(batch_size, 0, device=device)

            ret_pr_pred = qsa_pr_model(obs, acts)
            ret_adv_pred = qsa_adv_model(obs, acts, adv_acts)
            
            if len(ret_pr_pred.shape) == 3:
                ret_pr_pred_flat = (ret_pr_pred * acts).sum(dim=-1)
            else:
                ret_pr_pred_flat = ret_pr_pred
            
            if len(ret_adv_pred.shape) == 3:
                ret_adv_pred_flat = (ret_adv_pred * adv_acts).sum(dim=-1)
            else:
                ret_adv_pred_flat = ret_adv_pred
                
            ret_pr_pred_flat = torch.clamp(ret_pr_pred_flat, -1e6, 1e6)
            ret_adv_pred_flat = torch.clamp(ret_adv_pred_flat, -1e6, 1e6)

            if epoch < mse_epochs:
                if timestep_mask.sum() > 0:
                    pr_loss = (((ret_pr_pred_flat - ret) ** 2) * timestep_mask.float()).sum() / timestep_mask.sum()
                    adv_loss = (((ret_adv_pred_flat - ret) ** 2) * timestep_mask.float()).sum() / timestep_mask.sum()
                else:
                    pr_loss = adv_loss = torch.tensor(0.0, device=device, requires_grad=True)

                qsa_pr_optimizer.zero_grad()
                if pr_loss.requires_grad:
                    pr_loss.backward()
                    torch.nn.utils.clip_grad_norm_(qsa_pr_model.parameters(), max_norm=0.5)
                    qsa_pr_optimizer.step()
                
                qsa_adv_optimizer.zero_grad()
                if adv_loss.requires_grad:
                    adv_loss.backward()
                    torch.nn.utils.clip_grad_norm_(qsa_adv_model.parameters(), max_norm=0.5)
                    qsa_adv_optimizer.step()
                
                total_loss = pr_loss + adv_loss
                epoch_pr_loss += pr_loss.item()
                epoch_adv_loss += adv_loss.item()
            else:
                v_pr_pred = compute_value_from_q(ret_pr_pred, value_type="deterministic")
                v_adv_pred = compute_value_from_q(ret_adv_pred, value_type="stochastic")
                
                v_pr_loss = _iql_loss_v_expectile(ret_pr_pred_flat, v_pr_pred, acts, timestep_mask, expectile=0.8)
                v_adv_loss = _iql_loss_v_expectile(ret_adv_pred_flat, v_adv_pred, adv_acts, timestep_mask, expectile=0.2)
                
                qsa_pr_optimizer.zero_grad()
                iql_loss_pr = _iql_loss_q(ret_pr_pred_flat, immediate_rewards, v_pr_pred, acts, transition_mask, gamma)
                q_only_loss_pr = _q_only_loss(ret_pr_pred, immediate_rewards, transition_mask, gamma, use_max=True)
                pr_total_loss = IQL_LOSS_WEIGHT * iql_loss_pr + Q_ONLY_LOSS_WEIGHT * q_only_loss_pr
                
                if pr_total_loss.requires_grad:
                    pr_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(qsa_pr_model.parameters(), max_norm=1.0)
                    qsa_pr_optimizer.step()
                
                qsa_adv_optimizer.zero_grad()
                iql_loss_adv = _iql_loss_adv_q(ret_adv_pred_flat, immediate_rewards, v_adv_pred, adv_acts, transition_mask, gamma)
                q_only_loss_adv = _q_only_loss(ret_adv_pred, immediate_rewards, transition_mask, gamma, use_max=False)
                adv_total_loss = IQL_LOSS_WEIGHT * iql_loss_adv + Q_ONLY_LOSS_WEIGHT * q_only_loss_adv
                
                if adv_total_loss.requires_grad:
                    adv_total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(qsa_adv_model.parameters(), max_norm=1.0)
                    qsa_adv_optimizer.step()
                
                total_loss = V_LOSS_WEIGHT * v_pr_loss + V_LOSS_WEIGHT * v_adv_loss + pr_total_loss + adv_total_loss
                
                epoch_v_pr_loss += v_pr_loss.item()
                epoch_v_adv_loss += v_adv_loss.item()
                epoch_iql_loss += (iql_loss_pr + iql_loss_adv).item()
                epoch_q_only_loss += (q_only_loss_pr + q_only_loss_adv).item()
            
            epoch_loss += total_loss.item()
            
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"❌ FATAL: NaN/Inf detected in total_loss at epoch {epoch}, batch {batch_idx}")
                print(f"   Total loss: {total_loss.item()}")
                raise RuntimeError("Training stopped due to NaN/Inf in loss")
            
            phase = "MSE" if epoch < mse_epochs else "IMPLICIT-Q"
            pbar.set_description(f"Epoch {epoch} ({phase}) | "
                               f"Loss: {epoch_loss/n_batches:.6f} | "
                               f"V_pr: {epoch_v_pr_loss/n_batches:.6f} | "
                               f"V_adv: {epoch_v_adv_loss/n_batches:.6f} | "
                               f"IQL: {epoch_iql_loss/n_batches:.6f}")

        pr_scheduler.step()
        adv_scheduler.step() 
        
        print(f"Epoch {epoch} completed - Average Loss: {epoch_loss/n_batches:.6f}, LR: {pr_scheduler.get_last_lr()[0]:.6f}")
        
        if epoch >= mse_epochs and epoch_loss/n_batches > 100:
            print(f"⚠️  WARNING: Loss explosion detected (avg loss: {epoch_loss/n_batches:.3f})")
            print("   Stopping training early to prevent further instability")
            break
        
        if epoch % 5 == 0 or epoch >= total_epochs - 3:
            phase_name = "MSE" if epoch < mse_epochs else "IMPLICIT-Q"
            print(f"{phase_name} Epoch {epoch}:")
            evaluate_models(qsa_pr_model, qsa_adv_model, dataloader, device)

    print("\n=== Trajectory Relabeling ===")
    qsa_pr_model.eval()
    relabeled_trajs = []
    prompt_value = -np.inf

    with torch.no_grad():
        for traj in tqdm(trajs, desc="Relabeling trajectories"):
            obs_tensor = torch.from_numpy(np.array(traj.obs)).float().to(device).unsqueeze(0)
            acts_tensor = torch.from_numpy(np.array(traj.actions)).float().to(device).unsqueeze(0)

            # Get model predictions
            returns_pred = qsa_pr_model(obs_tensor, acts_tensor)

            # Use torch.gather to get the Q-value for the action taken
            if len(returns_pred.shape) == 3 and returns_pred.shape[-1] > 1:
                action_indices = torch.argmax(acts_tensor, dim=-1, keepdim=True)
                returns_values = torch.gather(returns_pred, dim=-1, index=action_indices).squeeze(-1).squeeze(0)
            else:
                returns_values = returns_pred.squeeze(0)
                
            # De-scale the returns to the original reward scale
            returns = (returns_values / scale_factor).cpu().numpy()
            
            if len(returns) > 0 and prompt_value < returns[0]:
                prompt_value = returns[0]
                
            relabeled_traj = deepcopy(traj)
            relabeled_traj.minimax_returns_to_go = returns.tolist()
            relabeled_trajs.append(relabeled_traj)


    print(f"Relabeling complete. Prompt value: {prompt_value:.3f}")
    
    return relabeled_trajs, np.round(prompt_value, decimals=3)
