import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from copy import deepcopy

from data_loading.load_mujoco import Trajectory
from core_models.base_models.base_model import RtgFFN, RtgLSTM
from core_models.dataset.ardt_dataset import ARDTDataset


def approx_ce_batch(q_values: torch.Tensor, tau: float) -> torch.Tensor:
    """Approximate correlated equilibrium using softmax with temperature tau."""
    logits = q_values / tau
    flat_logits = logits.view(*logits.shape[:-2], -1)
    probs = torch.softmax(flat_logits, dim=-1)
    return probs.view_as(q_values)


def incentive_violation_loss_zero_sum(policy: torch.Tensor, q_values: torch.Tensor, valid_mask: torch.Tensor = None) -> torch.Tensor:
    """Incentive compatibility loss for zero-sum games with proper masking."""
    B, T, A1, A2 = q_values.shape
    pi = policy.view(B, T, A1, A2)
    expected_value = (pi * q_values).sum(dim=[2, 3], keepdim=True)

    a_marg = pi.sum(dim=3, keepdim=True)  # marginal over a2
    adv_marg = pi.sum(dim=2, keepdim=True)  # marginal over a1

    q_diff_1 = expected_value - q_values  # protagonist deviation
    q_diff_2 = q_values - expected_value  # adversary deviation (since -Q)

    viol_1 = F.relu(q_diff_1 * a_marg)
    viol_2 = F.relu(q_diff_2 * adv_marg)
    
    # Apply valid mask if provided
    if valid_mask is not None:
        valid_mask_expanded = valid_mask.unsqueeze(-1).unsqueeze(-1)  # [B, T, 1, 1]
        viol_1 = viol_1 * valid_mask_expanded
        viol_2 = viol_2 * valid_mask_expanded
        
        if valid_mask.sum() > 0:
            return (viol_1.sum() + viol_2.sum()) / valid_mask.sum()
        else:
            return torch.tensor(0.0, device=q_values.device, requires_grad=True)

    return viol_1.mean() + viol_2.mean()


def safe_action_extraction(q_values, acts, adv_acts=None):
    """Safely extract Q-values based on actions, handling invalid actions."""
    batch_size, seq_len = q_values.shape[:2]
    
    if len(q_values.shape) == 3:
        # Handle single action dimension (either protagonist or adversary)
        valid_actions = acts.sum(dim=-1) > 0  # [batch_size, seq_len]
        q_flat = (q_values * acts).sum(dim=-1)
        q_flat = q_flat * valid_actions.float()
        return q_flat, valid_actions
    elif len(q_values.shape) == 4:
        # Handle joint action space [B, T, A1, A2]
        valid_acts = acts.sum(dim=-1) > 0  # [batch_size, seq_len]
        valid_adv_acts = adv_acts.sum(dim=-1) > 0 if adv_acts is not None else valid_acts
        valid_joint = valid_acts & valid_adv_acts
        
        # Extract using outer product of actions
        acts_expanded = acts.unsqueeze(-1)  # [B, T, A1, 1]
        adv_acts_expanded = adv_acts.unsqueeze(-2) if adv_acts is not None else acts.unsqueeze(-2)  # [B, T, 1, A2]
        joint_actions = acts_expanded * adv_acts_expanded  # [B, T, A1, A2]
        
        q_flat = (q_values * joint_actions).sum(dim=[-2, -1])
        q_flat = q_flat * valid_joint.float()
        return q_flat, valid_joint
    else:
        # Already flat
        return q_values, torch.ones(batch_size, seq_len, device=q_values.device, dtype=torch.bool)


def correlated_q_learning(
    trajs: list[Trajectory],
    obs_size: int,
    action_size: int,
    adv_action_size: int,
    train_args: dict,
    device: str,
    is_simple_model: bool = False,
    action_type: str = 'discrete'
) -> torch.nn.Module:
    """Train Q-function using correlated Q-learning with CE policies."""

    print("=== Correlated Q-Learning Training (Fixed Action Handling) ===")

    # Initialize Q-model
    if is_simple_model:
        q_model = RtgFFN(obs_size, action_size, adv_action_size, include_adv=True).to(device)
    else:
        q_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)

    # Use same learning rate pattern as Implicit Q
    base_lr = train_args['model_lr'] * 0.2
    optimizer = torch.optim.AdamW(
        q_model.parameters(),
        lr=base_lr,
        weight_decay=train_args.get('model_wd', 1e-4)
    )

    # Dataset preparation
    max_len = max([len(traj.obs) for traj in trajs]) + 1
    dataset = ARDTDataset(trajs, max_len, gamma=train_args['gamma'], act_type=action_type)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_args['batch_size']
    )

    # Get scaling factor from dataset like Implicit Q
    obs, acts, adv_acts, ret, seq_len = next(iter(dataloader))
    max_return = torch.max(torch.abs(ret)).item()
    scale_factor = max_return if max_return > 0 else 1.0
    print(f"Computed scaling factor from dataset: {scale_factor:.4f}")

    # Training hyperparameters - match Implicit Q structure
    mse_epochs = train_args.get('mse_epochs', 5)
    ce_epochs = train_args.get('maxmin_epochs', 10)  # Use same as Implicit Q
    total_epochs = mse_epochs + ce_epochs
    tau = train_args.get('ce_temperature', 1.0)
    lambda_ce = train_args.get('lambda_ce', 0.3)  # Match IQL loss weight

    # Add scheduler like Implicit Q
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_epochs)

    print(f"MSE warmup epochs: {mse_epochs}")
    print(f"CE-Q learning epochs: {ce_epochs}")
    print(f"CE temperature (tau): {tau}")
    print(f"CE loss weight (lambda): {lambda_ce}")

    q_model.train()

    for epoch in range(total_epochs):
        total_q_loss = 0
        total_ce_loss = 0
        num_batches = 0

        phase = "MSE Warmup" if epoch < mse_epochs else "CE-Q Learning"
        pbar = tqdm(dataloader, desc=f"Epoch {epoch} ({phase})")

        for obs, acts, adv_acts, ret, seq_len in pbar:
            num_batches += 1
            
            # Match Implicit Q data processing
            seq_len = torch.clamp(seq_len, max=obs.shape[1])
            batch_size, obs_len = obs.shape[:2]
            obs = obs.view(batch_size, obs_len, -1).to(device)
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            ret = ret.to(device)

            # Create masks like Implicit Q
            timestep_mask = torch.arange(obs_len, device=device)[None, :] < seq_len[:, None]
            transition_mask = timestep_mask[:, :-1] & timestep_mask[:, 1:] if obs_len > 1 else None

            # Compute rewards from returns-to-go like Implicit Q
            if obs_len > 1:
                immediate_rewards = ret[:, :-1] - train_args['gamma'] * ret[:, 1:]
                immediate_rewards = torch.clamp(immediate_rewards, -1e6, 1e6)
            else:
                immediate_rewards = torch.zeros(batch_size, 0, device=device)

            # Forward pass to get Q-values for the entire trajectory
            q_all = q_model(obs, acts, adv_acts)

            if epoch < mse_epochs:
                # MSE warmup phase: predict return-to-go directly like Implicit Q
                q_pred_flat, valid_actions = safe_action_extraction(q_all, acts, adv_acts)
                q_pred_flat = torch.clamp(q_pred_flat, -1e6, 1e6)
                
                # Apply valid action mask to timestep mask
                effective_mask = timestep_mask.float() * valid_actions.float()
                
                if effective_mask.sum() > 0:
                    mse_loss = (((q_pred_flat - ret) ** 2) * effective_mask).sum() / effective_mask.sum()
                else:
                    mse_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                ce_loss = torch.tensor(0.0, device=device)
                total_loss = mse_loss
                
                # Debug info for MSE phase
                if num_batches % 200 == 0:
                    print(f"\n   MSE Debug - Valid actions: {valid_actions.sum().item()}/{valid_actions.numel()}")
                    print(f"   Q pred range: [{q_pred_flat.min().item():.3f}, {q_pred_flat.max().item():.3f}]")
                    print(f"   Return range: [{ret.min().item():.3f}, {ret.max().item():.3f}]")
                
            else:
                # CE-Q learning phase with proper next state handling
                if transition_mask is None or transition_mask.sum() == 0:
                    continue
                    
                # Get current Q values using slicing
                q_current_raw = q_all[:, :-1]
                q_next_raw = q_all[:, 1:]

                # Get current flat Q value using the action taken
                q_current_flat, valid_current = safe_action_extraction(q_all, acts, adv_acts)
                q_pred = q_current_flat[:, :-1]

                with torch.no_grad():
                    # Compute next Q values
                    if len(q_next_raw.shape) == 4:
                        # Joint action space - use CE policy
                        A1, A2 = acts.shape[-1], adv_acts.shape[-1]
                        
                        # Reshape to [B, T, A1, A2]
                        q_next_reshaped = q_next_raw.view(batch_size, obs_len - 1, A1, A2)
                        
                        # Compute CE policy
                        ce_policy = approx_ce_batch(q_next_reshaped, tau)
                        
                        # Compute next value from CE policy
                        next_value = (ce_policy * q_next_reshaped).sum(dim=[2, 3])

                    else:
                        # Single action space - use max for protagonist, min for adversary
                        # This would be more complex in a full minimax setup.
                        # For now, we'll assume a greedy protagonist.
                        next_value = q_next_raw.max(dim=-1)[0]
                
                # Bellman backup
                q_target = immediate_rewards + train_args['gamma'] * next_value.detach()
                q_target = torch.clamp(q_target, -1e6, 1e6)
                
                # Q loss with proper masking
                effective_transition_mask = transition_mask.float() * valid_current[:, :-1].float()
                
                if effective_transition_mask.sum() > 0:
                    mse_loss = (((q_pred - q_target) ** 2) * effective_transition_mask).sum() / effective_transition_mask.sum()
                else:
                    mse_loss = torch.tensor(0.0, device=device, requires_grad=True)

                # CE loss if we have joint action space
                if len(q_next_raw.shape) == 4:
                    ce_loss = incentive_violation_loss_zero_sum(
                        ce_policy, q_next_reshaped, valid_mask=effective_transition_mask
                    )
                else:
                    ce_loss = torch.tensor(0.0, device=device)
                
                total_loss = mse_loss + lambda_ce * ce_loss
                
                # Debug info for CE phase
                if num_batches % 200 == 0:
                    print(f"\n   CE Debug - Valid transitions: {effective_transition_mask.sum().item()}")
                    print(f"   Q current range: [{q_pred.min().item():.3f}, {q_pred.max().item():.3f}]")
                    print(f"   Q target range: [{q_target.min().item():.3f}, {q_target.max().item():.3f}]")
                    if len(q_next_raw.shape) == 4:
                        print(f"   CE policy entropy: {(-ce_policy * torch.log(ce_policy + 1e-8)).sum(-1).sum(-1).mean().item():.3f}")

            # Check for NaN/Inf like Implicit Q
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"❌ FATAL: NaN/Inf detected in total_loss at epoch {epoch}")
                print(f"   MSE loss: {mse_loss.item()}, CE loss: {ce_loss.item()}")
                raise RuntimeError("Training stopped due to NaN/Inf in loss")

            optimizer.zero_grad()
            if total_loss.requires_grad:
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(q_model.parameters(), max_norm=1.0)
                optimizer.step()

            total_q_loss += mse_loss.item()
            total_ce_loss += ce_loss.item()
            
            # Update progress bar like Implicit Q
            pbar.set_description(f"Epoch {epoch} ({phase}) | "
                               f"Q Loss: {total_q_loss/num_batches:.6f} | "
                               f"CE Loss: {total_ce_loss/num_batches:.6f}")

        # Step scheduler like Implicit Q
        scheduler.step()
        
        avg_q_loss = total_q_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        print(f"Epoch {epoch} ({phase}): Q Loss = {avg_q_loss:.6f}, CE Loss = {avg_ce_loss:.6f}, LR = {scheduler.get_last_lr()[0]:.6f}")
        
        # Early stopping like Implicit Q
        if epoch >= mse_epochs and avg_q_loss > 100:
            print(f"⚠️  WARNING: Loss explosion detected (avg loss: {avg_q_loss:.3f})")
            print("   Stopping training early to prevent further instability")
            break

    return q_model, scale_factor


def evaluate_q_model(q_model, trajs, obs_size, action_size, adv_action_size,
                     action_type, device, scale_factor, is_discretize=False):
    """Evaluate Q-model on trajectories and compute statistics."""
    q_model.eval()
    all_q_preds = []
    all_true_returns = []
    traj_q_vals = []

    with torch.no_grad():
        for i, traj in enumerate(tqdm(trajs, desc="Evaluating Q-model")):
            obs = torch.tensor(traj.obs, dtype=torch.float32, device=device).unsqueeze(0)
            acts = torch.tensor(traj.actions, dtype=torch.float32, device=device)
            adv_acts = torch.tensor(getattr(traj, 'adv_actions', traj.actions), dtype=torch.float32, device=device)

            if action_type == "discrete" and not is_discretize:
                if acts.dim() == 1 or (acts.dim() == 2 and acts.shape[-1] != action_size):
                    acts = F.one_hot(acts.long().squeeze(), num_classes=action_size).float()
                    adv_acts = F.one_hot(adv_acts.long().squeeze(), num_classes=adv_action_size).float()
                if acts.dim() == 2:
                    acts = acts.unsqueeze(0)
                    adv_acts = adv_acts.unsqueeze(0)
            else:
                try:
                    acts = acts.float().view(1, -1, action_size)
                    adv_acts = adv_acts.float().view(1, -1, adv_action_size)
                except Exception as e:
                    continue

            try:
                q_preds_raw = q_model(obs, acts, adv_acts)
                
                # Handle action selection with proper masking
                q_preds_flat, valid_actions = safe_action_extraction(q_preds_raw, acts, adv_acts)
                
                # Only use valid predictions
                q_preds = q_preds_flat.cpu().flatten().numpy()
                valid_mask = valid_actions.cpu().flatten().numpy()
                
                # De-scale like Implicit Q to match original reward scale
                q_preds = q_preds / scale_factor
                
                true_returns = np.array([np.sum(traj.rewards[t:]) for t in range(len(traj.rewards))])

                length = min(len(q_preds), len(true_returns), len(valid_mask))
                q_preds = q_preds[:length]
                true_returns = true_returns[:length]
                valid_mask = valid_mask[:length]
                
                # Only include valid predictions
                valid_indices = valid_mask.astype(bool)
                if valid_indices.sum() > 0:
                    q_preds_valid = q_preds[valid_indices]
                    true_returns_valid = true_returns[valid_indices]
                    
                    all_q_preds.extend(q_preds_valid)
                    all_true_returns.extend(true_returns_valid)
                    traj_q_vals.append(q_preds)
                else:
                    traj_q_vals.append(np.array([]))
                    
            except Exception as e:
                print(f"Error processing trajectory {i}: {e}")
                traj_q_vals.append(np.array([]))
                continue

    all_q_preds = np.array(all_q_preds)
    all_true_returns = np.array(all_true_returns)

    if len(all_q_preds) > 0 and len(all_true_returns) > 0:
        mse = np.mean((all_q_preds - all_true_returns) ** 2)
        mae = np.mean(np.abs(all_q_preds - all_true_returns))
        correlation = np.corrcoef(all_q_preds, all_true_returns)[0, 1] if len(all_true_returns) > 1 else 0.0
    else:
        mse = mae = correlation = 0.0

    initial_qs = [q[0] if len(q) > 0 else 0 for q in traj_q_vals]
    max_init_q = np.max(initial_qs) if initial_qs else 0.0
    mean_init_q = np.mean(initial_qs) if initial_qs else 0.0

    print(f"\n=== Q-Model Evaluation Results ===")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, Corr: {correlation:.4f}")
    print(f"Q-value mean: {np.mean(all_q_preds):.3f}, std: {np.std(all_q_preds):.3f}")
    print(f"Return mean: {np.mean(all_true_returns):.3f}, std: {np.std(all_true_returns):.3f}")
    print(f"Initial Q-values: max={max_init_q:.3f}, mean={mean_init_q:.3f}")
    print(f"Valid predictions: {len(all_q_preds)}/{sum(len(q) for q in traj_q_vals)}")

    return traj_q_vals, max_init_q


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
    Fixed Correlated Q-Learning implementation matching Implicit Q patterns.
    
    Returns:
        relabeled_trajs: Trajectories with Q-value based returns-to-go
        prompt_value: Maximum initial Q-value across trajectories
    """

    print("=== Fixed Correlated Q-Learning (Matching Implicit Q) ===")

    # Compute dataset statistics like Implicit Q
    all_rewards = []
    for traj in trajs:
        episode_reward = np.sum(traj.rewards)
        all_rewards.append(episode_reward)

    print(f"Dataset size: {len(trajs)} episodes")
    print(f"Reward stats: mean={np.mean(all_rewards):.3f}, std={np.std(all_rewards):.3f}")

    # Setup action spaces like Implicit Q
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

    print(f"Action type: {action_type}")
    print(f"Obs size: {obs_size}, Action size: {action_size}, Adv action size: {adv_action_size}")

    # STEP 1: Train Correlated Q-Learning Model
    print("\n=== Step 1: Correlated Q-Learning Training ===")
    q_model, scale_factor = correlated_q_learning(
        trajs, obs_size, action_size, adv_action_size,
        train_args, device, is_simple_model, action_type
    )

    # STEP 2: Evaluate Q-Model Performance
    print("\n=== Step 2: Q-Model Evaluation ===")
    learned_q_values, prompt_value = evaluate_q_model(
        q_model, trajs, obs_size, action_size, adv_action_size,
        action_type, device, scale_factor, is_discretize
    )

    # STEP 3: Trajectory Relabeling with Q-values like Implicit Q
    print("\n=== Step 3: Trajectory Relabeling with Q-values ===")
    relabeled_trajs = []

    for i, (traj, q_vals) in enumerate(zip(trajs, learned_q_values)):
        # Create relabeled trajectory
        relabeled_traj = deepcopy(traj)

        # Replace returns-to-go with Q-value estimates
        if len(q_vals) > 0:
            new_returns_to_go = q_vals.tolist()
        else:
            # Fallback: use original returns-to-go calculation
            original_returns = [np.sum(traj.rewards[t:]) for t in range(len(traj.rewards))]
            new_returns_to_go = original_returns

        relabeled_traj.minimax_returns_to_go = new_returns_to_go
        relabeled_trajs.append(relabeled_traj)

    print(f"Relabeled {len(relabeled_trajs)} trajectories with minimax returns-to-go")
    
    # Get final prompt value like Implicit Q
    initial_minimax_returns_all = [t.minimax_returns_to_go[0] for t in relabeled_trajs
                                   if hasattr(t, 'minimax_returns_to_go') and t.minimax_returns_to_go
                                   and len(t.minimax_returns_to_go) > 0]

    if initial_minimax_returns_all:
        prompt_value = np.max(initial_minimax_returns_all)
    else:
        prompt_value = 0.0

    print(f"Relabeling complete. Prompt value: {prompt_value:.3f}")

    return relabeled_trajs, np.round(prompt_value, decimals=3)