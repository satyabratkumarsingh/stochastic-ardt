import gym
import numpy as np
import torch
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


def incentive_violation_loss_zero_sum(policy: torch.Tensor, q_values: torch.Tensor) -> torch.Tensor:
    """Incentive compatibility loss for zero-sum games."""
    B, T, A1, A2 = q_values.shape
    pi = policy.view(B, T, A1, A2)
    expected_value = (pi * q_values).sum(dim=[2, 3], keepdim=True)

    a_marg = pi.sum(dim=3, keepdim=True)  # marginal over a2
    adv_marg = pi.sum(dim=2, keepdim=True)  # marginal over a1

    q_diff_1 = expected_value - q_values  # protagonist deviation
    q_diff_2 = q_values - expected_value  # adversary deviation (since -Q)

    viol_1 = F.relu(q_diff_1 * a_marg)
    viol_2 = F.relu(q_diff_2 * adv_marg)

    return viol_1.mean() + viol_2.mean()


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

    print("=== Correlated Q-Learning Training (Fixed to match Implicit Q) ===")

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

            if epoch < mse_epochs:
                # MSE warmup phase: predict return-to-go directly like Implicit Q
                q_pred = q_model(obs, acts, adv_acts)
                
                # Handle action selection like Implicit Q
                if len(q_pred.shape) == 3:
                    q_pred_flat = (q_pred * adv_acts).sum(dim=-1)
                else:
                    q_pred_flat = q_pred.view(batch_size, obs_len)
                
                # Clamp predictions like Implicit Q
                q_pred_flat = torch.clamp(q_pred_flat, -1e6, 1e6)
                
                if timestep_mask.sum() > 0:
                    mse_loss = (((q_pred_flat - ret) ** 2) * timestep_mask.float()).sum() / timestep_mask.sum()
                else:
                    mse_loss = torch.tensor(0.0, device=device, requires_grad=True)
                
                ce_loss = torch.tensor(0.0, device=device)
                total_loss = mse_loss
            else:
                # CE-Q learning phase with proper next state handling
                if transition_mask is None or transition_mask.sum() == 0:
                    continue
                    
                # Get current Q values
                q_current = q_model(obs, acts, adv_acts)
                
                # Handle action selection for current Q
                if len(q_current.shape) == 3:
                    q_current_flat = (q_current * adv_acts).sum(dim=-1)
                else:
                    q_current_flat = q_current.view(batch_size, obs_len)
                
                # Prepare next state data
                obs_next = obs.clone()
                obs_next[:, :-1] = obs[:, 1:]
                obs_next[:, -1] = obs[:, -1]  # Terminal state handling
                
                acts_next = acts.clone()
                acts_next[:, :-1] = acts[:, 1:]
                acts_next[:, -1] = acts[:, -1]
                
                adv_acts_next = adv_acts.clone()
                adv_acts_next[:, :-1] = adv_acts[:, 1:]
                adv_acts_next[:, -1] = adv_acts[:, -1]

                # Compute next Q values
                with torch.no_grad():
                    q_next = q_model(obs_next, acts_next, adv_acts_next)
                    
                    if len(q_next.shape) == 3:
                        # For CE, we need to consider all action combinations
                        A1, A2 = acts.shape[-1], adv_acts.shape[-1]
                        q_next_reshaped = q_next.view(batch_size, -1, A1, A2)
                        ce_policy = approx_ce_batch(q_next_reshaped, tau)
                        next_value = (ce_policy * q_next_reshaped).sum(dim=[2, 3])
                    else:
                        next_value = q_next.view(batch_size, obs_len)

                # Bellman backup like Implicit Q
                q_pred = q_current_flat[:, :-1]
                v_next = next_value[:, 1:].detach()
                q_target = immediate_rewards + train_args['gamma'] * v_next
                q_target = torch.clamp(q_target, -1e6, 1e6)
                
                # Q loss
                mse_loss = (((q_pred - q_target) ** 2) * transition_mask).sum() / transition_mask.sum()
                
                # CE loss if we have the policy
                if len(q_next.shape) == 3:
                    ce_loss = incentive_violation_loss_zero_sum(ce_policy, q_next_reshaped)
                else:
                    ce_loss = torch.tensor(0.0, device=device)
                
                total_loss = mse_loss + lambda_ce * ce_loss

            # Check for NaN/Inf like Implicit Q
            if torch.isnan(total_loss) or torch.isinf(total_loss):
                print(f"❌ FATAL: NaN/Inf detected in total_loss at epoch {epoch}")
                raise RuntimeError("Training stopped due to NaN/Inf in loss")

            optimizer.zero_grad()
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
                
                # Handle action selection like in training
                if len(q_preds_raw.shape) == 3:
                    q_preds = (q_preds_raw * adv_acts).sum(dim=-1).cpu().flatten().numpy()
                else:
                    q_preds = q_preds_raw.cpu().flatten().numpy()
                
                # De-scale like Implicit Q to match original reward scale
                q_preds = q_preds / scale_factor
                
                true_returns = np.array([np.sum(traj.rewards[t:]) for t in range(len(traj.rewards))])

                length = min(len(q_preds), len(true_returns))
                q_preds = q_preds[:length]
                true_returns = true_returns[:length]

                all_q_preds.extend(q_preds)
                all_true_returns.extend(true_returns)
                traj_q_vals.append(q_preds)
            except Exception as e:
                continue

    all_q_preds = np.array(all_q_preds)
    all_true_returns = np.array(all_true_returns)

    mse = np.mean((all_q_preds - all_true_returns) ** 2)
    mae = np.mean(np.abs(all_q_preds - all_true_returns))
    correlation = np.corrcoef(all_q_preds, all_true_returns)[0, 1] if len(all_true_returns) > 0 else 0.0

    initial_qs = [q[0] if len(q) > 0 else 0 for q in traj_q_vals]
    max_init_q = np.max(initial_qs) if initial_qs else 0.0
    mean_init_q = np.mean(initial_qs) if initial_qs else 0.0

    print(f"\n=== Q-Model Evaluation Results ===")
    print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, Corr: {correlation:.4f}")
    print(f"Q-value mean: {np.mean(all_q_preds):.3f}, std: {np.std(all_q_preds):.3f}")
    print(f"Return mean: {np.mean(all_true_returns):.3f}, std: {np.std(all_true_returns):.3f}")
    print(f"Initial Q-values: max={max_init_q:.3f}, mean={mean_init_q:.3f}")

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
        new_returns_to_go = q_vals.tolist()

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