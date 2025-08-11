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


def update_q_with_ce(q_model, obs, acts, adv_acts, obs_next, rewards, dones, gamma, tau):
    """Update Q-function using correlated equilibrium for next state values."""
    B, T, obs_dim = obs.shape
    A1, A2 = acts.shape[-1], adv_acts.shape[-1]

    # Generate all joint action combinations
    all_a = torch.eye(A1, device=obs.device)
    all_adv = torch.eye(A2, device=obs.device)
    joint = torch.cartesian_prod(torch.arange(A1), torch.arange(A2))
    a_joint = all_a[joint[:, 0]]
    adv_joint = all_adv[joint[:, 1]]
    N = A1 * A2

    # Expand next states and actions for all joint actions
    obs_next_exp = obs_next.unsqueeze(2).repeat(1, 1, N, 1)
    a_exp = a_joint.view(1, 1, N, -1).expand(B, T, -1, -1)
    adv_exp = adv_joint.view(1, 1, N, -1).expand(B, T, -1, -1)

    # Compute Q-values for all joint actions in next state
    with torch.no_grad():  # Don't need gradients for target computation
        q_next = q_model(obs_next_exp.reshape(-1, 1, obs_dim),
                         a_exp.reshape(-1, 1, A1),
                         adv_exp.reshape(-1, 1, A2)).view(B, T, A1, A2)

    # Compute correlated equilibrium policy
    ce_policy = approx_ce_batch(q_next, tau)

    # Bellman backup with CE policy and proper terminal state handling
    next_value = (ce_policy * q_next).sum(dim=[2, 3])
    q_target = rewards + gamma * (1 - dones.float()) * next_value

    return q_target.detach(), ce_policy.detach(), q_next.detach()


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

    print("=== Correlated Q-Learning Training (No Scaling) ===")

    # Initialize Q-model
    if is_simple_model:
        q_model = RtgFFN(obs_size, action_size, adv_action_size, include_adv=True).to(device)
    else:
        q_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)

    optimizer = torch.optim.AdamW(
        q_model.parameters(),
        lr=train_args['model_lr'],
        weight_decay=train_args.get('model_wd', 1e-4)
    )

    # Dataset preparation
    max_len = max([len(traj.obs) for traj in trajs]) + 1
    dataset = ARDTDataset(trajs, max_len, gamma=train_args['gamma'], act_type=action_type)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_args['batch_size']
    )

    # Training hyperparameters
    mse_epochs = train_args.get('mse_epochs', 5)
    ce_epochs = train_args.get('maxmin_epochs', 15)  # Using maxmin_epochs for consistency
    total_epochs = mse_epochs + ce_epochs
    tau = train_args.get('ce_temperature', 1.0)
    lambda_ce = train_args.get('lambda_ce', 1.0)

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

        for obs, acts, adv_acts, ret, seq_len in tqdm(dataloader, desc=f"Epoch {epoch} ({phase})"):
            batch_size, T = obs.shape[:2]
            obs = obs.view(batch_size, T, -1).to(device)

            # Prepare next state observations
            obs_next = obs.clone()
            obs_next[:, :-1] = obs[:, 1:]
            obs_next[:, -1] = obs[:, -1]  # Terminal state handling

            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            ret = ret.to(device)

            # Compute rewards and done flags from returns-to-go
            rewards = ret[:, :-1] - ret[:, 1:]
            rewards = F.pad(rewards, (0, 1), value=0.0)

            # Create proper done flags (terminal states)
            dones = torch.zeros_like(rewards, dtype=torch.bool)
            for i, length in enumerate(seq_len):
                if length > 0 and length <= rewards.shape[1]:
                    dones[i, length-1] = True  # Mark terminal state

            if epoch < mse_epochs:
                # MSE warmup phase: predict return-to-go directly
                q_pred = q_model(obs, acts, adv_acts).view(batch_size, T)
                mse_loss = F.mse_loss(q_pred, ret)
                ce_loss = torch.tensor(0.0, device=device)
                total_loss = mse_loss
            else:
                # CE-Q learning phase
                q_target, ce_policy, q_next = update_q_with_ce(
                    q_model, obs, acts, adv_acts, obs_next,
                    rewards, dones, train_args['gamma'], tau
                )

                q_pred = q_model(obs, acts, adv_acts).view(batch_size, T)
                mse_loss = F.mse_loss(q_pred, q_target)
                ce_loss = incentive_violation_loss_zero_sum(ce_policy, q_next)
                total_loss = mse_loss + lambda_ce * ce_loss

            optimizer.zero_grad()
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(q_model.parameters(), max_norm=1.0)
            optimizer.step()

            total_q_loss += mse_loss.item()
            total_ce_loss += ce_loss.item()
            num_batches += 1

        avg_q_loss = total_q_loss / num_batches
        avg_ce_loss = total_ce_loss / num_batches
        print(f"Epoch {epoch} ({phase}): Q Loss = {avg_q_loss:.6f}, CE Loss = {avg_ce_loss:.6f}")

    return q_model


def evaluate_q_model(q_model, trajs, obs_size, action_size, adv_action_size,
                     action_type, device, is_discretize=False):
    """Evaluate Q-model on trajectories and compute statistics with simple debug logs."""
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
                    # print(f"[{i}] One-hot encoding discrete actions")
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
                    # print(f"[{i}] Warning: Unable to reshape continuous actions: {e}")
                    continue

            try:
                q_preds = q_model(obs, acts, adv_acts).cpu().flatten().numpy()
                true_returns = np.array([np.sum(traj.rewards[t:]) for t in range(len(traj.rewards))])

                length = min(len(q_preds), len(true_returns))
                q_preds = q_preds[:length]
                true_returns = true_returns[:length]

                all_q_preds.extend(q_preds)
                all_true_returns.extend(true_returns)
                traj_q_vals.append(q_preds)
            except Exception as e:
                # print(f"[{i}] Error evaluating trajectory: {e}")
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
    Correlated Q-Learning implementation matching ARDT format, without scaling.
    
    Returns:
        relabeled_trajs: Trajectories with Q-value based returns-to-go
        prompt_value: Maximum initial Q-value across trajectories
    """

    print("=== Correlated Q-Learning ===")

    # Compute dataset statistics
    all_rewards = []
    for traj in trajs:
        episode_reward = np.sum(traj.rewards)
        all_rewards.append(episode_reward)

    print(f"Dataset size: {len(trajs)} episodes")
    print(f"Reward stats: mean={np.mean(all_rewards):.3f}, std={np.std(all_rewards):.3f}")

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

    print(f"Action type: {action_type}")
    print(f"Obs size: {obs_size}, Action size: {action_size}, Adv action size: {adv_action_size}")

    # STEP 1: Train Correlated Q-Learning Model
    print("\n=== Step 1: Correlated Q-Learning Training ===")
    q_model = correlated_q_learning(
        trajs, obs_size, action_size, adv_action_size,
        train_args, device, is_simple_model, action_type
    )

    # STEP 2: Evaluate Q-Model Performance
    print("\n=== Step 2: Q-Model Evaluation ===")
    learned_q_values, prompt_value = evaluate_q_model(
        q_model, trajs, obs_size, action_size, adv_action_size,
        action_type, device, is_discretize
    )

    # STEP 3: Trajectory Relabeling with Q-values
    print("\n=== Step 3: Trajectory Relabeling with Q-values ===")
    relabeled_trajs = []

    for i, (traj, q_vals) in enumerate(zip(trajs, learned_q_values)):
        # Create relabeled trajectory
        relabeled_traj = deepcopy(traj)

        # Replace returns-to-go with Q-value estimates
        # Round to 3 decimal places like in the original code
        new_returns_to_go = np.round(q_vals, decimals=3).tolist()

        relabeled_traj.minimax_returns_to_go = new_returns_to_go
        relabeled_trajs.append(relabeled_traj)

    print(f"Relabeled {len(relabeled_trajs)} trajectories with minimax returns-to-go")
    initial_minimax_returns_all = [t.minimax_returns_to_go[0] for t in relabeled_trajs
                                   if hasattr(t, 'minimax_returns_to_go') and t.minimax_returns_to_go
                                   and len(t.minimax_returns_to_go) > 0]

    if initial_minimax_returns_all:
        prompt_value = np.max(initial_minimax_returns_all)
    else:
        prompt_value = 0.0

    return relabeled_trajs, np.round(prompt_value, decimals=3)