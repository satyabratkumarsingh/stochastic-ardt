import gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm
from copy import deepcopy
from data_class.trajectory import Trajectory
from core_models.base_models.base_model import RtgFFN, RtgLSTM
from core_models.dataset.ardt_dataset import ARDTDataset

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


def evaluate_models(R_max_model, R_min_model, dataloader, device):
    """Evaluate R_max and R_min on one batch and print mean predictions + gap."""
    with torch.no_grad():
        obs, acts, adv_acts, ret, seq_len = next(iter(dataloader))
        obs = obs.to(device)
        acts = acts.to(device)
        adv_acts = adv_acts.to(device)
        ret = ret.to(device)  # No scaling

        pred_max = R_max_model(obs, acts).mean().item()
        pred_min = R_min_model(obs, acts, adv_acts).mean().item()
        true_mean = ret.mean().item()

    gap = pred_max - pred_min
    symbol = "✅" if gap >= 0 else "❌"
    print(f"   Eval -> True mean: {true_mean:.4f}, "
      f"R_max mean: {pred_max:.4f}, R_min mean: {pred_min:.4f}, "
      f"Gap: {gap:.4f} {symbol}")
    return pred_max, pred_min, gap


def ardt_minimax_expectile_regression(
    trajs: list[Trajectory],
    obs_size: int,
    action_size: int,
    adv_action_size: int,
    train_args: dict,
    device: str
) -> tuple[torch.nn.Module, torch.nn.Module]:

    # Models
    R_max_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=False).to(device)
    R_min_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)

    max_optimizer = torch.optim.AdamW(R_max_model.parameters(), lr=train_args['model_lr'])
    min_optimizer = torch.optim.AdamW(R_min_model.parameters(), lr=train_args['model_lr'])

    # Dataset
    max_len = max([len(traj.obs) for traj in trajs]) + 1
    dataset = ARDTDataset(trajs, max_len, gamma=train_args['gamma'], act_type='continuous')
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=train_args['batch_size'])

    alpha_max = 0.01
    alpha_min = 0.99
    
    warmup_epochs = train_args.get('warmup_epochs', 10)
    minimax_epochs = train_args.get('minimax_epochs', 30)
    gamma = train_args.get('gamma', 0.99)
    leaf_weight = train_args.get('leaf_weight', 0.1)

    print("=== ARDT Minimax Expectile Regression ===")
    print(f"Warmup epochs: {warmup_epochs}")
    print(f"Minimax epochs: {minimax_epochs}")
    print(f"Alpha_max (optimistic): {alpha_max}")
    print(f"Alpha_min (pessimistic): {alpha_min}")
    print("Using ALTERNATING updates during minimax phase")
    print("NO SCALING - using raw return values")

    # ---- Warmup ----
    for epoch in range(warmup_epochs):
        total_max_loss, total_min_loss = 0, 0
        num_batches = 0
        for obs, acts, adv_acts, ret, seq_len in tqdm(dataloader, desc=f"Warmup Epoch {epoch}"):
            batch_size, obs_len = obs.shape[0], obs.shape[1]
            obs = obs.view(batch_size, obs_len, -1).to(device)
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            ret = ret.to(device)  # No scaling
            timestep_mask = torch.arange(obs_len, device=device)[None, :] < seq_len[:, None]
            
            # R_max
            max_optimizer.zero_grad()
            max_pred = R_max_model(obs, acts).view(batch_size, obs_len)
            max_loss = (((max_pred - ret) ** 2) * timestep_mask).sum() / timestep_mask.sum()
            max_loss.backward()
            max_optimizer.step()

            # R_min
            min_optimizer.zero_grad()
            min_pred = R_min_model(obs, acts, adv_acts).view(batch_size, obs_len)
            min_loss = (((min_pred - ret) ** 2) * timestep_mask).sum() / timestep_mask.sum()
            min_loss.backward()
            min_optimizer.step()
            
            total_max_loss += max_loss.item()
            total_min_loss += min_loss.item()
            num_batches += 1
        
        print(f"Warmup Epoch {epoch}: Max Loss = {total_max_loss/num_batches:.6f}, Min Loss = {total_min_loss/num_batches:.6f}")
        evaluate_models(R_max_model, R_min_model, dataloader, device)

    # ---- Minimax with Alternating Updates ----
    for epoch in range(minimax_epochs):
        total_max_loss, total_min_loss = 0, 0
        num_batches = 0
        
        # Determine which model to update this epoch
        update_max = (epoch % 2 == 0)
        update_desc = "MAX" if update_max else "MIN"

        for obs, acts, adv_acts, ret, seq_len in tqdm(dataloader, desc=f"Minimax Epoch {epoch} ({update_desc})"):
            batch_size, obs_len = obs.shape[0], obs.shape[1]
            obs = obs.view(batch_size, obs_len, -1).to(device)
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            ret = ret.to(device)  # No scaling
            
            # Mask for valid timesteps
            timestep_mask = torch.arange(obs_len, device=device)[None, :] < seq_len[:, None]
            
            # Mask for valid transitions (all but the last timestep)
            transition_mask = timestep_mask[:, :-1] & timestep_mask[:, 1:]
            
            # Immediate rewards for Bellman backup
            if obs_len > 1:
                immediate_rewards = ret[:, :-1] - gamma * ret[:, 1:]
            
            if update_max:
                # --- R_max update (MAXIMIZE step) ---
                max_optimizer.zero_grad()
                
                if obs_len > 1 and transition_mask.sum() > 0:
                    max_pred_all = R_max_model(obs, acts).view(batch_size, obs_len)
                    max_pred_transitions = max_pred_all[:, :-1]
                    
                    # R_max maximizes against R_min's future predictions
                    with torch.no_grad():
                        min_next = R_min_model(obs[:, 1:], acts[:, 1:], adv_acts[:, 1:]).view(batch_size, obs_len - 1)
                        targets = immediate_rewards + gamma * min_next
                    
                    max_loss = (expectile_loss(max_pred_transitions - targets, alpha_max) * transition_mask).sum() / transition_mask.sum()
                else:
                    max_loss = torch.tensor(0.0, device=device)

                max_loss.backward()
                torch.nn.utils.clip_grad_norm_(R_max_model.parameters(), max_norm=1.0)
                max_optimizer.step()
                
                total_max_loss += max_loss.item()
                total_min_loss += 0.0
                
            else:
                # --- R_min update (MINIMIZE step) ---
                min_optimizer.zero_grad()
                
                if obs_len > 1 and transition_mask.sum() > 0:
                    min_pred_all = R_min_model(obs, acts, adv_acts).view(batch_size, obs_len)
                    min_pred_transitions = min_pred_all[:, :-1]
                    
                    # R_min minimizes against R_max's future predictions
                    with torch.no_grad():
                        max_next = R_max_model(obs[:, 1:], acts[:, 1:]).view(batch_size, obs_len - 1)
                        targets = immediate_rewards + gamma * max_next
                    
                    min_expectile_loss = (expectile_loss(min_pred_transitions - targets, alpha_min) * transition_mask).sum() / transition_mask.sum()
                    
                    # Terminal state regularization (leaf loss)
                    terminal_indices = torch.arange(batch_size, device=device)
                    valid_terminals = (seq_len > 0) & (seq_len <= obs_len)
                    if valid_terminals.sum() > 0:
                        terminal_seq_len = (seq_len[valid_terminals] - 1).clamp(0, obs_len - 1)
                        leaf_loss = ((min_pred_all[valid_terminals, terminal_seq_len] - 
                                    ret[valid_terminals, terminal_seq_len]) ** 2).mean()
                    else:
                        leaf_loss = torch.tensor(0.0, device=device)
                    
                    min_loss = (1 - leaf_weight) * min_expectile_loss + leaf_weight * leaf_loss
                else:
                    min_loss = torch.tensor(0.0, device=device)
                
                min_loss.backward()
                torch.nn.utils.clip_grad_norm_(R_min_model.parameters(), max_norm=1.0)
                min_optimizer.step()
                
                total_min_loss += min_loss.item()
                total_max_loss += 0.0

            num_batches += 1

        print(f"Minimax Epoch {epoch} ({update_desc}): Max Loss = {total_max_loss/num_batches:.6f}, Min Loss = {total_min_loss/num_batches:.6f}")
        evaluate_models(R_max_model, R_min_model, dataloader, device)

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
    
    # No scaling - work with raw values
    print("========= Using NO SCALING - raw return values")
    
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
        trajs, obs_size, action_size, adv_action_size, train_args, device
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
            
            # Get minimax return predictions (no scaling)
            minimax_returns = R_max_model(obs, acts.float()).squeeze().cpu().numpy()
            
            # Create new trajectory with relabeled returns-to-go
            relabeled_traj = deepcopy(traj)
            
            # Replace returns-to-go with minimax estimates
            new_returns_to_go = []
            for t in range(len(traj.obs)):
                new_returns_to_go.append(minimax_returns[t])
            
            # Store relabeled trajectory
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
    
    # STEP 3: Decision Transformer Training
    print("\n=== Step 3: Decision Transformer Training ===")
    
    return relabeled_trajs, prompt_value