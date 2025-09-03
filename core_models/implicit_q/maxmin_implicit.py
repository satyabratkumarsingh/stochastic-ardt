
import gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from data_class.trajectory import Trajectory
from core_models.base_models.base_model import RtgFFN, RtgLSTM
from core_models.dataset.ardt_dataset import ARDTDataset
from core_models.implicit_q.value_net import ValueNet
from copy import deepcopy
import numpy as np
import torch
from scipy.stats import entropy
import pickle
import pyspiel
import gc
import sys
import atexit
import json
import os

# Add proper cleanup handling
def cleanup_torch():
    """Ensure proper cleanup of PyTorch resources"""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
    except:
        pass

# Register cleanup function
atexit.register(cleanup_torch)

def load_solver(file_path):
    """Load a saved CFR solver from pickle file"""
    with open(file_path, "rb") as f:
        print(f"Loading expert solver from {file_path}")
        return pickle.load(f)


# Correct theoretical Nash equilibrium for Kuhn Poker
# Format: [P(Pass), P(Bet), P(NoAction)] where NoAction is always 0
KUHN_NASH_EQUILIBRIUM = {
    # Player 1's first action (no history)
    "0": [1.0, 0.0, 0.0],      # Jack: always pass
    "1": [2/3, 1/3, 0.0],      # Queen: pass 2/3, bet 1/3  
    "2": [0.0, 1.0, 0.0],      # King: always bet
    
    # Player 2's response after Player 1 passes
    "0p": [1.0, 0.0, 0.0],     # Jack: always pass (fold)
    "1p": [1.0, 0.0, 0.0],     # Queen: always pass (check)
    "2p": [0.0, 1.0, 0.0],     # King: always bet
    
    # Player 2's response after Player 1 bets  
    "0b": [1.0, 0.0, 0.0],     # Jack: always pass (fold)
    "1b": [2/3, 1/3, 0.0],     # Queen: call 1/3 of the time
    "2b": [0.0, 1.0, 0.0],     # King: always call
    
    # Player 1's response after pass-bet sequence
    "0pb": [1.0, 0.0, 0.0],    # Jack: always fold
    "1pb": [2/3, 1/3, 0.0],    # Queen: fold 2/3, call 1/3  
    "2pb": [0.0, 1.0, 0.0]     # King: always call
}

def print_model_q_values(
    epoch_id, qsa_pr_model, qsa_adv_model, value_model, device, stage_name, state_mapping,
    cfr_solver=None, game=None, log_adv_matrix=True, save_path="epoch_stats.json"
):
    print(f"\n=== {stage_name} ===")
    
    # Load existing data if file exists
    if os.path.exists(save_path):
        with open(save_path, "r") as f:
            try:
                all_stats = json.load(f)
            except json.JSONDecodeError:
                all_stats = {}
    else:
        all_stats = {}

    current_epoch_stats = {}
    l1_devs = []
    kl_devs = []
    exploitability_scores = []

    canonical_states = ["0", "0p", "0b", "0pb",
                        "1", "1p", "1b", "1pb", 
                        "2", "2p", "2b", "2pb"]
    
    actions = {"Pass": [1.0, 0.0, 0.0], "Bet": [0.0, 1.0, 0.0]}
    card_names = {"0": "Jack", "1": "Queen", "2": "King"}

    qsa_pr_model.eval()
    qsa_adv_model.eval() 
    value_model.eval()

    with torch.no_grad():
        for state_str in canonical_states:
            if state_str not in state_mapping:
                continue

            obs_tensor = torch.tensor([state_mapping[state_str]], dtype=torch.float32, device=device).unsqueeze(1)
            card_name = card_names[state_str[0]]
            history = state_str[1:] if len(state_str) > 1 else "initial"

            # Get Q-values for both actions
            pr_qs = {}
            for action_name, act_vec in actions.items():
                act_tensor = torch.tensor([[[act_vec[0], act_vec[1], act_vec[2]]]], dtype=torch.float32, device=device)
                q_val = qsa_pr_model(obs_tensor, act_tensor)
                pr_qs[action_name] = q_val.item()

            # Nash analysis
            nash_str = ""
            nash_probs = None
            l1_deviation = None
            kl_divergence = None
            exploitability = None
            
            if state_str in KUHN_NASH_EQUILIBRIUM:
                nash_probs = KUHN_NASH_EQUILIBRIUM[state_str]
                nash_str = f" | Nash: Pass={nash_probs[0]:.3f}, Bet={nash_probs[1]:.3f}"
                
                # Compute learned policy using softmax with temperature
                temperature = 1.0
                q_values = np.array([pr_qs["Pass"], pr_qs["Bet"]])
                
                # Softmax policy (what the model would actually play)
                exp_q = np.exp(q_values / temperature)
                learned_policy_2d = exp_q / np.sum(exp_q)
                learned_policy_3d = np.array([learned_policy_2d[0], learned_policy_2d[1], 0.0])
                
                # Also compute greedy policy for comparison
                greedy_policy_3d = np.zeros(3)
                greedy_policy_3d[np.argmax(q_values)] = 1.0
                
                # L1 deviation between learned softmax policy and Nash
                l1_deviation = float(np.sum(np.abs(learned_policy_3d - nash_probs)))
                l1_devs.append(l1_deviation)
                
                # KL divergence from Nash to learned policy (use 2D versions)
                eps = 1e-8
                nash_2d = np.array(nash_probs[:2]) + eps
                learned_2d = learned_policy_2d + eps
                
                # Normalize to ensure they sum to 1 (important for KL divergence)
                nash_2d = nash_2d / np.sum(nash_2d)
                learned_2d = learned_2d / np.sum(learned_2d)
                
                kl_divergence = float(np.sum(nash_2d * np.log(nash_2d / learned_2d)))
                kl_devs.append(kl_divergence)
                
                # Exploitability: gain from best response vs Nash
                nash_value = np.sum(nash_probs[:2] * q_values)
                best_response_value = np.max(q_values)
                exploitability = float(best_response_value - nash_value)
                exploitability_scores.append(exploitability)
                
                # Concise output
                print(f"{state_str:4s} ({card_name:5s}) | Q(Pass/Bet)={pr_qs['Pass']:.3f}/{pr_qs['Bet']:.3f} | "
                      f"Nash={nash_probs[0]:.2f}/{nash_probs[1]:.2f} | "
                      f"L1={l1_deviation:.3f} | Exploit={exploitability:.4f}")

            else:
                print(f"{state_str:4s} ({card_name:5s}) | Q(Pass/Bet)={pr_qs['Pass']:.3f}/{pr_qs['Bet']:.3f}")

            # Compute adversary matrix but don't print unless specifically requested
            adv_matrix = None
            if log_adv_matrix:
                adv_matrix = np.zeros((2, 2))
                for i, pr_name in enumerate(["Pass", "Bet"]):
                    for j, adv_name in enumerate(["Pass", "Bet"]):
                        pr_action = actions[pr_name]
                        adv_action = actions[adv_name]
                        pr_tensor = torch.tensor([[[pr_action[0], pr_action[1], pr_action[2]]]], dtype=torch.float32, device=device)
                        adv_tensor = torch.tensor([[[adv_action[0], adv_action[1], adv_action[2]]]], dtype=torch.float32, device=device)
                        adv_matrix[i, j] = qsa_adv_model(obs_tensor, pr_tensor, adv_tensor).item()

            # Get state value but don't print
            v = value_model(obs_tensor)
            value = v.item()

            # Save stats for the current epoch
            current_epoch_stats[state_str] = {
                "card": card_name,
                "history": history,
                "protagonist_q": pr_qs,
                "nash_probs": nash_probs,
                "l1_deviation": l1_deviation,
                "kl_divergence": kl_divergence,
                "exploitability": exploitability,
                "state_value": value,
                "adv_matrix": adv_matrix.tolist() if adv_matrix is not None else None
            }

    # Summary statistics for the current epoch
    if l1_devs or kl_devs or exploitability_scores:
        print(f"\n{'='*60}")
        print(f"Nash Analysis: {stage_name}")
        if l1_devs:
            print(f"L1 Deviation:  Mean={np.mean(l1_devs):.3f}  Max={np.max(l1_devs):.3f}")
        if kl_devs:
            print(f"KL Divergence: Mean={np.mean(kl_devs):.3f}  Max={np.max(kl_devs):.3f}")
        if exploitability_scores:
            print(f"Exploitability:Mean={np.mean(exploitability_scores):.4f} Max={np.max(exploitability_scores):.4f}")
        print(f"{'='*60}")
    else:
        print(f"\n=== {stage_name} Complete ===\n")

    current_epoch_stats["summary"] = {
        "mean_l1_deviation": float(np.mean(l1_devs)) if l1_devs else None,
        "max_l1_deviation": float(np.max(l1_devs)) if l1_devs else None,
        "mean_kl_divergence": float(np.mean(kl_devs)) if kl_devs else None,
        "mean_exploitability": float(np.mean(exploitability_scores)) if exploitability_scores else None
    }
    
    # Add the current epoch's data to the master dictionary
    all_stats[f"epoch_{epoch_id}"] = current_epoch_stats

    # Write the entire dictionary back to the same file
    with open(save_path, "w") as f:
        json.dump(all_stats, f, indent=2)
    print(f"Saved epoch stats to {save_path}")


def _expectile_fn(
        td_error: torch.Tensor, 
        acts_mask: torch.Tensor, 
        alpha: float = 0.01, 
        discount_weighted: bool = False
    ) -> torch.Tensor:
    """
    Expectile loss function to focus on different quantiles of the TD-error distribution.

    Args:
        td_error (torch.Tensor): Temporal difference error.
        acts_mask (torch.Tensor): Mask for invalid actions.
        alpha (float, optional): Expectile quantile parameter (default is 0.01).
        discount_weighted (bool, optional): If True, apply discount weighting.

    Returns:
        torch.Tensor: Computed expectile loss.
    """
    # Normalize and apply ReLU to the TD-error
    batch_loss = torch.abs(alpha - F.normalize(F.relu(td_error), dim=-1))
    
    # Square the TD-error
    batch_loss *= (td_error ** 2)

    # Apply discount weighting if needed
    if discount_weighted:
        weights = 0.5 ** np.array(range(len(batch_loss)))[::-1]
        return (
            batch_loss[~acts_mask] * torch.from_numpy(weights).to(td_error.device)
        ).mean()
    else:
        # Calculate expectile loss for valid actions
        return (batch_loss.squeeze(-1) * ~acts_mask).mean()
    

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
        state_mapping  = None
    ) -> tuple[np.ndarray, float]:
    """
    Train a correlated equilibrium Q-learning model to handle correlated equilibrium returns.

    Args:
        trajs (list[Trajectory]): List of trajectories.
        action_space (gym.spaces.Space): The action space of the environment.
        adv_action_space (gym.spaces.Space): Adversarial action space.
        train_args (dict): Training arguments including epochs, learning rates, and batch size.
        device (str): Device to run computations on ('cpu' or 'cuda').
        n_cpu (int): Number of CPUs to use for data loading.
        is_simple_model (bool, optional): Use a simpler model for testing (default is False).
        is_toy (bool, optional): Whether the environment is a toy model (default is False).
        is_discretize (bool, optional): Whether to discretize actions for certain environments (default is False).

    Returns:
        tuple: Learned return labels and highest returns-to-go (prompt value).
    """
    # Initialize state and action spaces
    solver = load_solver("kuhn_poker_solver_cfr_plus.pkl")
    game = pyspiel.load_game("kuhn_poker") if solver is not None else None
    if isinstance(action_space, gym.spaces.Discrete):
        obs_size = np.prod(trajs[0].obs[0].shape)
        action_size = 3
        adv_action_size = 3
        action_type = 'discrete'
    else:
        obs_size = np.prod(np.array(trajs[0].obs[0]).shape)
        action_size = 3
        adv_action_size = 3
        action_type = 'continuous'

    # Build dataset and dataloader for training
    max_len = max([len(traj.obs) for traj in trajs]) + 1
    gamma=train_args['gamma']
    dataset = ARDTDataset(trajs, max_len, gamma=gamma, act_type=action_type, include_player_ids= True)

    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_args['batch_size'], num_workers=n_cpu
    )

    # Set up the models (MLP or LSTM-based) involved in the ARDT algorithm
    print(f'Creating models... (simple={is_simple_model})')
    if is_simple_model:
        qsa_pr_model = RtgFFN(obs_size, action_size, adv_action_size, include_adv=False).to(device)
        qsa_adv_model = RtgFFN(obs_size, action_size, adv_action_size, include_adv=True).to(device)
        value_model = ValueNet(obs_size, is_lstm=False, train_args=train_args).to(device)

    else:
        qsa_pr_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=False).to(device)
        qsa_adv_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)
        value_model = ValueNet(obs_size, is_lstm=True, train_args=train_args).to(device)


    qsa_pr_optimizer = torch.optim.AdamW(
        qsa_pr_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )
    qsa_adv_optimizer = torch.optim.AdamW(
        qsa_adv_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )
    value_model_optimizer = torch.optim.AdamW(
        value_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )



    # Start training and running the ARDT algorithm
    mse_epochs = train_args.get('mse_epochs', 5)
    maxmin_epochs = train_args.get('minimax_epochs', 15)
    total_epochs = mse_epochs + maxmin_epochs
    assert maxmin_epochs % 3 == 0

    print('Training...')
    

    print_model_q_values(-1, qsa_pr_model, qsa_adv_model, value_model, device, "Before training starts", state_mapping, solver, game)
    
    qsa_pr_model.train()
    qsa_adv_model.train()
    value_model.train()
    

    print_model_q_values(-1, qsa_pr_model, qsa_adv_model, value_model, device, "Before MSE training begins", state_mapping, solver, game)

    for epoch in range(total_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        total_loss = 0
        total_pr_loss = 0
        total_adv_loss = 0
        total_batches = 0
        total_value_loss = 0
        
        # 在MSE训练开始前打印Q值
        if epoch == 0:
            print_model_q_values(epoch, qsa_pr_model, qsa_adv_model, value_model, device, "Before MSE training begins.", state_mapping, solver, game)

        for obs, acts, adv_acts, ret, rew, seq_len, player_ids in pbar:
            total_batches += 1

            qsa_pr_optimizer.zero_grad()
            qsa_adv_optimizer.zero_grad()
            value_model_optimizer.zero_grad()

            if seq_len.max() >= obs.shape[1]:
                seq_len -= 1

            # Set up variables
            batch_size = obs.shape[0]
            obs_len = obs.shape[1]
            
            obs = obs.view(batch_size, obs_len, -1).to(device)
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            
            current_player = 0
            padding_mask = (player_ids == -1)  # Padding timesteps
            no_action_mask = (acts[:, :, 2] == 1.0)  # "No action" timesteps  
            zero_action_mask = torch.all(acts == 0, dim=-1)  # All-zero padding timesteps
            not_player_turn = (player_ids != current_player)  # Not current player's turn
            
            # Combined invalid mask: True for timesteps to EXCLUDE from training
            acts_mask = padding_mask | zero_action_mask 
            
            # For adversarial actions, similar logic
            adv_no_action_mask = (adv_acts[:, :, 2] == 1.0)
            adv_zero_action_mask = torch.all(adv_acts == 0, dim=-1)
            adv_acts_mask = padding_mask | adv_zero_action_mask 

            
            ret = (ret / train_args['scale']).to(device)
            scale_factor = train_args['scale']
            seq_len = seq_len.to(device)

            # Calculate the losses at the different tages
            if epoch < mse_epochs:
                # MSE-based learning stage to learn general loss landscape
                ret_pr_pred = qsa_pr_model(obs, acts).view(batch_size, obs_len)
                ret_pr_loss = (((ret_pr_pred - ret) ** 2) * ~acts_mask).mean()
                
                ret_adv_pred = qsa_adv_model(obs, acts, adv_acts).view(batch_size, obs_len)
                ret_adv_loss = (((ret_adv_pred - ret) ** 2) * ~adv_acts_mask).mean()
                
                value_pred = value_model(obs).view(batch_size, obs_len)
                value_loss = (((value_pred - ret) ** 2) * ~acts_mask).mean()

                
                # Backpropagate - 分别训练每个模型
                qsa_pr_optimizer.zero_grad()
                ret_pr_loss.backward()
                qsa_pr_optimizer.step()
                
                qsa_adv_optimizer.zero_grad()
                ret_adv_loss.backward()
                qsa_adv_optimizer.step()
                
                value_model_optimizer.zero_grad()
                value_loss.backward()
                value_model_optimizer.step()

                # Update losses
                total_loss += ret_pr_loss.item() + ret_adv_loss.item() + value_loss.item()
                total_pr_loss += ret_pr_loss.item()
                total_adv_loss += ret_adv_loss.item()
                
                # 在MSE训练结束后打印Q值
                if epoch == mse_epochs - 1 and total_batches == len(dataloader):
                    print_model_q_values(epoch, qsa_pr_model, qsa_adv_model, value_model, device, "After MSE training ends", state_mapping, solver, game)

            elif epoch % 3 == 1:
               # Min step: adversary attempts to minimise at each node                        
                rewards = (ret[:, :-1] - ret[:, 1:]).view(batch_size, -1, 1)
                value_pred = value_model(obs)
                ret_adv_pred = qsa_adv_model(obs, acts, adv_acts)
                

                
                ret_tree_loss = (((value_pred[:, 1:].detach() + rewards - ret_adv_pred[:, :-1]) ** 2) * adv_acts_mask[:, :-1].unsqueeze(-1)).mean()
                ret_leaf_loss = (
                    (ret_adv_pred[range(batch_size), seq_len].flatten() - ret[range(batch_size), seq_len]) ** 2
                ).mean()
                ret_adv_loss = ret_tree_loss * (1 - train_args['leaf_weight']) + ret_leaf_loss * train_args['leaf_weight']                    
                # Backpropagate
                ret_adv_loss.backward()
                qsa_adv_optimizer.step()
                # Update losses
                total_loss += ret_adv_loss.item()
                total_adv_loss += ret_adv_loss.item()

            elif epoch % 3 == 0:
                

                # Get pc Q^π(s_{t+1}, a_{t+1}) - V(S) from protagonist model
                with torch.no_grad():
                    max_next_returns = qsa_pr_model(obs, acts)
                

                
                # Train Value Network to predict these Q-values
                value_pred = value_model(obs)  # V(s_t)
                
                td_error = max_next_returns.detach() - value_pred 

                value_loss = _expectile_fn(
                    td_error, 
                    acts_mask, 
                    alpha=train_args.get('alpha')
                )
                    
                value_loss.backward()
                value_model_optimizer.step()
                total_loss += value_loss.item()
                total_value_loss += value_loss.item()

            else:
                ret_pr_pred = qsa_pr_model(obs, acts)
                ret_adv_pred = qsa_adv_model(obs, acts, adv_acts)
                ret_pr_loss = _expectile_fn(ret_pr_pred - ret_adv_pred.detach(), acts_mask, train_args['alpha'])            
                # Backpropagate
                ret_pr_loss.backward()
                qsa_pr_optimizer.step()
                # Update losses
                total_loss += ret_pr_loss.item()
                total_pr_loss += ret_pr_loss.item()
                


            pbar.set_description(
                f"Epoch {epoch} | "
                f"Total Loss: {total_loss / total_batches:.6f} | "
                f"Pr Loss: {total_pr_loss / total_batches:.6f} | "
                f"Adv Loss: {total_adv_loss / total_batches:.6f} | "
                f"Value Loss: {total_value_loss / total_batches:.6f}"
            )
        
        # 每个epoch结束时打印Q值
        print_model_q_values(epoch, qsa_pr_model, qsa_adv_model, value_model, device, f"Epoch {epoch} finish", state_mapping, solver, game)
        

    # Get the learned return labels and prompt values (i.e. highest returns-to-go)
    print("\n=== Trajectory Relabeling ===")
    qsa_pr_model.eval()
    relabeled_trajs = []
    prompt_value = -np.inf
    scale_factor = train_args['scale']

    with torch.no_grad():
        for traj in tqdm(trajs, desc="Relabeling trajectories"):
            # Convert trajectory data to tensors
            obs_tensor = torch.from_numpy(np.array(traj.obs)).float().to(device).unsqueeze(0)  # (1, seq_len, obs_dim)
            acts_tensor = torch.from_numpy(np.array(traj.actions)).float().to(device).unsqueeze(0)  # (1, seq_len, 3)

            # Debug: Print shapes for first trajectory
            if len(relabeled_trajs) == 0:
                print(f"Debug - obs_tensor shape: {obs_tensor.shape}")
                print(f"Debug - acts_tensor shape: {acts_tensor.shape}")

            # Get Q-values for the (state, action) pairs in this trajectory
            # qsa_pr_model(obs, acts) should return Q(s_t, a_t) for each timestep
            returns_pred = qsa_pr_model(obs_tensor, acts_tensor)
            
            # Debug: Print prediction shape for first trajectory  
            if len(relabeled_trajs) == 0:
                print(f"Debug - returns_pred shape: {returns_pred.shape}")
            
            # Handle different possible output shapes
            if len(returns_pred.shape) == 3:
                if returns_pred.shape[-1] == 1:
                    # Shape: (1, seq_len, 1) -> (seq_len,)
                    returns_values = returns_pred.squeeze(-1).squeeze(0)
                else:
                    # This shouldn't happen with your model, but just in case
                    print(f"Warning: Unexpected shape {returns_pred.shape}")
                    returns_values = returns_pred.squeeze(0)
            else:
                # Shape: (1, seq_len) -> (seq_len,)
                returns_values = returns_pred.squeeze(0)
            
            returns = (returns_values * scale_factor).cpu().numpy()
            
            # Ensure it's a 1D array
            returns = returns.flatten()
            
            # Debug: Print values for first trajectory
            if len(relabeled_trajs) == 0:
                print(f"Debug - returns shape: {returns.shape}")
                print(f"Debug - returns values: {returns}")
                print(f"Debug - original minimax_returns_to_go: {getattr(traj, 'minimax_returns_to_go', 'Not found')}")
            
            # Update prompt value (use the first timestep's return-to-go as the "prompt value")
            if len(returns) > 0:
                current_prompt = returns[0]
                if prompt_value == -np.inf or current_prompt > prompt_value:
                    prompt_value = current_prompt
            
            # Create relabeled trajectory
            relabeled_traj = deepcopy(traj)
            relabeled_traj.minimax_returns_to_go = returns.tolist()
            relabeled_trajs.append(relabeled_traj)

    print(f"Relabeling complete. Prompt value: {prompt_value:.3f}")
    print(f"Total trajectories relabeled: {len(relabeled_trajs)}")

    return relabeled_trajs, np.round(prompt_value, decimals=3)