
import gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from data_loading.load_mujoco import Trajectory
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


# Option 1: Use theoretical Nash equilibrium values for Kuhn Poker
KUHN_NASH_EQUILIBRIUM = {
    "0pb": [1.0, 0.0, 0.0],  # Pass always
    "1pb": [2/3, 1/3, 0.0],  # Mixed
    "2pb": [0.0, 1.0, 0.0]   # Bet always
}

def print_model_q_values(
    qsa_pr_model, qsa_adv_model, value_model, device, stage_name, state_mapping,
    cfr_solver=None, game=None, log_adv_matrix=True, save_path="epoch_stats.json"
):
    print(f"\n=== {stage_name} ===")
    stats = {}
    l1_devs = []

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

            # --- Protagonist Q-values ---
            pr_qs = {}
            for action_name, act_vec in actions.items():
                act_tensor = torch.tensor([[[act_vec[0], act_vec[1], act_vec[2]]]], dtype=torch.float32, device=device)
                q_val = qsa_pr_model(obs_tensor, act_tensor)
                pr_qs[action_name] = q_val.item()

            # --- Nash probabilities using theoretical values ---
            nash_str = ""
            nash_probs = [None, None, None]
            deviation = None
            
            if state_str in KUHN_NASH_EQUILIBRIUM:
                nash_probs = KUHN_NASH_EQUILIBRIUM[state_str]
                nash_str = f" | Nash: Pass={nash_probs[0]:.3f}, Bet={nash_probs[1]:.3f}"
                
                # Include NoAction dimension in softmax
                q_arr = np.array([pr_qs["Pass"], pr_qs["Bet"], 0.0])  # NoAction placeholder
                exp_q = np.exp(q_arr - np.max(q_arr))
                softmax_policy = exp_q / np.sum(exp_q)
                
                deviation = float(np.sum(np.abs(softmax_policy - nash_probs)))
                l1_devs.append(deviation)

            # --- Adversary payoff matrix ---
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
                        
                print(f"State {state_str} ({card_name}, {history}) | Protagonist Q: "
                      f"Pass={pr_qs['Pass']:.3f}, Bet={pr_qs['Bet']:.3f}{nash_str}")
                print(f"Adversary payoff matrix:\n{adv_matrix}")

            # --- State value ---
            v = value_model(obs_tensor)
            value = v.item()
            print(f"State Value V(s): {value:.3f}\n")

            # --- Save stats ---
            stats[state_str] = {
                "card": card_name,
                "history": history,
                "protagonist_q": pr_qs,
                "nash_probs": nash_probs,
                "l1_deviation": deviation,
                "state_value": value,
                "adv_matrix": adv_matrix.tolist() if adv_matrix is not None else None
            }

    # --- Compute mean L1 deviation for epoch ---
    mean_l1 = float(np.mean(l1_devs)) if l1_devs else None
    if mean_l1 is not None:
        print(f"Mean L1 deviation from Nash for 'pb' states: {mean_l1:.4f}")
    else:
        print("No 'pb' states found for L1 deviation calculation.")

    # Save all epoch stats to file
    stats["mean_l1_deviation"] = mean_l1
    with open(save_path, "w") as f:
        json.dump(stats, f, indent=2)
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
    dataset = ARDTDataset(trajs, max_len, gamma=gamma, act_type=action_type)

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
    

    print_model_q_values(qsa_pr_model, qsa_adv_model, value_model, device, "Before training starts", state_mapping, solver, game)
    
    qsa_pr_model.train()
    qsa_adv_model.train()
    value_model.train()
    

    print_model_q_values(qsa_pr_model, qsa_adv_model, value_model, device, "Before MSE training begins", state_mapping, solver, game)

    for epoch in range(total_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        total_loss = 0
        total_pr_loss = 0
        total_adv_loss = 0
        total_batches = 0
        total_value_loss = 0
        
        # 在MSE训练开始前打印Q值
        if epoch == 0:
            print_model_q_values(qsa_pr_model, qsa_adv_model, value_model, device, "Before MSE training begins.", state_mapping, solver, game)

        for obs, acts, adv_acts, ret, seq_len in pbar:
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
            
            
            acts_mask = (acts[:, :, 2] == 1.0)
            adv_acts_mask = (adv_acts[:, :, 2] == 1.0) 
            
            ret = (ret / train_args['scale']).to(device)
            scale_factor = train_args['scale']
            seq_len = seq_len.to(device)

            # Adjustment for initial prompt learning
            # obs[:, 0] = obs[:, 1]
            # ret[:, 0] = ret[:, 1]
            # acts_mask[:, 0] = True
            # adv_acts_mask[:, 0] = True

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
                    print_model_q_values(qsa_pr_model, qsa_adv_model, value_model, device, "After MSE training ends", state_mapping, solver, game)

            elif epoch % 3 == 1:
               # Min step: adversary attempts to minimise at each node                        
                rewards = (ret[:, :-1] - ret[:, 1:]).view(batch_size, -1, 1)
                value_pred = value_model(obs)
                ret_adv_pred = qsa_adv_model(obs, acts, adv_acts)
                

                
                ret_tree_loss = (((value_pred[:, 1:].detach() + rewards - ret_adv_pred[:, :-1]) ** 2) * ~adv_acts_mask[:, :-1].unsqueeze(-1)).mean()
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
        print_model_q_values(qsa_pr_model, qsa_adv_model, value_model, device, f"Epoch {epoch} finish", state_mapping, solver, game)
        

    # Get the learned return labels and prompt values (i.e. highest returns-to-go)
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
                prompt_value = float(returns[0])
                
            relabeled_traj = deepcopy(traj)
            relabeled_traj.minimax_returns_to_go = returns.tolist()
            relabeled_trajs.append(relabeled_traj)


    print(f"Relabeling complete. Prompt value: {prompt_value:.3f}")
    
    return relabeled_trajs, np.round(prompt_value, decimals=3)