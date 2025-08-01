import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from offline_setup.base_offline_env import BaseOfflineEnv
from tqdm import tqdm


def worst_case_env_step(
    state: np.array,
    action: np.array, # This is the agent's chosen action
    timestep: int,
    env_name: str,
    env # The KuhnPokerEnv instance
) -> tuple[np.array, float, bool, bool, dict]:
    """
    Simulate worst-case adversaries for Kuhn Poker by feeding the correct actions to env.step.
    """
    # Ensure agent's action is int
    if isinstance(action, np.ndarray):
        action = np.argmax(action) if action.size > 1 else int(action.item())
    elif isinstance(action, (np.integer, int, torch.Tensor)):
        action = int(action)

    if env_name == "kuhn_poker":
        current_player_turn = env.player_turn # Get current player from env

        if current_player_turn == 0: # It's agent's turn
            # Agent's action is `action`
            new_state, reward, terminated, truncated, info = env.step(action)
            
            # After agent's step, it might become adversary's turn if game not done
            if not (terminated or truncated) and env.player_turn == 1:
                # It's now the adversary's turn within the same outer `worst_case_env_step` call.
                # Adversary's worst-case action (always bet/call = 1)
                adv_action_to_take = 1 # Hardcoded for worst-case
                
                # Now, step the environment with the adversary's action
                new_state, reward, terminated, truncated, info_adv = env.step(adv_action_to_take)
                info.update(info_adv) # Merge info from adversary's step
                
            info["adv_action"] = info.get("adv_action", -1) # Ensure adv_action is set in info

        elif current_player_turn == 1: # Should not happen if game flow is correctly managed above
                                      # meaning the wrapper completes all internal turns in one go.
            # If this path is hit, it means the game state in env is out of sync or max_ep_len is too high
            # for the Kuhn Poker game.
            print(f"WARNING: worst_case_env_step called for Kuhn Poker but env.player_turn is already 1. This indicates a potential flow issue.")
            # Still, process the agent's action as if it's the current player, or raise error.
            # For robustness, we can try to proceed assuming agent's action (though it's P1's turn)
            # This is where game logic gets tricky.
            # Best is that `worst_case_env_step` handles all sub-turns.
            new_state, reward, terminated, truncated, info = env.step(action) # Assuming agent's action is passed
            
        else: # Game ended, or invalid player_turn
            print(f"WARNING: worst_case_env_step called when game is likely done (player_turn: {current_player_turn}). Returning last state.")
            return state, 0.0, True, False, {} # Return done=True with 0 reward to end episode.
            
        return new_state, reward, terminated, truncated, info

def evaluate_episode(
    env_instance,
    env_name: str,
    state_dim: int,
    act_dim: int,
    action_type: str,
    model: torch.nn.Module,
    model_type: str,
    max_ep_len: int,
    scale: float,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    target_return: float,
    adv_act_dim: int = None,
    normalize_states: bool = False,
    worst_case: bool = True,
    with_noise: bool = False,
    device: str = 'cpu',
) -> tuple[float, int]:
    model.eval()
    model.to(device=device)

    # Ensure state_mean and state_std are torch tensors on the correct device
    state_mean_tensor = torch.from_numpy(state_mean).to(device=device, dtype=torch.float32)
    state_std_tensor = torch.from_numpy(state_std).to(device=device, dtype=torch.float32)

    # Reset the environment
    state = env_instance.reset()
    if with_noise:
        state = state + np.random.normal(0, 0.1, size=state.shape)

    # Infer adv_act_dim if not provided
    if adv_act_dim is None:
        if hasattr(env_instance.adv_action_space, 'n'):
            adv_act_dim = env_instance.adv_action_space.n
        elif hasattr(env_instance.adv_action_space, 'shape') and env_instance.adv_action_space.shape:
            adv_act_dim = env_instance.adv_action_space.shape[0]
        else:
            adv_act_dim = 1

    # Initialize histories with consistent batch dimensions
    current_state = torch.from_numpy(state).to(device=device, dtype=torch.float32).reshape(1, state_dim)
    states_history = current_state  # Shape: (1, state_dim)
    actions_history = torch.zeros((1, 0, act_dim), device=device, dtype=torch.float32)  # Shape: (1, 0, act_dim)
    adv_actions_history = torch.zeros((1, 0, adv_act_dim), device=device, dtype=torch.float32)  # Shape: (1, 0, adv_act_dim)
    rewards_history = torch.zeros((1, 0), device=device, dtype=torch.float32)  # Shape: (1, 0)
    returns_to_go_history = torch.tensor([[target_return]], device=device, dtype=torch.float32)  # Shape: (1, 1)
    timesteps_history = torch.tensor([[0]], device=device, dtype=torch.long)  # Shape: (1, 1)

    episode_return, episode_length = 0, 0

    for t in range(max_ep_len):
        # Current sequence length
        seq_len = states_history.shape[0] if len(states_history.shape) == 2 else states_history.shape[1]
        
        # Prepare inputs for the model - ensure consistent batch dimensions
        if len(states_history.shape) == 2:  # (seq_len, state_dim)
            states_input = states_history.unsqueeze(0)  # (1, seq_len, state_dim)
        else:  # Already has batch dimension
            states_input = states_history
            
        # Pad actions for next prediction
        next_action_pad = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
        actions_input = torch.cat([actions_history, next_action_pad], dim=1)  # (1, seq_len+1, act_dim)
        
        next_adv_action_pad = torch.zeros((1, 1, adv_act_dim), device=device, dtype=torch.float32)
        adv_actions_input = torch.cat([adv_actions_history, next_adv_action_pad], dim=1)  # (1, seq_len+1, adv_act_dim)
        
        # Normalize states for model input
        if normalize_states:
            states_normalized = (states_input - state_mean_tensor.unsqueeze(0)) / state_std_tensor.unsqueeze(0)
        else:
            states_normalized = states_input

        # Get action from model based on model_type
        if model_type == 'dt':
            # Decision Transformer forward pass
            states_pred, action_preds, return_preds = model(
                states=states_normalized,  # (1, seq_len, state_dim)
                actions=actions_input,     # (1, seq_len+1, act_dim)
                returns_to_go=returns_to_go_history.unsqueeze(-1),  # (1, seq_len, 1)
                timesteps=timesteps_history,  # (1, seq_len)
            )
            # Get the action prediction for the current timestep
            action = action_preds[0, -1]  # (act_dim,)

        elif model_type == 'adt':
            action_pred_output = model(
                states=states_normalized,
                actions=actions_input,
                adv_actions=adv_actions_input,
                returns_to_go=returns_to_go_history.unsqueeze(-1),
                timesteps=timesteps_history,
            )
            action = action_pred_output[0, -1]
            
        elif model_type == "bc":
            action_pred_output = model(states=states_normalized)
            action = action_pred_output[0, -1]

        # Handle discrete and continuous action spaces
        if action_type == 'discrete':
            act_probs = F.softmax(action, dim=-1)
            action_idx = Categorical(probs=act_probs).sample()
            
            # Create one-hot action for history
            one_hot_action = torch.zeros((1, 1, act_dim), device=device, dtype=torch.float32)
            one_hot_action[0, 0, action_idx] = 1.0
            
            action_for_step = action_idx.item()
            
        else:
            one_hot_action = action.unsqueeze(0).unsqueeze(0)  # (1, 1, act_dim)
            action_for_step = action.detach().cpu().numpy()
            
        if worst_case and env_name in ["gambling", "toy", "mstoy", "kuhn_poker", "new_mstoy"]:
            state, reward, terminated, truncated, infos = worst_case_env_step(
                state, action_for_step, t, env_name, env_instance
            )
        else:
            # Use the proper gym interface with 5 return values
            step_result = env_instance.step(action_for_step)
            if len(step_result) == 4:  # Old gym interface
                state, reward, terminated, infos = step_result
                truncated = False
            else:  # New gym interface
                state, reward, terminated, truncated, infos = step_result

        done = terminated or truncated


        # Handle adversarial action
        adv_a = infos.get("adv", infos.get("adv_action", None))
        if adv_a is not None and adv_a != -1:  # -1 means no adversarial action
            if action_type == 'discrete':
                one_hot_adv_action = torch.zeros((1, 1, adv_act_dim), device=device, dtype=torch.float32)
                if isinstance(adv_a, np.ndarray):
                    adv_a = adv_a.item()
                one_hot_adv_action[0, 0, adv_a] = 1.0
            else:
                one_hot_adv_action = torch.tensor(adv_a, device=device, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        else:
            one_hot_adv_action = torch.zeros((1, 1, adv_act_dim), device=device, dtype=torch.float32)

        # Update histories - maintain consistent dimensions
        # Add current action and adversarial action to history
        actions_history = torch.cat([actions_history, one_hot_action], dim=1)
        adv_actions_history = torch.cat([adv_actions_history, one_hot_adv_action], dim=1)
        
        # Add current reward to history
        current_reward = torch.tensor([[reward]], device=device, dtype=torch.float32)
        rewards_history = torch.cat([rewards_history, current_reward], dim=1)
        
        # Add next state to history
        next_state = torch.from_numpy(state).to(device=device, dtype=torch.float32).reshape(1, state_dim)
        if len(states_history.shape) == 2:  # (seq_len, state_dim)
            states_history = torch.cat([states_history, next_state], dim=0)
        else:  # (1, seq_len, state_dim)
            states_history = torch.cat([states_history, next_state.unsqueeze(0)], dim=1)

        # Update returns-to-go
        next_return_to_go = target_return - (reward / scale)
        next_rtg = torch.tensor([[next_return_to_go]], device=device, dtype=torch.float32)
        returns_to_go_history = torch.cat([returns_to_go_history, next_rtg], dim=1)

        # Update timesteps
        next_timestep = torch.tensor([[t + 1]], device=device, dtype=torch.long)
        timesteps_history = torch.cat([timesteps_history, next_timestep], dim=1)

        episode_return += reward
        episode_length += 1
        target_return = next_return_to_go

        if done:
            break

    return episode_return, episode_length


def evaluate(
        env_name: str,
        task: BaseOfflineEnv,
        num_eval_episodes: int,
        state_dim: int,
        act_dim: int,
        adv_act_dim: int,
        action_type: str,
        model: torch.nn.Module,
        model_type: str,
        max_ep_len: int,
        scale: float,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        target_return: float,
        batch_size: int = 1,
        normalize_states: bool = True,
        device: str = 'cpu'
    ) -> tuple[list[float], list[int]]:
    """
    Evaluate the model over multiple episodes using the provided evaluation logic.
    """
    returns, lengths = [], []
    for episode_idx in tqdm(range(num_eval_episodes), desc=f"Evaluating {model_type} for target {target_return:.2f}"):
        # Create a fresh environment instance for each episode if needed
        # It's generally safer to recreate the env for each episode if it's not a singleton
        # Check if 'task' is an instantiated object or a class
        if hasattr(task, 'reset') and callable(task.reset):
            current_env_instance = task
        else:
            # Assuming 'task' is a class that needs to be instantiated if it doesn't have a reset method
            # This handles the case where you pass KuhnPokerEnv (the class) vs. my_kuhn_poker_env (the instance)
            try:
                current_env_instance = task() # Try to instantiate it
            except TypeError:
                print(f"Warning: 'task' object {type(task)} does not seem to be an environment instance or a callable class. Using it directly.")
                current_env_instance = task # Fallback to using it directly if instantiation fails

        with torch.no_grad():
            ret, length = evaluate_episode(
                current_env_instance,
                env_name,
                state_dim,
                act_dim,
                action_type,
                model,
                model_type,
                max_ep_len,
                scale,
                state_mean,
                state_std,
                target_return / scale, # Ensure target_return is scaled for the model
                adv_act_dim=adv_act_dim,
                normalize_states=normalize_states,
                worst_case=True, # Assuming you always want worst-case evaluation for these envs
                device=device
            )
        returns.append(ret)
        lengths.append(length)

    return returns, lengths