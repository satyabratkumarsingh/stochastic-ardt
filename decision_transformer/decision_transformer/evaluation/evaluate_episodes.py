import gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm

from offline_setup.base_offline_env import BaseOfflineEnv

def worst_case_env_step(
    state: np.array,
    action: np.array,
    timestep: int,
    env_name: str,
    env: BaseOfflineEnv
) -> tuple[np.array, float, bool, bool, dict]:
    """
    Simulate worst-case adversaries for toy environments, including Kuhn Poker.

    Args:
        state (np.array): Current state, shape [10] for Kuhn Poker.
        action (np.array): Current action, one-hot [2] or scalar (0=fold, 1=bet).
        timestep (int): Current timestep (0-based).
        env_name (str): Name of the environment ("kuhn_poker").
        env (BaseOfflineEnv): Environment object.

    Returns:
        tuple: (new_state, reward, done, truncated, info)
    """
    # Convert action to int
    if isinstance(action, np.ndarray) and action.size > 1:
        action = np.argmax(action)
    elif isinstance(action, (np.ndarray, np.integer, int)):
        action = int(action)

    # Initialize
    new_state = state
    reward = 0.0
    done = False
    truncated = False
    info = {'adv_action': None}

    if env_name == "kuhn_poker":
        card_idx = np.argmax(state[:3])  # J=0, Q=1, K=2
        player_card = ['J', 'Q', 'K'][card_idx]
        
        if timestep == 0:
            env.step(action)
            if action == 0:  # Fold
                reward = -1.0
                done = True
                adv_action = 0
            else:  # Bet
                adv_action = 1  # Worst-case: bet
                new_state = np.zeros_like(state)
                new_state[card_idx] = 1.0
                new_state[3] = 1.0
                new_state[4] = 1.0
            info['adv_action'] = adv_action
        elif timestep == 1:
            if env.actions_taken[-1] == 1:  # Adversary bet
                if action == 0:  # Fold
                    reward = -1.0
                    done = True
                else:  # Bet
                    player_rank = ['J', 'Q', 'K'].index(player_card)
                    opponent_rank = ['J', 'Q', 'K'].index(env.opponent_card)
                    reward = 2.0 if player_rank > opponent_rank else -2.0
                    done = True
                new_state = np.zeros_like(state)
                new_state[card_idx] = 1.0
                new_state[3] = 1.0
                new_state[4] = 1.0
                new_state[5] = 1.0
            else:  # Adversary folded
                reward = 1.0
                done = True
            info['adv_action'] = env.actions_taken[-1]
        else:
            raise RuntimeError("Kuhn Poker timestep exceeds max length (2).")
    else:
        new_state_idx = -1
        adv_action = np.random.choice(2, 1)
        _, reward, terminated, _ = env.step(action)
        done = terminated
        if env_name == "gambling":
            if timestep == 0:
                if action == 0:
                    w_reward = -15
                elif action == 1:
                    w_reward = -6
                else:
                    w_reward = 1
            elif timestep == 1:
                reward = w_reward
                # Remove assert done to handle cases where episode continues
        elif env_name == "toy":
            if timestep == 0:
                if action == 0:
                    env.w_reward = 0
                else:
                    env.w_reward = 1
            else:
                reward = env.w_reward
                if not isinstance(action, torch.Tensor):
                    action = torch.tensor(action)
                if reward != 0 or torch.any(action != -10):
                    done = True
        elif env_name == "mstoy":
            done = False
            if timestep == 0:
                if action > 0:
                    reward = 4
                    done = True
                else:
                    new_state_idx = 1
                    reward = 0
                    adv_action = 0
            else:
                reward = env.reward_list[action + (state.argmax() - 1) * 3]
                done = True
        elif env_name == "new_mstoy":
            done = False
            if timestep == 0:
                if action > 0:
                    reward = 4
                    done = True
                else:
                    new_state_idx = 1
                    reward = 0
            else:
                reward = env.reward_list[action * 2 + (state.argmax() - 1) * 3]
                done = True
        else:
            raise RuntimeError(f"Unsupported environment: {env_name}")
        new_state = np.eye(state.size)[new_state_idx] if new_state_idx != -1 else state
        info = {"adv_action": adv_action}

    return new_state, reward, done, truncated, info

# def worst_case_env_step(
#         state: np.array, 
#         action: np.array, 
#         timestep: int,
#         env_name: str, 
#         env: BaseOfflineEnv
#     ) -> tuple[np.array, float, bool, bool, dict]:
#     """
#     Function to simulate worst-case adversaries in toy environments.

#     Args:
#         state (np.array): Current state.
#         action (np.array): Current action.
#         timestep (int): Current timestep.
#         env_name (str): Name of the environment.
#         env (BaseOfflineEnv): Environment object.

#     Returns:
#         tuple: New state, reward, done, truncated, and info (incl. adversarial action).
#     """
#     new_state_idx = -1
#     adv_action = np.random.choice(2, 1)
    
#     _, reward, terminated, _ = env.step(action)
#     done = terminated
    
#     if env_name == "gambling":
#         if timestep == 0:
#             if action == 0:
#                 w_reward = -15 
#             elif action == 1:
#                 w_reward = -6
#             else:
#                 w_reward = 1
#         elif timestep == 1:
#             reward = w_reward
#             assert done
#     elif env_name == "toy":
#         if timestep == 0:
#             if action == 0:
#                 env.w_reward = 0
#             else:
#                 env.w_reward = 1
#         else:
#             #print(env.w_reward)
#             reward = env.w_reward
#             if not isinstance(action, torch.Tensor):
#                 action = torch.tensor(action)
#             if reward != 0 or torch.any(action != -10):  # implies it's a real move
#             # only then check for done
#                 if timestep == 1:
#                     assert done
#     elif env_name == "mstoy":
#         done = False
#         if timestep == 0:
#             if action > 0:
#                 reward = 4
#                 done = True
#             else:
#                 new_state_idx = 1
#                 reward = 0
#                 adv_action = 0
#         else:
#             reward = env.reward_list[action + (state.argmax() - 1) * 3]
#             done = True
#     elif env_name == "new_mstoy":
#         done = False
#         if timestep == 0:
#             if action > 0:
#                 reward = 4
#                 done = True
#             else:
#                 new_state_idx = 1
#                 reward = 0
#         else:
#             reward = env.reward_list[action * 2 + (state.argmax() - 1) * 3]
#             done = True
#     else:
#         raise RuntimeError("Environment Error.")

#     new_state = np.eye(state.size)[new_state_idx] if new_state_idx != -1 else state
#     return new_state, reward, done, False, {"adv_action": adv_action}
  

def evaluate_episode(
    env,
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

    state_mean = torch.from_numpy(state_mean).to(device=device)
    state_std = torch.from_numpy(state_std).to(device=device)

    state = env.reset()
    if with_noise:
        state = state + np.random.normal(0, 0.1, size=state.shape)

    if not adv_act_dim:
        adv_act_dim = act_dim

    # Initialize histories and variables
    states = torch.from_numpy(state).reshape(1, state_dim).to(device=device, dtype=torch.float32)
    actions = torch.zeros((0, act_dim), device=device, dtype=torch.float32)
    adv_actions = torch.zeros((0, adv_act_dim), device=device, dtype=torch.float32)
    rewards = torch.zeros(0, device=device, dtype=torch.float32)
    timesteps = torch.tensor(0, device=device, dtype=torch.long).reshape(1, 1)
    target_return = torch.tensor(target_return, device=device, dtype=torch.float32).reshape(1, 1)

    # Evaluate episode
    episode_return, episode_length = 0, 0

    for t in range(max_ep_len):
        actions = torch.cat([actions, torch.zeros((1, act_dim), device=device)], dim=0)
        adv_actions = torch.cat([adv_actions, torch.zeros((1, adv_act_dim), device=device)], dim=0)
        rewards = torch.cat([rewards, torch.zeros(1, device=device)])

        if normalize_states:
            normalized_states = (states.to(dtype=torch.float32) - state_mean) / state_std
        else:
            normalized_states = states.to(dtype=torch.float32)
        
        if model_type == 'dt':
            action = model.get_action(
                states=normalized_states,
                actions=actions,
                rewards=rewards,
                returns_to_go=target_return,
                timesteps=timesteps,
                batch_size=1
            )[0, -1]
        elif model_type == 'adt':
            action = model.get_action(
                states=normalized_states,
                actions=actions,
                adv_actions=adv_actions,
                rewards=rewards,
                returns_to_go=target_return,
                timesteps=timesteps,
                batch_size=1
            )[0, -1]
        elif model_type == "bc":
            action = model.get_action(
                states=normalized_states,
                actions=actions,
                rewards=rewards,
            )

        # Handle discrete and continuous action spaces
        if action_type == 'discrete':
            act_probs = F.softmax(action, dim=-1)
            action_idx = Categorical(probs=act_probs).sample()
            one_hot_action = torch.zeros(1, act_dim, device=device, dtype=torch.float32)
            one_hot_action[0, action_idx] = 1.0
            actions[-1] = one_hot_action
            action_for_step = one_hot_action.detach().cpu().numpy()[0]  # [2]
        else:
            actions[-1] = action
            action_for_step = action.detach().cpu().numpy()

        if worst_case and env_name in ["gambling", "toy", "mstoy", "kuhn_poker"]:
            state, reward, terminated, truncated, infos = worst_case_env_step(state, action_for_step, t, env_name, env)
        else:
            state, reward, terminated, truncated, infos = env.step(action_for_step)

        done = terminated or truncated

        # Handle adversarial action
        adv_a = infos.get("adv", infos.get("adv_action", None))
        if action_type == 'discrete':
            one_hot_adv_action = torch.zeros(1, adv_act_dim, device=device, dtype=torch.float32)
            if adv_a is not None:
                one_hot_adv_action[0, adv_a] = 1.0
            adv_actions[-1] = one_hot_adv_action
        else:
            if adv_a is not None:
                adv_actions[-1] = torch.from_numpy(adv_a).to(device)

        # Update states, rewards, and timesteps
        curr_state = torch.from_numpy(state).to(device=device).reshape(1, state_dim)
        states = torch.cat([states, curr_state], dim=0)
        rewards[-1] = reward

        pred_return = target_return[0, -1] - (reward / scale)
        target_return = torch.cat([target_return, pred_return.reshape(1, 1)], dim=1)

        timesteps = torch.cat([timesteps, torch.ones((1, 1), device=device, dtype=torch.long) * (t + 1)], dim=1)

        episode_return += reward
        episode_length += 1

        if done:
            break

    return episode_return, episode_length


def evaluate(
        env_name: str,
        task,
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
    Evaluate the model over multiple episodes.

    Args:
        env_name (str): The name of the environment.
        task: The task instance.
        num_eval_episodes (int): Number of evaluation episodes.
        state_dim (int): Dimension of the state space.
        act_dim (int): Dimension of the action space.
        adv_act_dim (int): Dimension of the adversarial action space.
        action_type (str): Type of action space ('discrete' or 'continuous').
        model (torch.nn.Module): The model used for decision-making.
        model_type (str): The type of model ('dt', 'adt', or 'bc').
        max_ep_len (int): Maximum length of each episode.
        scale (float): Scale for normalization of returns.
        state_mean (np.ndarray): Mean of the states for normalization.
        state_std (np.ndarray): Standard deviation of the states for normalization.
        target_return (float): Target return value for the evaluation.
        batch_size (int, optional): Batch size for the evaluation. Default is 1.
        normalize_states (bool, optional): Whether to normalize the states. Default is True.
        device (str, optional): Device to run the model on. Default is 'cpu'.

    Returns:
        tuple: List of returns and lengths for each episode.
    """
    test_env = task.test_env_cls()

    returns, lengths = [], []
    for _ in tqdm(range(num_eval_episodes)):
        with torch.no_grad():
            ret, length = evaluate_episode(
                test_env,
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
                target_return / scale,
                adv_act_dim=adv_act_dim,
                normalize_states=normalize_states,
                worst_case=True,
                device=device
            )
        returns.append(ret)
        lengths.append(length)

    return returns, lengths
