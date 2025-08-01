import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm.autonotebook import tqdm
from copy import deepcopy
from dataclasses import dataclass
from typing import List
from collections import namedtuple
from functools import partial
import pickle
import os
import yaml
from pathlib import Path

# --- Your Provided ARDTDataset Class ---
import torch
from torch.utils.data import IterableDataset

# Dummy return_labels for demonstration since it's from an external utility
def return_labels(traj, gamma, new_rewards):
    """
    Dummy implementation of return_labels for demonstration.
    In a real scenario, this would compute discounted returns.
    """
    if new_rewards and hasattr(traj, 'new_rewards') and traj.new_rewards is not None:
        rewards_to_use = traj.new_rewards
    else:
        rewards_to_use = traj.rewards

    # Calculate discounted returns
    discounted_returns = []
    current_return = 0
    for r in reversed(rewards_to_use):
        current_return = r + gamma * current_return
        discounted_returns.append(current_return)
    return discounted_returns[::-1]


class ARDTDataset(IterableDataset):
    """
    Modified class to provide an iterable dataset that processes trajectories sequentially.
    Supports using either original returns or relabeled minimax returns.
    """

    def __init__(
            self,
            trajs: list,
            horizon: int,
            gamma: float = 1,
            act_type: str = 'discrete', # 'discrete' or 'continuous'
            epoch_len: float = 1e5,
            new_rewards: bool = False,
            use_minimax_returns: bool = False,
    ):
        if not trajs:
            raise ValueError("Trajectory list is empty")

        self.trajs = trajs
        self.horizon = horizon
        self.act_type = act_type
        self.epoch_len = epoch_len
        self.new_rewards = new_rewards
        self.gamma = gamma
        self.use_minimax_returns = use_minimax_returns

        # Determine n_actions and n_adv_actions based on the first trajectory
        first_traj = trajs[0]

        # --- Corrected logic for n_actions and n_adv_actions ---
        # This handles cases where actions are scalar (e.g., [0, 1, 0])
        # or one-hot/vector (e.g., [[1,0], [0,1]])
        def get_action_dim(action_data):
            if isinstance(action_data, list):
                if not action_data: # Empty list, default to 1 (or handle as error)
                    return 1
                if isinstance(action_data[0], (int, float, np.number)): # Scalar action
                    return 1
                elif isinstance(action_data[0], (list, np.ndarray, torch.Tensor)): # Vector/one-hot action
                    return len(action_data[0])
            elif isinstance(action_data, np.ndarray):
                if action_data.ndim == 1: # Scalar action (N,)
                    return 1
                elif action_data.ndim > 1: # Vector/one-hot action (N, D)
                    return action_data.shape[-1]
            elif torch.is_tensor(action_data):
                if action_data.ndim == 1: # Scalar action (N,)
                    return 1
                elif action_data.ndim > 1: # Vector/one-hot action (N, D)
                    return action_data.shape[-1]
            raise TypeError(f"Unsupported action data type: {type(action_data)}")

        self.n_actions = get_action_dim(first_traj.actions)
        self.n_adv_actions = get_action_dim(first_traj.adv_actions)

        # Validate consistency across all trajectories (optional but good practice)
        # This loop ensures all trajectories have consistent action dimensions
        for i, traj in enumerate(trajs):
            if get_action_dim(traj.actions) != self.n_actions:
                raise ValueError(f"Inconsistent action dimension in trajectory {i}: {get_action_dim(traj.actions)} vs {self.n_actions}")
            if get_action_dim(traj.adv_actions) != self.n_adv_actions:
                raise ValueError(f"Inconsistent adv_action dimension in trajectory {i}: {get_action_dim(traj.adv_actions)} vs {self.n_adv_actions}")


    def segment_generator_sequential(self, start_idx: int, end_idx: int):
        for traj_idx in range(start_idx, end_idx):
            traj = self.trajs[traj_idx]

            # Use minimax returns if specified, otherwise calculate from rewards
            if self.use_minimax_returns:
                if not hasattr(traj, 'minimax_returns_to_go') or traj.minimax_returns_to_go is None:
                    print(f"Warning: Trajectory {traj_idx} missing minimax_returns_to_go. Falling back to original returns.")
                    rets_data = return_labels(traj, self.gamma, self.new_rewards)
                else:
                    rets_data = np.array(traj.minimax_returns_to_go, dtype=np.float32)
            else:
                rets_data = return_labels(traj, self.gamma, self.new_rewards)

            obs_np = np.array(traj.obs, dtype=np.float32)
            actions_np = np.array(traj.actions)
            adv_actions_np = np.array(traj.adv_actions)
            
            if self.n_actions == 1 and actions_np.ndim == 1:
                actions = torch.tensor(actions_np[:, None], dtype=torch.float32)
            else:
                actions = torch.tensor(actions_np, dtype=torch.float32)

            if self.n_adv_actions == 1 and adv_actions_np.ndim == 1:
                adv_actions = torch.tensor(adv_actions_np[:, None], dtype=torch.float32) # Make (N, 1)
            else: # Already (N, D) or (N,) and n_adv_actions > 1
                adv_actions = torch.tensor(adv_actions_np, dtype=torch.float32) # Ensure float32


            obs = torch.tensor(obs_np, dtype=torch.float32)
            rets = torch.tensor(rets_data, dtype=torch.float32)
            true_seq_length = obs.shape[0] # This is the true, unpadded length of the episode

            # --- Padding to self.horizon for batching ---
            # Initialize padded tensors with correct shapes based on inferred dimensions
            padded_obs = torch.zeros((self.horizon, obs.shape[1]), dtype=torch.float32)
            padded_actions = torch.zeros((self.horizon, self.n_actions), dtype=torch.float32)
            padded_adv_actions = torch.zeros((self.horizon, self.n_adv_actions), dtype=torch.float32)
            padded_rets = torch.zeros(self.horizon, dtype=torch.float32) # Returns are scalar

            segment_len = min(true_seq_length, self.horizon)

            padded_obs[:segment_len] = obs[:segment_len]
            padded_actions[:segment_len] = actions[:segment_len]
            padded_adv_actions[:segment_len] = adv_actions[:segment_len]
            padded_rets[:segment_len] = rets[:segment_len]

            yield (
                padded_obs,
                padded_actions, # Renamed from padded_acts for clarity
                padded_adv_actions, # Renamed from padded_adv_acts for clarity
                padded_rets,
                torch.tensor(true_seq_length, dtype=torch.int32)
            )

    def __iter__(self):
        worker_info = torch.utils.data.get_worker_info()
        num_trajs = len(self.trajs)

        if worker_info is None:
            start_idx = 0
            end_idx = num_trajs
        else:
            per_worker = int(torch.ceil(torch.tensor(num_trajs / worker_info.num_workers)))
            worker_id = worker_info.id
            start_idx = worker_id * per_worker
            end_idx = min(start_idx + per_worker, num_trajs)

        return self.segment_generator_sequential(start_idx, end_idx)

    def __len__(self):
        return len(self.trajs)