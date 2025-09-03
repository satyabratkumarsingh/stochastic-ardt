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
import torch
from torch.utils.data import IterableDataset

def get_true_sequence_length(traj):
    """
    Get the actual episode length based on done flags.
    Returns the index of the first True in dones + 1, or total length if no done flags.
    """
    if hasattr(traj, 'dones') and traj.dones is not None:
        dones = traj.dones
        # Find first True (episode end)
        for i, done in enumerate(dones):
            if done:
                return i + 1  # +1 because we include the terminal step
        # If no True found, use full length (episode didn't terminate)
        return len(dones)
    else:
        # Fallback to observation count if no done flags available
        return len(traj.obs)

def return_labels(traj, gamma, new_rewards, use_minimax_returns=False):
    """Calculate returns from trajectory rewards"""
    # Choose which rewards/returns to use
    if use_minimax_returns and hasattr(traj, 'minimax_returns_to_go') and traj.minimax_returns_to_go is not None:
        rewards_to_use = traj.minimax_returns_to_go
    elif new_rewards and hasattr(traj, 'new_rewards') and traj.new_rewards is not None:
        rewards_to_use = traj.new_rewards
    else:
        rewards_to_use = traj.rewards

    # Calculate discounted returns from chosen rewards/values
    discounted_returns = []
    current_return = 0
    for r in reversed(rewards_to_use):
        current_return = r + gamma * current_return
        discounted_returns.append(current_return)
        
    return discounted_returns[::-1], rewards_to_use

def add_player_ids_to_trajectory(traj):
    """
    Add player_ids to a single trajectory based on action patterns for 2-player games
    Assumes alternating turns starting with player 0
    """
    if hasattr(traj, 'player_ids') and traj.player_ids is not None:
        return traj
        
    player_ids = []
    current_player = 0
    
    for i, action in enumerate(traj.actions):
        # Convert to numpy array if needed
        if isinstance(action, (list, tuple)):
            action = np.array(action)
        elif torch.is_tensor(action):
            action = action.numpy()
            
        # If action is "no action" [0,0,1], it's the opponent's turn
        if len(action) >= 3 and action[2] == 1.0:  # No action
            player_ids.append(1 - current_player)  # Opponent
        else:
            player_ids.append(current_player)  # Current player
            # Switch players for next turn (in alternating games)
            current_player = 1 - current_player
            
    traj.player_ids = player_ids
    return traj

def add_player_ids_to_trajectories(trajs):
    """Add player_ids to all trajectories"""
    return [add_player_ids_to_trajectory(traj) for traj in trajs]

class ARDTDataset(IterableDataset):
    """
    Modified class to provide an iterable dataset that processes trajectories sequentially.
    Supports using either original returns or relabeled minimax returns, and now returns
    both returns-to-go and raw rewards. Also handles player_ids for turn-based games.
    Uses done flags to determine true episode length.
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
            include_player_ids: bool = False,
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
        self.include_player_ids = include_player_ids

        # Auto-generate player_ids if requested but missing
        if self.include_player_ids:
            self.trajs = add_player_ids_to_trajectories(self.trajs)

        def get_action_dim(action_data):
            if isinstance(action_data, list):
                if not action_data: 
                    return 1
                if isinstance(action_data[0], (int, float, np.number)):
                    return 1
                elif isinstance(action_data[0], (list, np.ndarray, torch.Tensor)):
                    return len(action_data[0])
            elif isinstance(action_data, np.ndarray):
                if action_data.ndim == 1:
                    return 1
                elif action_data.ndim > 1:
                    return action_data.shape[-1]
            elif torch.is_tensor(action_data):
                if action_data.ndim == 1:
                    return 1
                elif action_data.ndim > 1:
                    return action_data.shape[-1]
            raise TypeError(f"Unsupported action data type: {type(action_data)}")

        self.n_actions = get_action_dim(trajs[0].actions)
        self.n_adv_actions = get_action_dim(trajs[0].adv_actions)

        # Validate consistency across all trajectories
        for i, traj in enumerate(trajs):
            if get_action_dim(traj.actions) != self.n_actions:
                raise ValueError(f"Inconsistent action dimension in trajectory {i}: {get_action_dim(traj.actions)} vs {self.n_actions}")
            if get_action_dim(traj.adv_actions) != self.n_adv_actions:
                raise ValueError(f"Inconsistent adv_action dimension in trajectory {i}: {get_action_dim(traj.adv_actions)} vs {self.n_adv_actions}")

        # Validate episode lengths using done flags
        self._validate_episode_lengths()

    def _validate_episode_lengths(self):
        """Validate that episode lengths are consistent with done flags"""
        inconsistent_episodes = []
        
        for i, traj in enumerate(self.trajs):
            obs_len = len(traj.obs)
            true_len = get_true_sequence_length(traj)
            
            if true_len != obs_len:
                inconsistent_episodes.append({
                    'traj_idx': i,
                    'obs_len': obs_len,
                    'done_based_len': true_len,
                    'dones': getattr(traj, 'dones', None)
                })
        
        if inconsistent_episodes:
            print(f"Warning: Found {len(inconsistent_episodes)} trajectories with inconsistent lengths:")
            for ep in inconsistent_episodes[:3]:  # Show first 3 examples
                print(f"  Trajectory {ep['traj_idx']}: obs_len={ep['obs_len']}, done_len={ep['done_based_len']}")
            if len(inconsistent_episodes) > 3:
                print(f"  ... and {len(inconsistent_episodes) - 3} more")

    def segment_generator_sequential(self, start_idx: int, end_idx: int):
        for traj_idx in range(start_idx, end_idx):
            traj = self.trajs[traj_idx]

            # Use done flags to determine true sequence length
            true_seq_length = get_true_sequence_length(traj)

            # Get both returns and raw rewards from the modified function
            rets_data, raw_rewards = return_labels(traj, self.gamma, self.new_rewards, self.use_minimax_returns)

            # Only use data up to the true episode length
            obs_np = np.array(traj.obs[:true_seq_length], dtype=np.float32)
            actions_np = np.array(traj.actions[:true_seq_length])
            adv_actions_np = np.array(traj.adv_actions[:true_seq_length])
            
            # Also truncate returns and rewards to true length
            rets_data = rets_data[:true_seq_length]
            raw_rewards = raw_rewards[:true_seq_length]
            
            if self.n_actions == 1 and actions_np.ndim == 1:
                actions = torch.tensor(actions_np[:, None], dtype=torch.float32)
            else:
                actions = torch.tensor(actions_np, dtype=torch.float32)

            if self.n_adv_actions == 1 and adv_actions_np.ndim == 1:
                adv_actions = torch.tensor(adv_actions_np[:, None], dtype=torch.float32)
            else:
                adv_actions = torch.tensor(adv_actions_np, dtype=torch.float32)

            obs = torch.tensor(obs_np, dtype=torch.float32)
            rets = torch.tensor(rets_data, dtype=torch.float32)
            rewards = torch.tensor(raw_rewards, dtype=torch.float32)

            # Handle player_ids if requested
            if self.include_player_ids:
                if hasattr(traj, 'player_ids') and traj.player_ids is not None:
                    player_ids_np = np.array(traj.player_ids[:true_seq_length], dtype=np.int32)
                else:
                    # Generate default alternating player IDs if missing
                    player_ids_np = np.array([i % 2 for i in range(true_seq_length)], dtype=np.int32)
                
                player_ids = torch.tensor(player_ids_np, dtype=torch.long)
                
                # Pad player_ids
                padded_player_ids = torch.full((self.horizon,), -1, dtype=torch.long)  # -1 for padding
                segment_len = min(true_seq_length, self.horizon)
                padded_player_ids[:segment_len] = player_ids[:segment_len]

            # Pad all sequences
            padded_obs = torch.zeros((self.horizon, obs.shape[1]), dtype=torch.float32)
            padded_actions = torch.zeros((self.horizon, self.n_actions), dtype=torch.float32)
            padded_adv_actions = torch.zeros((self.horizon, self.n_adv_actions), dtype=torch.float32)
            padded_rets = torch.zeros(self.horizon, dtype=torch.float32)
            padded_rewards = torch.zeros(self.horizon, dtype=torch.float32)

            segment_len = min(true_seq_length, self.horizon)

            padded_obs[:segment_len] = obs[:segment_len]
            padded_actions[:segment_len] = actions[:segment_len]
            padded_adv_actions[:segment_len] = adv_actions[:segment_len]
            padded_rets[:segment_len] = rets[:segment_len]
            padded_rewards[:segment_len] = rewards[:segment_len]

            if self.include_player_ids:
                yield (
                    padded_obs,
                    padded_actions,
                    padded_adv_actions,
                    padded_rets,
                    padded_rewards,
                    torch.tensor(true_seq_length, dtype=torch.int32),  # Return true length, not padded
                    padded_player_ids
                )
            else:
                yield (
                    padded_obs,
                    padded_actions,
                    padded_adv_actions,
                    padded_rets,
                    padded_rewards,
                    torch.tensor(true_seq_length, dtype=torch.int32)  # Return true length, not padded
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
