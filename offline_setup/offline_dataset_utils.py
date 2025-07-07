import json
import os
from pathlib import Path
import numpy as np
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder

Trajectory = namedtuple("Trajectory", ["obs", "actions", "rewards", "adv_actions", "adv_rewards", "infos"])

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

def pad_and_one_hot_encode(sequence, target_length, pad_value=-10, all_categories=None):
    if all_categories is None or not all_categories:
        raise ValueError("all_categories must be provided and non-empty")
    
    num_classes = len(all_categories)
    padded_sequence = sequence + [None] * (target_length - len(sequence))
    
    category_to_index = {cat: idx for idx, cat in enumerate(all_categories)}
    indexed_sequence = [category_to_index.get(act, -1) for act in padded_sequence]
    
    one_hot_encoded = np.array([
        np.eye(num_classes)[idx] if idx >= 0 else np.full(num_classes, pad_value)
        for idx in indexed_sequence
    ])
    return one_hot_encoded

def one_hot_encode(state, state_mapping, state_dim):
    if state not in state_mapping:
        index = len(state_mapping)
        print(f"INDEX {index}")
        one_hot_vector = np.zeros(state_dim)
        one_hot_vector[min(index, state_dim - 1)] = 1
        state_mapping[state] = one_hot_vector
    return state_mapping[state]

def convert_dataset(dataset, state_dim=10):
    trajectories = []
    state_mapping = {}

    all_action_categories = sorted(set(action for episode in dataset for action in episode["str_actions"]))
    max_str_actions_length = max(len(episode["str_actions"]) for episode in dataset)
    max_timesteps = max(len(episode["num_states"]) for episode in dataset)

    for episode in dataset:
        if len(episode['str_actions']) != len(episode['player_ids']):
            raise ValueError(f"Mismatch in lengths: str_actions ({len(episode['str_actions'])}) and player_ids ({len(episode['player_ids'])}) in episode {episode['episode_id']}")
        if len(episode['num_states']) != len(episode['str_actions']):
            raise ValueError(f"Mismatch in lengths: num_states ({len(episode['num_states'])}) and str_actions ({len(episode['str_actions'])}) in episode {episode['episode_id']}")
        if len(episode['rewards']) != 2:
            raise ValueError(f"Expected 2 rewards in episode {episode['episode_id']}, got {len(episode['rewards'])}")
        if not episode['num_states']:
            raise ValueError(f"Empty num_states in episode {episode['episode_id']}")

        obs = np.array([one_hot_encode(state, state_mapping, state_dim=state_dim) for state in episode['num_states']])
        if obs.shape[0] < max_timesteps:
            pad_width = ((0, max_timesteps - obs.shape[0]), (0, 0))
            obs = np.pad(obs, pad_width, mode='constant', constant_values=0)
        
        protagonist_actions = [episode['str_actions'][i] for i in range(len(episode['str_actions'])) if episode['player_ids'][i] == 1]
        adversary_actions = [episode['str_actions'][i] for i in range(len(episode['str_actions'])) if episode['player_ids'][i] == 0]

        embeded_pr_actions = pad_and_one_hot_encode(protagonist_actions,
                                                    max_str_actions_length,
                                                    pad_value=-10,
                                                    all_categories=all_action_categories)
        embedded_adv_actions = pad_and_one_hot_encode(adversary_actions,
                                                     max_str_actions_length,
                                                     pad_value=-10,
                                                     all_categories=all_action_categories)

        num_timesteps = len(episode['num_states'])
        protagonist_reward = np.zeros(max_timesteps)
        adversary_reward = np.zeros(max_timesteps)
        protagonist_reward[num_timesteps - 1] = episode['rewards'][1]
        adversary_reward[num_timesteps - 1] = episode['rewards'][0]

        infos = [{'player_id': pid, 'adv': int(action == 1)} for pid, action in zip(episode['player_ids'], episode['num_actions'])]

        trajectory = Trajectory(
            obs=obs,
            actions=embeded_pr_actions,
            adv_actions=embedded_adv_actions,
            rewards=protagonist_reward,
            adv_rewards=adversary_reward,
            infos=infos
        )
        trajectories.append(trajectory)
    
    return trajectories


def get_offline_data(file_name):
    try:
        json_path = Path(__file__).parent.parent / 'offline_game_data' / file_name
        with open(json_path, "r") as file:
            data = json.load(file)
            print("==============Offline Data file found with name {} ==============".format(file_name))
            return data
    except FileNotFoundError:
        print("==============Offline Data file not found ================")
        raise

def get_trajectory_for_offline(file_name, state_dim=10):
    raw_data = get_offline_data(file_name=file_name)
    trajectories = convert_dataset(raw_data, state_dim=state_dim)
    return trajectories