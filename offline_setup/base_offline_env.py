from offline_setup.trajectory_sampler import TrajectorySampler
from os import path
import json
import numpy as np
import os
from pathlib import Path
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder

# Define the Trajectory namedtuple
Trajectory = namedtuple("Trajectory", ["obs", "actions", "rewards", "adv_actions", "adv_rewards", "infos"])

class BaseOfflineEnv:

    def __init__(self, p, env_cls, data_policy, horizon, n_interactions, test=False, state_dim=12):
        self.env_cls = env_cls
        self.data_policy = data_policy
        self.horizon = horizon
        self.n_interactions = n_interactions
        self.p = p
        self.state_dim = state_dim  # Added for state dimension
        if test:
            return

        # Construct the full path for existence check
        json_path = Path(__file__).parent.parent / 'offline_game_data' / Path(self.p).name if self.p is not None else None
        if json_path is not None and path.exists(json_path):
            print('Dataset file found. Loading existing trajectories.')
            try:
                self.trajs = self._load_trajectories()
            except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
                print(f'Error loading dataset: {e}. Generating new trajectories.')
                self.trajs = []
                self.generate_and_save()
        else:
            print('Dataset file not found. Generating trajectories.')
            self.generate_and_save()

    def _load_trajectories(self):
        """Load trajectories from JSON file using get_offline_data."""
        raw_data = get_offline_data(Path(self.p).name)  # Use get_offline_data with file name
        return convert_dataset(raw_data, state_dim=self.state_dim)

    def generate_and_save(self):
        """Generate new trajectories and save to JSON."""
        self.trajs = self.collect_trajectories()

        if self.p is not None:
            os.makedirs(path.dirname(self.p), exist_ok=True)
            with open(self.p, 'w') as file:
                json_data = self._convert_trajs_to_json(self.trajs)
                json.dump(json_data, file, indent=4)
                print('Saved trajectories to dataset file.')

    def collect_trajectories(self):
        """Collect trajectories using TrajectorySampler."""
        data_policy = self.data_policy()
        sampler = TrajectorySampler(env_cls=self.env_cls,
                                    policy=data_policy,
                                    horizon=self.horizon)
        trajs = sampler.collect_trajectories(self.n_interactions)
        return trajs

    def _convert_trajs_to_json(self, trajs):
        """Convert Trajectory namedtuples to JSON format matching the provided structure."""
        json_data = []
        for i, traj in enumerate(trajs):
            # Extract unique actions for encoding
            str_actions = [str(np.argmax(act) if not np.all(act == -10) else -10) for act in traj.actions]
            unique_actions = sorted(set(act for act in str_actions if act != '-10'))
            
            episode = {
                'episode_id': i,
                'str_states': [str(obs) for obs in traj.obs],
                'num_states': [str(np.argmax(obs) if np.sum(obs) > 0 else -1) for obs in traj.obs],
                'player_ids': [info.get('player_id', 0) for info in traj.infos],
                'str_actions': str_actions,
                'num_actions': [int(float(act)) if act != '-10' else -10 for act in str_actions],
                'rewards': [traj.rewards[-1], traj.adv_rewards[-1]],  # Last rewards for protagonist and adversary
                'obs': traj.obs.tolist(),
                'actions': traj.actions.tolist(),
                'adv_actions': traj.adv_actions.tolist(),
                'adv_rewards': traj.adv_rewards.tolist(),
                'infos': traj.infos
            }
            json_data.append(episode)
        return json_data

# Integrated dataset processing functions
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
        one_hot_vector = np.zeros(state_dim)
        one_hot_vector[min(index, state_dim - 1)] = 1
        state_mapping[state] = one_hot_vector
    return state_mapping[state]

def convert_dataset(dataset, state_dim=12):
    trajectories = []
    state_mapping = {}
    unique_states = set()
    for episode in dataset:
        unique_states.update(episode['num_states'])
    print(f"Unique states: {unique_states}, Count: {len(unique_states)}")
    state_dim = len(unique_states)
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

def default_path(name, is_data=True):
    # Get the path of the current file
    file_path = Path(__file__).parent.parent
    if is_data:
        # Go to offline_game_data directory
        full_path = file_path / 'offline_game_data'
    else:
        full_path = file_path
    # Append the name of the dataset
    return str(full_path / name)