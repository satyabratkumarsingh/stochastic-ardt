
from offline_setup.trajectory_sampler import TrajectorySampler
from os import path
import json
import numpy as np
import os
from pathlib import Path
from collections import namedtuple
from sklearn.preprocessing import LabelEncoder
from tqdm import tqdm
from data_class.trajectory import Trajectory

class BaseOfflineEnv:
    def __init__(self, p, env_cls, data_policy, horizon, n_interactions, test=False, state_dim=12):
        self.env_cls = env_cls
        self.data_policy = data_policy
        self.horizon = horizon
        self.n_interactions = n_interactions
        self.p = p
        self.state_dim = state_dim
        if test:
            return

        json_path = Path(__file__).parent.parent / 'offline_game_data' / Path(self.p).name if self.p is not None else None
        if json_path is not None and path.exists(json_path):
            print('Dataset file found. Loading existing trajectories.')
            try:
                self.trajs = self._load_trajectories()
            except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
                print(f'Error loading dataset: {e}. Generating new trajectories.')
                self.trajs = []
                # self.generate_and_save() # Commented out to avoid dummy generation conflicts
        else:
            print('Dataset file not found. Generating trajectories.')
            self.trajs = []
            # self.generate_and_save() # Commented out

    def _load_trajectories(self):
        raw_data = get_offline_data(Path(self.p).name)
        print('Converting dataset to trajectories...')
        return convert_dataset(raw_data, state_dim=self.state_dim)

    def generate_and_save(self):
        json_path = (Path(__file__).parent.parent / 'offline_game_data' / 'kuhn_poker_trajectories.json')
        os.makedirs(json_path.parent, exist_ok=True)
        with open(json_path, 'w') as file:
            json_data = self._convert_trajs_to_json(self.trajs)
            json.dump(json_data, file, indent=4)
            print(f'Saved trajectories to {json_path}')

    def _convert_trajs_to_json(self, trajs):
        json_data = []
        for traj in tqdm(trajs, desc="Converting trajectories to JSON"):
            # Ensure actions are handled correctly if they are one-hot encoded now
            # You might need to adjust this if `traj.actions` is no longer a simple scalar array
            # and contains -10 for padding
            str_actions = [str(np.argmax(act) if np.sum(np.abs(act)) > 0 else -10) for act in traj.actions]

            episode = {
                'episode_id': traj.episode_id,
                'str_states': [str(obs) for obs in traj.obs], # Convert back from one-hot if needed, or keep as is
                'num_states': [str(np.argmax(obs) if np.sum(obs) > 0 else -1) for obs in traj.obs],
                'player_ids': [info.get('player_id', 0) for info in traj.infos],
                'str_actions': str_actions,
                'num_actions': [int(float(act)) if act != '-10' else -10 for act in str_actions],
                'rewards': [traj.rewards[-1], traj.adv_rewards[-1]], # Only final rewards
                'obs': traj.obs.tolist(),
                'actions': traj.actions.tolist(),
                'adv_actions': traj.adv_actions.tolist(),
                'adv_rewards': traj.adv_rewards.tolist(),
                'infos': traj.infos,
                'dones': traj.dones.tolist() # Ensure dones are saved correctly
            }
            json_data.append(episode)
        return json_data

def load_dataset(file_path):
    with open(file_path, 'r') as f:
        return json.load(f)

# The one_hot_encode for states seems fine as it handles mapping.
# It uses 'state_mapping' and pads to 'state_dim' internally, which is okay for the state representation.
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
    # Removed global max_timesteps calculation, as we will use per-episode length
    # max_timesteps = max(len(episode["num_states"]) for episode in dataset)

    all_action_categories = sorted(set(action for episode in dataset for action in episode["str_actions"] if action is not None and action != 'None' and action != ''))
    num_classes = len(all_action_categories)
    category_to_index = {cat: idx for idx, cat in enumerate(all_action_categories)}
    pad_vec = np.zeros(num_classes, dtype=np.float32)

    for episode in tqdm(dataset, desc="Processing episodes"):
        # --- Input validation ---
        if len(episode['str_actions']) != len(episode['player_ids']):
            raise ValueError(f"Mismatch in lengths: str_actions ({len(episode['str_actions'])}) and player_ids ({len(episode['player_ids'])}) in episode {episode['episode_id']}")
        if len(episode['num_states']) != len(episode['str_actions']):
            raise ValueError(f"Mismatch in lengths: num_states ({len(episode['num_states'])}) and str_actions ({len(episode['str_actions'])}) in episode {episode['episode_id']}")
        if len(episode['rewards']) != 2:
            raise ValueError(f"Expected 2 rewards in episode {episode['episode_id']}, got {len(episode['rewards'])}")
        if not episode['num_states']:
            print(f"Warning: Empty num_states in episode {episode['episode_id']}. Skipping.")
            continue # Skip empty episodes

        episode_id = episode['episode_id']
        # This is the TRUE length of the current episode:
        num_timesteps = len(episode['num_states'])

        # 1. One-hot encode observations (no padding here to global max_timesteps)
        obs = np.array([one_hot_encode(state, state_mapping, state_dim=state_dim) for state in episode['num_states']])
        # No `if obs.shape[0] < max_timesteps:` padding here. `obs` will have length `num_timesteps`.

        # 2. Build per-timestep action sequences (no padding here to global max_timesteps)
        embeded_pr_actions = []
        embedded_adv_actions = []
        # Loop for the actual number of steps in this episode
        for pid, act_str in zip(episode['player_ids'], episode['str_actions']):
            if act_str in category_to_index and act_str not in ['None', '']:
                vec = np.eye(num_classes)[category_to_index[act_str]]
            else:
                vec = pad_vec # e.g., [0,0] for 2 action classes
            if pid == 0: # Protagonist's turn
                embeded_pr_actions.append(vec)
                embedded_adv_actions.append(pad_vec) # Opponent's action is padded for protagonist's turn
            else: # Adversary's turn
                embeded_pr_actions.append(pad_vec) # Protagonist's action is padded for adversary's turn
                embedded_adv_actions.append(vec)

        embeded_pr_actions = np.array(embeded_pr_actions)
        embedded_adv_actions = np.array(embedded_adv_actions)
        # These arrays will now have length `num_timesteps`.

        # 3. Correct reward assignment (size based on num_timesteps)
        protagonist_reward = np.zeros(num_timesteps)
        adversary_reward = np.zeros(num_timesteps)
        # Assign final rewards at the true last step of the episode
        protagonist_reward[num_timesteps - 1] = episode['rewards'][0]
        adversary_reward[num_timesteps - 1] = episode['rewards'][1]

        # 4. Info per step (already correctly based on episode length)
        # Ensure infos doesn't exceed num_timesteps if it's derived elsewhere
        infos = [{'player_id': pid, 'adv': int(action == 1)} for pid, action in zip(episode['player_ids'], episode['num_actions'])]
        # This `infos` list correctly has `num_timesteps` elements.

        # 5. Done flag (size based on num_timesteps)
        dones = np.zeros(num_timesteps, dtype=bool)
        dones[num_timesteps - 1] = True # Mark true at the actual last step of THIS episode

        # Save trajectory
        trajectory = Trajectory(
            episode_id=episode_id,
            obs=obs,
            actions=embeded_pr_actions,
            rewards=protagonist_reward,
            adv_actions=embedded_adv_actions,
            adv_rewards=adversary_reward,
            infos=infos,
            dones=dones
        )
        trajectories.append(trajectory)

    return trajectories

def get_offline_data(file_name):
    try:
        json_path = Path(__file__).parent.parent / 'offline_game_data' / file_name
        with open(json_path, "r") as file:
            data = json.load(file)
            print(f"==============Offline Data file found with name {file_name} ==============")
            return data
    except FileNotFoundError:
        print("==============Offline Data file not found ================")
        raise

def default_path(name, is_data=True):
    file_path = Path(__file__).parent.parent
    if is_data:
        full_path = file_path / 'offline_game_data'
    else:
        full_path = file_path
    return str(full_path / name)
