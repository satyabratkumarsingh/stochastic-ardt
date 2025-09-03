import numpy as np
import os
import json
from pathlib import Path
from tqdm import tqdm
from data_class.trajectory import Trajectory

class BaseOfflineEnv:
    """
    Simplified offline environment for Kuhn Poker
    """
    def __init__(self, p, env_cls=None, data_policy=None, horizon=100, n_interactions=1000, 
                 test=False, game_name='kuhn_poker'):
        self.p = p
        self.game_name = game_name
        self.trajs = []
        if test:
            return

        json_path = Path(__file__).parent.parent / 'offline_game_data' / Path(self.p).name if self.p is not None else None
        
        if json_path is not None and json_path.exists():
            print('Dataset file found. Loading existing trajectories.')
            try:
                raw_data = self._get_offline_data(Path(self.p).name)
                # Build simplified state mapping
                self.state_mapping, self.state_dim = self._build_kuhn_state_mapping(raw_data)
                self.trajs = self._convert_dataset(raw_data)
            except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
                print(f'Error loading dataset: {e}')
                self.trajs = []
        else:
            print('Dataset file not found.')
            self.trajs = []

    def _get_offline_data(self, file_name):
        """Load JSON data from file"""
        try:
            json_path = Path(__file__).parent.parent / 'offline_game_data' / file_name
            with open(json_path, "r") as file:
                data = json.load(file)
                print(f"Loaded offline data from {file_name}")
                return data
        except FileNotFoundError:
            print(f"Offline data file {file_name} not found")
            raise
        
    @staticmethod
    def _build_kuhn_state_mapping(dataset):
        """Build state mapping using one-hot encoding"""
        
        canonical_states = ["0", "0p", "0b", "0pb",
                        "1", "1p", "1b", "1pb", 
                        "2", "2p", "2b", "2pb"]
        
        # Get actual states from dataset
        actual_states = sorted(list(set(state for episode in dataset for state in episode['num_states'])))
        
        # Use canonical order for states that exist
        ordered_states = []
        for state in canonical_states:
            if state in actual_states:
                ordered_states.append(state)
        
        # Add any unexpected states
        for state in actual_states:
            if state not in ordered_states:
                ordered_states.append(state)
                print(f"Warning: Found unexpected state '{state}' - adding to end of mapping")
        
        state_dim = len(ordered_states)
        # One-hot encoding for states
        state_mapping = {state: np.eye(state_dim, dtype=np.float32)[i] for i, state in enumerate(ordered_states)}
        
        print(f"\nKuhn Poker State Mapping (Total states: {state_dim}):")
        for i, state in enumerate(ordered_states):
            card_name = {"0": "Jack", "1": "Queen", "2": "King"}[state[0]]
            history = state[1:] if len(state) > 1 else "initial"
            print(f"  Index {i:2d}: '{state}' -> {card_name}, {history}")
        
        return state_mapping, state_dim

    def _convert_dataset(self, dataset):
        """Convert raw dataset to 3D one-hot encoded trajectory format"""
        print(f"Game type: {self.game_name}")
        print(f"State dimension: {self.state_dim}")
        
        trajectories = []
        # Updated to 3D action space: [Pass, Bet, NoAction]
        action_mapping = {"Pass": 0, "Bet": 1, "NoAction": 2}
        num_action_classes = 3  # Pass, Bet, NoAction
        action_pad_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)  # NoAction vector
        
        print(f"Action mapping: {action_mapping}")
        print("Action encoding: [Pass, Bet, NoAction]")

        for episode in tqdm(dataset, desc="Processing Kuhn episodes"):
            episode_id = episode['episode_id']
            num_timesteps = len(episode['num_states'])
            
            # Convert states to one-hot vectors
            try:
                obs = np.array([self.state_mapping[state] for state in episode['num_states']])
            except KeyError as e:
                raise ValueError(f"Unexpected state {e.args[0]} in episode {episode_id}")
            
            # Process actions for both players with 3D one-hot encoding
            protagonist_actions, adversary_actions = self._process_actions_onehot(
                episode, action_mapping, action_pad_vec, num_timesteps)
            
            # Process rewards and episode data
            rewards, adv_rewards, infos, dones = self._process_episode_data(episode, num_timesteps)
            
            trajectory = Trajectory(
                episode_id=episode_id,
                obs=obs,                        # One-hot state vectors (12D)
                actions=protagonist_actions,    # 3D one-hot protagonist actions
                adv_actions=adversary_actions,  # 3D one-hot adversary actions
                rewards=rewards,
                adv_rewards=adv_rewards,
                infos=infos,
                dones=dones
            )
            trajectories.append(trajectory)

        return trajectories

    def _process_actions_onehot(self, episode, action_to_index, action_pad_vec, num_timesteps):
        """Process actions using 3D one-hot encoding: [Pass, Bet, NoAction]"""
        protagonist_actions = []
        adversary_actions = []
        
        # 3D action space: [Pass, Bet, NoAction]
        no_action_vec = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        
        for i, (pid, action_str) in enumerate(zip(episode['player_ids'], episode['str_actions'])):
            # Create 3D one-hot action vector
            if action_str == "Pass":
                action_vec = np.array([1.0, 0.0, 0.0], dtype=np.float32)
            elif action_str == "Bet":
                action_vec = np.array([0.0, 1.0, 0.0], dtype=np.float32)
            else:
                action_vec = no_action_vec.copy()  # Invalid/unknown action
            
            # Assign actions based on player ID
            if pid == 0:  # Protagonist's turn
                protagonist_actions.append(action_vec)
                adversary_actions.append(no_action_vec.copy())
            else:  # Adversary's turn
                protagonist_actions.append(no_action_vec.copy())
                adversary_actions.append(action_vec)
        
        return np.array(protagonist_actions), np.array(adversary_actions)

    def _process_episode_data(self, episode, num_timesteps):
        """Process rewards, info, and done flags"""
        protagonist_reward = np.zeros(num_timesteps, dtype=np.float32)
        adversary_reward = np.zeros(num_timesteps, dtype=np.float32)
        
        # Assign final rewards to the last timestep
        if 'rewards' in episode and len(episode['rewards']) >= 2:
            # Player 0 is always protagonist in our simplified format
            protagonist_reward[num_timesteps - 1] = episode['rewards'][0]
            adversary_reward[num_timesteps - 1] = episode['rewards'][1]

        infos = [{'player_id': pid, 'action': action} 
                for pid, action in zip(episode['player_ids'], episode['num_actions'])]
        
        dones = np.zeros(num_timesteps, dtype=bool)
        dones[num_timesteps - 1] = True
        
        return protagonist_reward, adversary_reward, infos, dones

    def save_trajectories(self, filepath=None):
        """Save one-hot encoded trajectories to file"""
        if filepath is None:
            filepath = Path(__file__).parent.parent / 'offline_game_data' / f'{self.game_name}_onehot_trajectories.json'
        
        os.makedirs(filepath.parent, exist_ok=True)
        
        json_data = []
        for traj in tqdm(self.trajs, desc="Converting trajectories to JSON"):
            episode = {
                'episode_id': traj.episode_id,
                'obs': traj.obs.tolist(),           # List of one-hot state vectors
                'actions': traj.actions.tolist(),   # List of one-hot action vectors
                'adv_actions': traj.adv_actions.tolist(),  # List of one-hot adversary action vectors
                'rewards': traj.rewards.tolist(),
                'adv_rewards': traj.adv_rewards.tolist(),
                'infos': traj.infos,
                'dones': traj.dones.tolist()
            }
            json_data.append(episode)
        
        with open(filepath, 'w') as file:
            json.dump(json_data, file, indent=4)
        print(f'Saved one-hot encoded trajectories to {filepath}')

    def get_state_action_pairs(self):
        """Extract all (state, action, player) tuples for analysis"""
        pairs = []
        for traj in self.trajs:
            for i in range(len(traj.obs)):
                pairs.append({
                    'state_idx': traj.obs[i],
                    'action_idx': traj.actions[i],
                    'player': traj.players[i],
                    'reward': traj.rewards[i],
                    'episode_id': traj.episode_id
                })
        return pairs