import re
from pathlib import Path
from tqdm import tqdm
from data_class.trajectory import Trajectory
import json
import numpy as np
import os

class BaseOfflineEnv:
    """
    Clean offline environment that automatically detects game type and sets appropriate state dimensions
    """
    
    def __init__(self, p, env_cls=None, data_policy=None, horizon=100, n_interactions=1000, 
                 test=False, game_name=None):
        self.env_cls = env_cls
        self.data_policy = data_policy
        self.horizon = horizon
        self.n_interactions = n_interactions
        self.p = p
        self.game_name = game_name
        
        if test:
            return

        # Load and process data
        json_path = Path(__file__).parent.parent / 'offline_game_data' / Path(self.p).name if self.p is not None else None
        if json_path is not None and json_path.exists():
            print('Dataset file found. Loading existing trajectories.')
            try:
                self.trajs = self._load_trajectories()
            except (json.JSONDecodeError, FileNotFoundError, ValueError) as e:
                print(f'Error loading dataset: {e}')
                self.trajs = []
        else:
            print('Dataset file not found.')
            self.trajs = []

    def _load_trajectories(self):
        """Load raw data and convert to trajectories"""
        raw_data = self._get_offline_data(Path(self.p).name)
        print('Converting dataset to trajectories...')
        return self._convert_dataset(raw_data)
    
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

    def _detect_game_type(self, dataset):
        """Auto-detect game type from data structure"""
        if not dataset or len(dataset) == 0:
            return 'kuhn_poker'
        
        # Check first episode's state format
        first_episode = dataset[0]
        
        if 'num_states' in first_episode and len(first_episode['num_states']) > 0:
            first_state = str(first_episode['num_states'][0])
            
            # Check for Leduc format patterns
            if ('[Observer:' in first_state or 
                '[Private:' in first_state or
                'Round:' in first_state or
                'Pot:' in first_state):
                return 'leduc_poker'
        
        # Default to Kuhn poker for simple states
        return 'kuhn_poker'

    def _get_state_dim(self, game_name):
        """Get appropriate state dimension for game type"""
        if game_name == 'kuhn_poker':
            return 12
        elif game_name == 'leduc_poker':
            # 2 (observer) + 6 (private) + 2 (round) + 2 (player) + 1 (pot) + 
            # 2 (money) + 7 (public) + 10 (round1) + 10 (round2) = 42
            return 42
        else:
            raise ValueError(f"Unknown game type: {game_name}")

    def _convert_dataset(self, dataset):
        """Convert raw dataset to trajectory format"""
        # Auto-detect game type if not specified
        if self.game_name is None:
            self.game_name = self._detect_game_type(dataset)
        
        # Get appropriate state dimension
        state_dim = self._get_state_dim(self.game_name)
        
        print(f"Game type: {self.game_name}")
        print(f"State dimension: {state_dim}")
        
        # Convert based on game type
        if self.game_name == 'kuhn_poker':
            return self._convert_kuhn_dataset(dataset, state_dim)
        elif self.game_name == 'leduc_poker':
            return self._convert_leduc_dataset(dataset, state_dim)
        else:
            raise ValueError(f"Unsupported game type: {self.game_name}")

    def _convert_kuhn_dataset(self, dataset, state_dim):
        """Convert Kuhn poker dataset with simple state encoding"""
        trajectories = []
        state_mapping = {}
        
        # Get action categories
        all_actions = sorted(set(action for episode in dataset 
                               for action in episode["str_actions"] 
                               if action and action != 'None'))
        num_classes = len(all_actions)
        action_to_index = {action: idx for idx, action in enumerate(all_actions)}
        
        # Manually set the Kuhn poker action map for robustness
        kuhn_poker_actions = ['Pass', 'Bet']
        if all(action in all_actions for action in kuhn_poker_actions):
            action_to_index = {action: i for i, action in enumerate(kuhn_poker_actions)}
            num_classes = len(kuhn_poker_actions)
        
        pad_vec = np.zeros(num_classes, dtype=np.float32)
        
        print(f"Kuhn poker actions: {all_actions}")
        print(f"Action mapping: {action_to_index}")

        for episode in tqdm(dataset, desc="Processing Kuhn episodes"):
            episode_id = episode['episode_id']
            num_timesteps = len(episode['num_states'])
            
            # Simple one-hot encoding for Kuhn states
            obs = np.array([self._one_hot_encode_simple(state, state_mapping, state_dim) 
                           for state in episode['num_states']])
            
            # Process actions
            protagonist_actions, adversary_actions = self._process_actions(
                episode, action_to_index, pad_vec, num_classes)
            
            # Process rewards and info
            rewards, adv_rewards, infos, dones = self._process_episode_data(episode, num_timesteps)
            
            trajectory = Trajectory(
                episode_id=episode_id,
                obs=obs,
                actions=protagonist_actions,
                rewards=rewards,
                adv_actions=adversary_actions,
                adv_rewards=adv_rewards,
                infos=infos,
                dones=dones
            )
            trajectories.append(trajectory)

        return trajectories

    def _convert_leduc_dataset(self, dataset, state_dim):
        """Convert Leduc poker dataset with structured state parsing"""
        trajectories = []
        
        # Get action categories
        all_actions = sorted(set(action for episode in dataset 
                               for action in episode["str_actions"] 
                               if action and action != 'None'))
        num_classes = len(all_actions)
        action_to_index = {action: idx for idx, action in enumerate(all_actions)}
        pad_vec = np.zeros(num_classes, dtype=np.float32)
        
        print(f"Leduc poker actions: {all_actions}")

        for episode in tqdm(dataset, desc="Processing Leduc episodes"):
            episode_id = episode['episode_id']
            num_timesteps = len(episode['num_states'])
            
            # Parse Leduc states
            obs_list = []
            for state_str in episode['num_states']:
                features = self._parse_leduc_state(state_str)
                state_vec = self._create_leduc_state_vector(features, state_dim)
                obs_list.append(state_vec)
            obs = np.array(obs_list)
            
            # Process actions
            protagonist_actions, adversary_actions = self._process_actions(
                episode, action_to_index, pad_vec, num_classes)
            
            # Process rewards and info
            rewards, adv_rewards, infos, dones = self._process_episode_data(episode, num_timesteps)
            
            trajectory = Trajectory(
                episode_id=episode_id,
                obs=obs,
                actions=protagonist_actions,
                rewards=rewards,
                adv_actions=adversary_actions,
                adv_rewards=adv_rewards,
                infos=infos,
                dones=dones
            )
            trajectories.append(trajectory)

        return trajectories

    def _one_hot_encode_simple(self, state, state_mapping, state_dim):
        """Simple one-hot encoding for Kuhn poker states"""
        if state not in state_mapping:
            index = len(state_mapping)
            one_hot_vector = np.zeros(state_dim)
            one_hot_vector[min(index, state_dim - 1)] = 1
            state_mapping[state] = one_hot_vector
        return state_mapping[state]

    def _parse_leduc_state(self, state_str):
        """Parse Leduc poker state string into structured features"""
        features = {}
        
        # Extract observer
        observer_match = re.search(r'\[Observer: (\d+)\]', state_str)
        features['observer'] = int(observer_match.group(1)) if observer_match else 0
        
        # Extract private card
        private_match = re.search(r'\[Private: (\d+)\]', state_str)
        features['private_card'] = int(private_match.group(1)) if private_match else -1
        
        # Extract round
        round_match = re.search(r'\[Round (\d+)\]', state_str)
        features['round'] = int(round_match.group(1)) if round_match else 1
        
        # Extract current player
        player_match = re.search(r'\[Player: (-?\d+)\]', state_str)
        features['current_player'] = int(player_match.group(1)) if player_match else 0
        
        # Extract pot
        pot_match = re.search(r'\[Pot: (\d+)\]', state_str)
        features['pot'] = int(pot_match.group(1)) if pot_match else 0
        
        # Extract money
        money_match = re.search(r'\[Money: ([\d\s]+)\]', state_str)
        if money_match:
            money_values = [int(x) for x in money_match.group(1).split()]
            features['money_p1'] = money_values[0] if len(money_values) > 0 else 0
            features['money_p2'] = money_values[1] if len(money_values) > 1 else 0
        else:
            features['money_p1'] = features['money_p2'] = 0
        
        # Extract public card
        public_match = re.search(r'\[Public: (-?\d+)\]', state_str)
        features['public_card'] = int(public_match.group(1)) if public_match else -1
        
        # Extract round 1 actions
        round1_match = re.search(r'\[Round1: ([^\]]*)\]', state_str)
        round1_actions = []
        if round1_match and round1_match.group(1).strip():
            round1_actions = [int(x) for x in round1_match.group(1).split()]
        features['round1_actions'] = round1_actions
        
        # Extract round 2 actions
        round2_match = re.search(r'\[Round2: ([^\]]*)\]', state_str)
        round2_actions = []
        if round2_match and round2_match.group(1).strip():
            round2_actions = [int(x) for x in round2_match.group(1).split()]
        features['round2_actions'] = round2_actions
        
        return features

    def _create_leduc_state_vector(self, features, state_dim):
        """Create fixed-length state vector for Leduc poker"""
        vector = []
        
        # Observer (2 dims)
        observer_vec = np.zeros(2)
        if 0 <= features['observer'] < 2:
            observer_vec[features['observer']] = 1
        vector.extend(observer_vec)
        
        # Private card (6 dims for cards 0-5)
        private_vec = np.zeros(6)
        if 0 <= features['private_card'] < 6:
            private_vec[features['private_card']] = 1
        vector.extend(private_vec)
        
        # Round (2 dims)
        round_vec = np.zeros(2)
        if features['round'] in [1, 2]:
            round_vec[features['round'] - 1] = 1
        vector.extend(round_vec)
        
        # Current player (2 dims)
        player_vec = np.zeros(2)
        if 0 <= features['current_player'] < 2:
            player_vec[features['current_player']] = 1
        vector.extend(player_vec)
        
        # Pot (normalized)
        vector.append(min(features['pot'] / 200.0, 1.0))
        
        # Money (normalized)
        vector.append(features['money_p1'] / 100.0)
        vector.append(features['money_p2'] / 100.0)
        
        # Public card (7 dims: 0-5 + no card)
        public_vec = np.zeros(7)
        if features['public_card'] == -1:
            public_vec[6] = 1  # No public card
        elif 0 <= features['public_card'] < 6:
            public_vec[features['public_card']] = 1
        vector.extend(public_vec)
        
        # Round 1 actions (10 dims)
        round1_vec = np.zeros(10)
        for i, action in enumerate(features['round1_actions'][:10]):
            round1_vec[i] = action
        vector.extend(round1_vec)
        
        # Round 2 actions (10 dims)
        round2_vec = np.zeros(10)
        for i, action in enumerate(features['round2_actions'][:10]):
            round2_vec[i] = action
        vector.extend(round2_vec)
        
        # Ensure correct length
        vector = np.array(vector)
        if len(vector) < state_dim:
            padded_vector = np.zeros(state_dim)
            padded_vector[:len(vector)] = vector
            return padded_vector
        return vector[:state_dim]

    def _process_actions(self, episode, action_to_index, pad_vec, num_classes):
        """Process actions for both players"""
        protagonist_actions = []
        adversary_actions = []
        
        for pid, action_str in zip(episode['player_ids'], episode['str_actions']):
            action_vec = np.zeros(num_classes, dtype=np.float32)
            if action_str and action_str in action_to_index:
                action_vec[action_to_index[action_str]] = 1.0
            
            if pid == 0:  # Protagonist
                protagonist_actions.append(action_vec)
                adversary_actions.append(pad_vec)
            else:  # Adversary
                protagonist_actions.append(pad_vec)
                adversary_actions.append(action_vec)
        
        return np.array(protagonist_actions), np.array(adversary_actions)

    def _process_episode_data(self, episode, num_timesteps):
        """Process rewards, info, and done flags"""
        # Rewards
        protagonist_reward = np.zeros(num_timesteps, dtype=np.float32)
        adversary_reward = np.zeros(num_timesteps, dtype=np.float32)
        
        # Assign final rewards to the last timestep
        if 'rewards' in episode and len(episode['rewards']) >= 2:
            protagonist_reward[num_timesteps - 1] = episode['rewards'][0]
            adversary_reward[num_timesteps - 1] = episode['rewards'][1]
        
        # Info
        infos = [{'player_id': pid, 'action': action} 
                for pid, action in zip(episode['player_ids'], episode['num_actions'])]
        
        # Done flags
        dones = np.zeros(num_timesteps, dtype=bool)
        dones[num_timesteps - 1] = True
        
        return protagonist_reward, adversary_reward, infos, dones

    def save_trajectories(self, filepath=None):
        """Save trajectories to file"""
        if filepath is None:
            filepath = Path(__file__).parent.parent / 'offline_game_data' / f'{self.game_name}_trajectories.json'
        
        os.makedirs(filepath.parent, exist_ok=True)
        
        json_data = []
        for traj in tqdm(self.trajs, desc="Converting trajectories to JSON"):
            # Use raw data to maintain integrity
            episode = {
                'episode_id': traj.episode_id,
                'obs': traj.obs.tolist(),
                'actions': traj.actions.tolist(),
                'rewards': traj.rewards.tolist(),
                'adv_actions': traj.adv_actions.tolist(),
                'adv_rewards': traj.adv_rewards.tolist(),
                'infos': traj.infos,
                'dones': traj.dones.tolist()
            }
            json_data.append(episode)
        
        with open(filepath, 'w') as file:
            json.dump(json_data, file, indent=4)
        print(f'Saved trajectories to {filepath}')