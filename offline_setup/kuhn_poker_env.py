import gym
import numpy as np
import torch
import torch.nn.functional as F
from gym import spaces

# Assuming BaseOfflineEnv is defined elsewhere, or you can remove its inheritance if not needed.
# For a standard Gym environment, you typically don't inherit from BaseOfflineEnv directly.
# Let's assume you've imported it correctly or it's implicitly handled.


class KuhnPokerEnv(gym.Env):
    def __init__(self):
        super().__init__() # Call parent constructor
        self.env_name = "kuhn_poker"
        # Define action spaces (discrete: 0 = pass, 1 = bet)
        self.action_space = spaces.Discrete(2) # Player 0 actions
        self.adv_action_space = spaces.Discrete(2) # Player 1 (adversary) actions
        
        # State: One-hot encoding for cards (J, Q, K) + game history.
        # Max history: P0-bet, P1-bet (2 actions) or P0-pass, P1-bet, P0-bet (3 actions)
        # States could be: (Player Card) + (Opponent Card if seen) + (Action History)
        # For simplicity, a small fixed observation space might represent abstract states.
        # Your current state_mapping logic is abstracting states.
        # Let's simplify state for now to focus on rewards, then revisit.
        # For Kuhn Poker, a state might include (Player Card, History of actions so far).
        # A 12-dim observation space might be sufficient for a one-hot + history encoding.
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.uint8) # Placeholder
        
        self.max_episode_steps = 3 # Maximum 3 actions in Kuhn Poker
        self.cards_deck = [0, 1, 2] # Cards: Jack (0), Queen (1), King (2)
        
        self.player_card_idx = None
        self.opponent_card_idx = None
        self.player_turn = 0  # 0 for Player 0 (Agent), 1 for Player 1 (Adversary)
        self.history = []     # List of actions taken in sequence [P0_action, P1_action, P0_action_if_needed]
        self.pot = 0          # Tracks total money in pot, starts at 2 (1 from each player)
        self.state_mapping = {} 
        self.actions_taken = [] 

        # State mapping is for your abstract observation space, might be complex
        # if not carefully managed with actual game states.
        # For now, let's make get_obs simpler or remove state mapping until rewards are correct.
        # A simple state for Kuhn Poker is [player_card_one_hot] + [opponent_card_one_hot (if revealed)] + [action_history_one_hot]
        # Given your 12-dim state, it likely implies some compact encoding.
        # We'll stick to your `get_obs` for now but focus on `step`.

    def get_obs(self):
        state_key = self._get_state_key()
        if state_key not in self.state_mapping:
            index = min(len(self.state_mapping), self.observation_space.shape[0] - 1)
            one_hot = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            one_hot[index] = 1
            self.state_mapping[state_key] = one_hot
        obs = self.state_mapping[state_key].astype(np.float32)
        return obs


    def _get_state_key(self):
        """Generate a unique key for the current state based on card and action history."""
        # Your existing logic seems to try and represent the full game state.
        # This is where the actual game state for observation comes from.
        card_str = str(self.player_card_idx)
        history_str = "".join([str(a) for a in self.history]) # Use 0/1 directly
        
        # This logic is trying to infer a state key based on turn and history, which is complex.
        # For standard Kuhn Poker, the history `[P0_action, P1_action, P0_final_action]`
        # combined with the player's card defines the "state".
        return f"C{card_str}_H{history_str}_P{self.player_turn}"

    def reset(self, seed=None, options=None):
        """Reset the environment to a new game."""
        if seed is not None:
            np.random.seed(seed)
            # Important: `np.random.permutation` also uses numpy's global RNG
            rng = np.random.default_rng(seed)
            self.cards_deck = rng.permutation([0, 1, 2])
        else:
            self.cards_deck = np.random.permutation([0, 1, 2])
        
        self.player_card_idx = self.cards_deck[0]
        self.opponent_card_idx = self.cards_deck[1]
        
        # For clearer debugging:
        # self.player_card = ['J', 'Q', 'K'][self.player_card_idx]
        # self.opponent_card = ['J', 'Q', 'K'][self.opponent_card_idx]

        self.history = []     # Store actual actions (0 or 1)
        self.player_turn = 0  # Player 0 (Agent) starts
        self.pot = 2          # Ante: 1 from each player

        # Clear actions_taken if it's for worst_case_env_step, or remove if redundant
        if hasattr(self, 'actions_taken'):
            del self.actions_taken # Clear it if it's meant for a single episode
        self.actions_taken = [] # For compatibility with worst_case_env_step


        return self.get_obs()


    def step(self, action):
        action = int(action)
        if action not in [0, 1]:
            raise ValueError(f"Invalid action: {action}, must be 0 (pass) or 1 (bet)")
        reward = 0
        done = False
        truncated = False
        info = {"player_id": self.player_turn, "adv_action": -1}
  
        if self.player_turn == 0:
            self.history.append(action)
            if len(self.history) == 1:
                if action == 0:
                    self.player_turn = 1
                elif action == 1:
                    self.pot += 1
                    self.player_turn = 1
            elif len(self.history) == 3 and self.history[:2] == [0, 1]:
                if action == 0:
                    reward = -2
                    done = True
                elif action == 1:
                    self.pot += 1
                    reward = 2 if self.player_card_idx > self.opponent_card_idx else -2
                    done = True
                self.player_turn = -1
        elif self.player_turn == 1:
            self.history.append(action)
            info["adv_action"] = action
            if len(self.history) == 2:
                if self.history[0] == 0:
                    if action == 0:
                        reward = 1 if self.player_card_idx > self.opponent_card_idx else -1
                        done = True
                    elif action == 1:
                        self.pot += 1
                        self.player_turn = 0
                elif self.history[0] == 1:
                    self.pot += 1
                    if action == 0:
                        reward = 1
                        done = True
                    elif action == 1:
                        self.pot += 1
                        reward = 2 if self.player_card_idx > self.opponent_card_idx else -2
                        done = True
    
        return self.get_obs(), reward, done, truncated, info