import gym
import numpy as np
from gym import spaces
import torch

class KuhnPokerEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.env_name = "kuhn_poker"
        
        # Action spaces: 0=Pass/Check/Fold, 1=Bet/Call
        self.action_space = spaces.Discrete(3)
        self.adv_action_space = spaces.Discrete(3)
        
        # Build state mapping with only valid non-terminal states
        self.state_mapping, self.state_dim = self._build_kuhn_state_mapping()
        
        # Observation space
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(self.state_dim,), dtype=np.float32
        )
        
        self.max_episode_steps = 3
        self.cards = [0, 1, 2]  # Jack=0, Queen=1, King=2
        
        # Standard Kuhn Poker payoffs (ante=1, bet=1)
        self.ante = 1
        self.bet_size = 1
        
        self.reset()

    def _build_kuhn_state_mapping(self):
        """Build state mapping for non-terminal Kuhn Poker states only"""
        canonical_states = [
            # Initial states (card only)
            "0", "1", "2",
            # After first pass (card + p)
            "0p", "1p", "2p",
            # After first bet (card + b) 
            "0b", "1b", "2b",
            # After pass-bet (card + pb) - still non-terminal
            "0pb", "1pb", "2pb"
        ]
        
        state_dim = len(canonical_states)
        state_mapping = {
            state: np.eye(state_dim, dtype=np.float32)[i] 
            for i, state in enumerate(canonical_states)
        }
        
        return state_mapping, state_dim

    def reset(self, seed=None, options=None):
        if seed is not None:
            np.random.seed(seed)
        
        dealt_cards = np.random.choice(self.cards, size=2, replace=False)
        self.player_card = dealt_cards[0]
        self.opponent_card = dealt_cards[1]
        
        self.history = ""
        self.current_player = 0
        self.game_over = False
        self.pot = 2 * self.ante  # Both players ante up
        
        self.episode_data = {
            'obs': [],
            'actions': [],
            'adv_actions': [],
            'rewards': [],
            'adv_rewards': [],
            'infos': [],
            'dones': []
        }
        
        # Record initial observation
        self._record_timestep(None, None)
        return self.get_obs(), {}

    def get_obs(self):
        """Get observation from current player's perspective for non-terminal states only"""
        if self.game_over:
            # Terminal states have no observation
            return np.zeros(self.state_dim, dtype=np.float32)
        
        # Always from Player 0's perspective for consistency
        state_key = str(self.player_card) + self.history
        if state_key in self.state_mapping:
            return self.state_mapping[state_key].copy()
        else:
            # Should not happen now
            raise ValueError(f"Unknown state '{state_key}' in non-terminal observation!")

    def step(self, action):
        if self.game_over:
            raise ValueError("Game over. Call reset().")
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action {action}. Use 0=Pass/Check/Fold, 1=Bet/Call, 2=NoAction")
        
        reward, done = self._execute_action(action)
        self._record_timestep(action, reward)
        
        info = {
            "player_id": self.current_player,
            "action": action,
            "history": self.history,
            "terminal": done,
            "pot": self.pot
        }
        
        return self.get_obs(), reward, done, False, info

    def _execute_action(self, action):
        """Execute action and return reward and done status (always from Player 0's perspective)"""
        # Add bet to pot if player bets/calls
        if action == 1:
            self.pot += self.bet_size
        
        self.history += "p" if action == 0 else "b"
        reward = 0
        done = False
        
        # Terminal conditions - calculate Player 0's final payoff
        if self.history == "pp":  # Both pass
            done = True
            if self.player_card > self.opponent_card:
                reward = self.ante  # Player 0 wins ante from opponent
            else:
                reward = -self.ante  # Player 0 loses ante to opponent
                
        elif self.history == "bp":  # Player 0 bets, Player 1 folds
            done = True
            reward = self.ante  # Player 0 wins opponent's ante
            
        elif self.history == "bb":  # Both bet - showdown
            done = True
            if self.player_card > self.opponent_card:
                reward = self.ante + self.bet_size  # Win ante + bet
            else:
                reward = -(self.ante + self.bet_size)  # Lose ante + bet
                
        elif self.history == "pbp":  # Player 0 passes, Player 1 bets, Player 0 folds
            done = True  
            reward = -self.ante  # Player 0 loses ante
            
        elif self.history == "pbb":  # Player 0 passes, Player 1 bets, Player 0 calls
            done = True
            if self.player_card > self.opponent_card:
                reward = self.ante + self.bet_size  # Win ante + bet
            else:
                reward = -(self.ante + self.bet_size)  # Lose ante + bet
        
        if done:
            self.game_over = True
            
        if not done:
            # Switch players
            self.current_player = 1 - self.current_player
            
        return reward, done

    def _record_timestep(self, action, reward):
        """Record timestep data consistently from Player 0's perspective"""
        if not self.game_over:
            self.episode_data['obs'].append(self.get_obs())
        
        if action is not None:
            if self.current_player == 0:
                # Player 0's turn
                self.episode_data['actions'].append(self.action_to_onehot(action))
                self.episode_data['adv_actions'].append([0.0, 0.0, 1.0])  # Opponent didn't act
                self.episode_data['rewards'].append(reward)
                self.episode_data['adv_rewards'].append(-reward)
            else:
                # Player 1's turn - record as adversarial action
                self.episode_data['actions'].append([0.0, 0.0, 1.0])  # Player 0 didn't act
                self.episode_data['adv_actions'].append(self.action_to_onehot(action))
                # Rewards from Player 0's perspective
                self.episode_data['rewards'].append(-reward if reward != 0 else 0)
                self.episode_data['adv_rewards'].append(reward if reward != 0 else 0)
            
            # Info dictionary
            info = {
                "player_id": self.current_player,
                "action": action,
                "no_action": action == 2
            }
            self.episode_data['infos'].append(info)
            self.episode_data['dones'].append(self.game_over)

    def action_to_onehot(self, action):
        """Convert action to one-hot encoding for Kuhn Poker"""
        if action == 0:
            return [1.0, 0.0, 0.0]  # Pass/Check/Fold
        elif action == 1:
            return [0.0, 1.0, 0.0] # Bet/Call
        elif action == 2:
            return [0.0, 0.0, 1.0] # NoAction
        else:
            raise ValueError(f"Invalid action {action}. Kuhn Poker only has actions 0 (pass) and 1 (bet)")

    def get_episode_data(self):
        """Return episode data with minimax returns-to-go"""
        if not self.game_over:
            return None
            
        # Calculate standard returns-to-go
        rewards = self.episode_data['rewards']
        returns_to_go = []
        cumulative_return = 0
        for reward in reversed(rewards):
            cumulative_return += reward
            returns_to_go.insert(0, cumulative_return)
        
        # For this simple environment, use the same values for minimax RTG
        # In practice, these would come from minimax value calculations
        minimax_returns_to_go = returns_to_go.copy()
        
        return {
            "episode_id": np.random.randint(1, 1000000),
            "obs": self.episode_data['obs'],
            "actions": self.episode_data['actions'],
            "adv_actions": self.episode_data['adv_actions'], 
            "rewards": self.episode_data['rewards'],
            "adv_rewards": self.episode_data['adv_rewards'],
            "infos": self.episode_data['infos'],
            "dones": self.episode_data['dones'],
            "minimax_returns_to_go": minimax_returns_to_go
        }

    def get_legal_actions(self):
        """Get legal actions for current state"""
        return [] if self.game_over else [0, 1]

    def get_nash_equilibrium_value(self):
        """Return the Nash equilibrium value for Player 0 in standard Kuhn Poker"""
        # This is the theoretical value with ante=1, bet=1
        return 1/18

    def render(self, mode='human'):
        """Render the current game state"""
        cards = ['Jack', 'Queen', 'King']
        player_card_name = cards[self.player_card]
        opponent_card_name = cards[self.opponent_card] if self.game_over else "Hidden"
        
        print(f"=== Kuhn Poker ===")
        print(f"Player 0 card: {player_card_name}")
        print(f"Player 1 card: {opponent_card_name}")
        print(f"History: '{self.history}'")
        print(f"Current player: {self.current_player}")
        print(f"Pot: {self.pot}")
        print(f"Game over: {self.game_over}")
        print("==================")

    def get_optimal_strategy(self, player_card, history):
        """
        Get the Nash equilibrium mixed strategy for given state
        Returns probabilities for [pass/fold, bet/call]
        """
        state_key = f"{player_card}{history}"
        
        # Nash equilibrium strategies for standard Kuhn Poker
        nash_strategies = {
            # Player 0 (first to act)
            "0": [2/3, 1/3],      # Jack: Pass 2/3, Bet 1/3
            "1": [1.0, 0.0],      # Queen: Always Pass
            "2": [0.0, 1.0],      # King: Always Bet
            
            # Player 1 responses to pass
            "0p": [2/3, 1/3],     # Jack: Check 2/3, Bet 1/3  
            "1p": [1.0, 0.0],     # Queen: Always Check
            "2p": [0.0, 1.0],     # King: Always Bet
            
            # Player 1 responses to bet
            "0b": [1.0, 0.0],     # Jack: Always Fold
            "1b": [2/3, 1/3],     # Queen: Fold 2/3, Call 1/3
            "2b": [0.0, 1.0],     # King: Always Call
            
            # Player 0 responses to pass-bet
            "0pb": [1.0, 0.0],    # Jack: Always Fold
            "1pb": [2/3, 1/3],    # Queen: Fold 2/3, Call 1/3  
            "2pb": [0.0, 1.0],    # King: Always Call
        }
        
        return nash_strategies.get(state_key, [0.5, 0.5])  # Default to uniform if unknown