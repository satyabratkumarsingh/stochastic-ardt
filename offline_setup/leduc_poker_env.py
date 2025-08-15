
import gym
import numpy as np
from gym import spaces

class LeducPokerEnv(gym.Env):
    """
    A gym environment for the two-player Leduc Poker game.
    Updated to match dataset format with 3 actions: 0=fold, 1=call/check, 2=bet/raise
    """
    def __init__(self):
        super().__init__()
        self.env_name = "leduc_poker"
        
        # Actions: 0 = fold, 1 = call/check, 2 = bet/raise (matching your dataset)
        self.action_space = spaces.Discrete(3)
        self.adv_action_space = spaces.Discrete(3)

        # Observation space for one-hot encoded states (expanded to match dataset complexity)
        self.observation_space = spaces.Box(low=0, high=1, shape=(288,), dtype=np.uint8)

        self.max_episode_steps = 8  # Increased for more complex game tree
        self.cards_deck = [0, 1, 2, 3, 4, 5]  # 6 cards: 0-5 matching your dataset

        # Game state variables
        self.player_card = None
        self.opponent_card = None
        self.community_card = None
        self.player_turn = 0  # 0 for Player 0, 1 for Player 1
        self.betting_round = 1 # 1 or 2
        self.history = []     # Stores actions taken in current round
        self.round1_history = []  # Complete round 1 action history
        self.pot = 2          # Tracks total money in pot (starts with antes)
        self.player_money = [99, 99]  # Player money after antes

        # State mapping for creating one-hot vectors
        self.state_mapping = {}

    def _get_state_key(self):
        """Generates a unique key matching your dataset format."""
        # Create state key in format: [Observer: X][Private: X][Round X][Player: X][Pot: X][Money: X X][Public: X][Round1: X][Round2: X]
        observer = self.player_turn
        private_card = self.player_card if observer == 0 else self.opponent_card
        
        # Format action sequences to match dataset
        round1_str = " ".join([str(a) for a in self.round1_history]) if self.round1_history else ""
        round2_str = " ".join([str(a) for a in self.history if self.betting_round == 2]) if self.betting_round == 2 else ""
        
        public_card = self.community_card if self.community_card is not None else -1
        money_str = f"{self.player_money[0]} {self.player_money[1]}"
        
        return f"[Observer: {observer}][Private: {private_card}][Round {self.betting_round}][Player: {self.player_turn}][Pot: {self.pot}][Money: {money_str}][Public: {public_card}][Round1: {round1_str}][Round2: {round2_str}]"

    def get_obs(self):
        """Creates a one-hot observation vector from the state key."""
        state_key = self._get_state_key()
        if state_key not in self.state_mapping:
            index = min(len(self.state_mapping), self.observation_space.shape[0] - 1)
            one_hot = np.zeros(self.observation_space.shape[0], dtype=np.float32)
            one_hot[index] = 1
            self.state_mapping[state_key] = one_hot
        obs = self.state_mapping[state_key].astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        """Resets the environment to a new game."""
        # Use random number generator for reproducibility
        if seed is not None:
            rng = np.random.default_rng(seed)
            self.cards_deck = rng.permutation([0, 1, 2, 3, 4, 5])
        else:
            self.cards_deck = np.random.permutation([0, 1, 2, 3, 4, 5])
        
        # Deal cards
        self.player_card = self.cards_deck[0]
        self.opponent_card = self.cards_deck[1]
        self.community_card = None
        
        # Initialize game state
        self.player_turn = 0
        self.betting_round = 1
        self.history = []
        self.round1_history = []
        self.pot = 2  # Ante of 1 from each player
        self.player_money = [99, 99]  # Money after antes

        return self.get_obs(), {}

    def _calculate_reward(self):
        """Determines the winner and calculates the reward at showdown."""
        # Check for pairs (card matches community card)
        p0_pair = (self.player_card == self.community_card)
        p1_pair = (self.opponent_card == self.community_card)

        if p0_pair and not p1_pair:
            return 1  # P0 wins
        elif not p0_pair and p1_pair:
            return -1  # P1 wins, P0 loses
        elif p0_pair and p1_pair:
            # Both have a pair, higher rank wins
            if self.player_card > self.opponent_card:
                return 1
            elif self.player_card < self.opponent_card:
                return -1
            else:
                return 0  # Tie
        else:
            # Neither has a pair, higher card wins
            if self.player_card > self.opponent_card:
                return 1
            elif self.player_card < self.opponent_card:
                return -1
            else:
                return 0  # Tie

    def step(self, action):
        """Takes a step in the environment based on the current player's action."""
        action = int(action)
        if action not in [0, 1, 2]:
            raise ValueError(f"Invalid action: {action}, must be 0 (fold), 1 (call/check), or 2 (bet/raise)")

        reward = 0
        done = False
        truncated = False
        info = {"player_id": self.player_turn, "adv_action": -1}
        
        self.history.append(action)

        # Handle fold action (new addition)
        if action == 0:  # Fold
            done = True
            # Player who folds loses, opponent wins
            reward = -1 if self.player_turn == 0 else 1
            self.player_turn = -1  # Game over
            return self.get_obs(), reward, done, truncated, info

        # Betting Round 1
        if self.betting_round == 1:
            if len(self.history) == 1:  # Player 0's first action
                if action == 2:  # Player 0 bets/raises
                    self.pot += 2  # Bet amount
                    self.player_money[0] -= 2
                self.player_turn = 1
                
            elif len(self.history) == 2:  # Player 1's action
                info["adv_action"] = action
                if self.history[0] == 1 and action == 1:  # Check-check, proceed to round 2
                    self.round1_history = self.history.copy()
                    self.betting_round = 2
                    self.history = []
                    self.player_turn = 0
                    self.community_card = self.cards_deck[2]  # Deal community card
                elif self.history[0] == 1 and action == 2:  # P0 checks, P1 bets
                    self.pot += 2
                    self.player_money[1] -= 2
                    self.player_turn = 0
                elif self.history[0] == 2 and action == 1:  # P0 bets, P1 calls
                    self.pot += 2
                    self.player_money[1] -= 2
                    self.round1_history = self.history.copy()
                    self.betting_round = 2
                    self.history = []
                    self.player_turn = 0
                    self.community_card = self.cards_deck[2]  # Deal community card
                elif self.history[0] == 2 and action == 2:  # P0 bets, P1 raises
                    self.pot += 4  # P1 raises (call + raise)
                    self.player_money[1] -= 4
                    self.player_turn = 0  # P0 needs to respond to raise
            
            # Additional logic for P0's response to P1's actions
            elif len(self.history) == 3 and self.betting_round == 1:  # P0's response
                if self.history[0] == 1 and self.history[1] == 2:  # P0 checked, P1 bet, P0 responds
                    if action == 1:  # P0 calls
                        self.pot += 2
                        self.player_money[0] -= 2
                        self.round1_history = self.history.copy()
                        self.betting_round = 2
                        self.history = []
                        self.player_turn = 0
                        self.community_card = self.cards_deck[2]
                    elif action == 2:  # P0 raises
                        self.pot += 4  # Call + raise
                        self.player_money[0] -= 4
                        self.player_turn = 1  # P1 needs to respond
                        
                elif self.history[0] == 2 and self.history[1] == 2:  # P0 bet, P1 raised, P0 responds
                    if action == 1:  # P0 calls the raise
                        self.pot += 2  # Call the additional raise amount
                        self.player_money[0] -= 2
                        self.round1_history = self.history.copy()
                        self.betting_round = 2
                        self.history = []
                        self.player_turn = 0
                        self.community_card = self.cards_deck[2]
                    elif action == 2:  # P0 re-raises
                        self.pot += 4  # Call + re-raise
                        self.player_money[0] -= 4
                        self.player_turn = 1
        
        # Betting Round 2
        elif self.betting_round == 2:
            if len(self.history) == 1:  # Player 0's action in round 2
                if action == 2:  # P0 bets
                    self.pot += 4  # Round 2 bet size
                    self.player_money[0] -= 4
                self.player_turn = 1
                
            elif len(self.history) == 2:  # Player 1's action in round 2
                info["adv_action"] = action
                if self.history[0] == 1 and action == 1:  # Check-check, showdown
                    done = True
                    reward = self._calculate_reward()
                elif self.history[0] == 1 and action == 2:  # P0 checks, P1 bets
                    self.pot += 4
                    self.player_money[1] -= 4
                    self.player_turn = 0  # P0 needs to respond
                elif self.history[0] == 2 and action == 1:  # P0 bets, P1 calls
                    self.pot += 4
                    self.player_money[1] -= 4
                    done = True
                    reward = self._calculate_reward()
                elif self.history[0] == 2 and action == 2:  # P0 bets, P1 raises
                    self.pot += 8  # Call + raise
                    self.player_money[1] -= 8
                    self.player_turn = 0  # P0 needs to respond
                    
            elif len(self.history) == 3 and self.betting_round == 2:  # Final actions
                if self.history[0] == 1 and self.history[1] == 2:  # P0 checked, P1 bet, P0 responds
                    if action == 1:  # P0 calls
                        self.pot += 4
                        self.player_money[0] -= 4
                        done = True
                        reward = self._calculate_reward()
                    # Fold is handled at the beginning
                elif self.history[0] == 2 and self.history[1] == 2:  # P0 bet, P1 raised, P0 responds
                    if action == 1:  # P0 calls
                        self.pot += 4  # Call the raise
                        self.player_money[0] -= 4
                        done = True
                        reward = self._calculate_reward()

        if done:
            self.player_turn = -1  # Game over
            # Reset money for next game (as shown in your dataset)
            if self.player_money[0] + self.player_money[1] + self.pot <= 200:
                self.player_money = [100, 100]
                self.pot = 0
        
        return self.get_obs(), reward, done, truncated, info