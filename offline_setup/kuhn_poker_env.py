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
        # This state mapping is custom and crucial for your DT. Ensure it's correct.
        # For now, let's just return a placeholder, or a one-hot based on player_card_idx.
        # You need to ensure this actually creates distinct states for the DT to learn.

        # A more standard observation for Kuhn Poker:
        # Player's card (one-hot) + actions taken so far (one-hot or indices)
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        if self.player_card_idx is not None:
            # Assuming first 3 dims are for player card J, Q, K
            obs[self.player_card_idx] = 1.0 
        
        # Then encode history into the rest of the observation.
        # This is where your state_mapping comes in.
        # For debugging, let's keep it simple:
        # You need to fill this out according to how you encode game history into state.
        # Example for history: 0=P, 1=B. History 'PB' means `[0,1]`.
        # You need to map [P0_action, P1_action, P0_final_action] onto your 12-dim state.
        # The `_get_state_key` logic seems to be creating these distinct states.
        
        state_key = self._get_state_key()
        # If your state_mapping is supposed to provide the full 12-dim state, let's trust it.
        # BUT the way it creates new indices might lead to duplicate states if not careful.
        if state_key not in self.state_mapping:
            # This logic will just fill the next available index.
            # This won't create meaningful states for a DT to learn from
            # if the states are not distinct in a semantically useful way.
            # For a given (player_card, history), it should always map to the same observation.
            current_index = len(self.state_mapping) 
            # This is problematic. If J-P maps to index 0, Q-P should map to something else.
            # It seems to treat all unique history-card combos as a new index.
            # This is more like state ID generation, not state representation.

            # For now, let's assume `get_obs` needs to convert the actual game state
            # (player_card_idx, history) into the 12-dim vector.
            # This means your `state_mapping` needs to be pre-defined or more robust.
            # Let's simplify and make `get_obs` just return a one-hot of the player card
            # and zeros for history for now, just to get rewards working.
            # YOU MUST REPLACE THIS WITH YOUR ACTUAL STATE ENCODING.
            # For the 12-dim space, you could have:
            # [P0_card_J, P0_card_Q, P0_card_K, History_empty, History_P, History_B, History_PP, History_PB, History_BP, History_BB, History_BP_P0F, History_BP_P0C]
            # This is complex. Let's return a simple fixed state for now to debug rewards.
            # return np.zeros(self.observation_space.shape, dtype=np.float32) # For debugging reward only
            
            # Let's re-use your state_mapping approach for now, but be aware it's unusual.
            index = len(self.state_mapping) # Current size of mapping is the next available index
            # This effectively makes a new one-hot for every unique (card, history) combination encountered.
            # Max possible states for Kuhn: 3 cards * (1 (initial) + 2 (P0 pass/bet) + 4 (P1 pass/bet) + 2 (P0's final))
            # ~ 3 * (1+2+4+2) = 27 theoretical game states. 12-dim is small for one-hotting all.
            # If 12 means one-hot of 12 distinct abstract states, this is fine.
            # Otherwise, it might be an issue. Let's proceed assuming this is what you intended.
            index = min(index, self.observation_space.shape[0] - 1) # Cap index at 11
            one_hot = np.zeros(self.observation_space.shape[0], dtype=np.uint8)
            one_hot[index] = 1
            self.state_mapping[state_key] = one_hot
        return self.state_mapping[state_key].astype(np.float32)


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
        """
        Take a step in the environment with the given action.
        `action` is the action of the *current* player (`self.player_turn`).
        """
        action = int(action)
        if action not in [0, 1]:
            raise ValueError(f"Invalid action: {action}, must be 0 (pass) or 1 (bet)")
        
        reward = 0
        done = False
        truncated = False
        info = {"player_id": self.player_turn, "adv_action": -1} # Default adversary action to -1 (none)
       
        # Important: The `worst_case_env_step` needs to call `env.step` with the
        # correct player's action (agent's or adversary's forced action).
        # Your current `worst_case_env_step` always passes the agent's action.
        # This is where the core issue for game flow lies.

        # Let's assume for now that the `action` argument passed to this `step` method
        # is always the *correct* action for `self.player_turn`.
        # This means `worst_case_env_step` MUST ensure this.

        current_player_action = action # This is the action from the agent or adversary.

        if self.player_turn == 0: # Agent's turn
            self.history.append(current_player_action)
            
            if len(self.history) == 1: # P0's first action
                if current_player_action == 0: # P0 Pass
                    self.player_turn = 1 # Turn to P1
                elif current_player_action == 1: # P0 Bet
                    self.pot += 1 # P0 contributes to pot
                    self.player_turn = 1 # Turn to P1
            
            # This is the path for P0's *second* action, only if P0 Pass, P1 Bet
            elif len(self.history) == 3 and self.history[:2] == [0, 1]: # P0 passed, P1 bet, now P0 acts again
                if current_player_action == 0: # P0 Folds
                    reward = -2 # P0 loses ante + P1's bet
                    done = True
                elif current_player_action == 1: # P0 Calls (showdown)
                    self.pot += 1 # P0 contributes to pot
                    # Showdown
                    if self.player_card_idx > self.opponent_card_idx:
                        reward = 2 # P0 wins (ante + P1's bet)
                    else:
                        reward = -2 # P0 loses
                    done = True
                self.player_turn = -1 # Game ends
            
            if not done: # If game didn't end, return control
                return self.get_obs(), reward, done, truncated, info

        elif self.player_turn == 1: # Adversary's turn
            # Assuming `current_player_action` is the adversary's choice (forced by wrapper)
            self.history.append(current_player_action)
            info["adv_action"] = current_player_action # Store the actual adv action taken

            # Based on the sequence of moves and current player's action
            # P1 acts after P0's initial move
            if len(self.history) == 2:
                # P0's first action was self.history[0]
                # P1's action is current_player_action
                
                if self.history[0] == 0: # P0 Passed
                    if current_player_action == 0: # P1 Pass (Pass-Pass) -> Showdown
                        reward = 1 if self.player_card_idx > self.opponent_card_idx else -1
                        done = True
                    elif current_player_action == 1: # P1 Bet (Pass-Bet) -> P0 acts again
                        self.pot += 1 # P1 contributes to pot
                        self.player_turn = 0 # Turn to P0
                
                elif self.history[0] == 1: # P0 Bet
                    self.pot += 1 # Add P0's bet to pot for P1's consideration
                    if current_player_action == 0: # P1 Folds (Bet-Pass) -> P0 wins
                        reward = 1 # P0 wins ante + P0's bet
                        done = True
                    elif current_player_action == 1: # P1 Calls (Bet-Bet) -> Showdown
                        self.pot += 1 # P1 contributes to pot
                        if self.player_card_idx > self.opponent_card_idx:
                            reward = 2 # P0 wins (ante + P0 bet + P1 bet)
                        else:
                            reward = -2 # P0 loses
                        done = True
            

                if not done: # If game didn't end, return control
                    return self.get_obs(), reward, done, truncated, info

            else:
                # Should not happen in basic Kuhn Poker logic if sequence is correctly managed
                pass # Or raise error

       
        
        # If the game didn't end, and it's not a turn change where we returned early, something is off.
        # This return is for when a game scenario leads to termination.
        return self.get_obs(), reward, done, truncated, info