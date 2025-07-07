import gym
import numpy as np
from gym import spaces

class KuhnPokerEnv(gym.Env):
    def __init__(self):
        self.action_space = spaces.Discrete(2)
        self.adv_action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(low=0, high=1, shape=(12,), dtype=np.uint8)
        self.cards = [0, 1, 2]
        self.state = None
        self.action_history = []
        self.player = 0
        self.state_mapping = {}

    def get_obs(self):
        state_key = self._get_state_key()
        if state_key not in self.state_mapping:
            index = len(self.state_mapping)
            one_hot = np.zeros(self.observation_space.shape[0])
            one_hot[min(index, self.observation_space.shape[0] - 1)] = 1
            self.state_mapping[state_key] = one_hot
        return self.state_mapping[state_key]

    def _get_state_key(self):
        if not self.action_history:
            return str(self.cards[self.player])
        history = "".join(["p" if a == 0 else "b" for a in self.action_history])
        if self.player == 0 and len(self.action_history) == 2:
            return f"{self.cards[0]}p{history[-1]}"
        elif self.player == 1:
            return f"{self.cards[1]}{history}"
        return str(self.cards[self.player])

    def reset(self):
        self.cards = np.random.permutation([0, 1, 2])[:2]
        self.state = None
        self.action_history = []
        self.player = 0
        return self.get_obs()

    def step(self, action):
        self.action_history.append(action)
        reward = 0
        done = False
        info = {"player_id": self.player, "adv": int(action == 1)}
        if self.player == 0:
            self.player = 1
            return self.get_obs(), reward, done, info
        elif self.player == 1:
            adv_action = np.random.choice(2)
            self.action_history.append(adv_action)
            info["adv"] = int(adv_action == 1)
            if self.action_history == [0, 0]:
                done = True
                reward = 1 if self.cards[1] > self.cards[0] else -1
            elif self.action_history == [0, 1]:
                self.player = 0
                return self.get_obs(), reward, done, info
            elif self.action_history == [1, 0]:
                done = True
                reward = -1
            elif self.action_history == [1, 1]:
                done = True
                reward = 2 if self.cards[1] > self.cards[0] else -2
            if len(self.action_history) == 3 and self.action_history[:2] == [0, 1]:
                done = True
                reward = 2 if self.cards[1] > self.cards[0] else -2
        return self.get_obs(), reward, done, info