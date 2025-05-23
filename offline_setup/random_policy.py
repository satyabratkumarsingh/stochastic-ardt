from offline_setup.base_policy import BasePolicy, PolicyStep
from collections import namedtuple

PolicyInfo = namedtuple("PolicyInfo", [])


class RandomPolicy(BasePolicy):

    def __init__(self, action_space):
        self.action_space = action_space
        self._name = 'random'

    def sample(self, obs, reward, t):
        action = self.action_space.sample()
        return PolicyStep(action=action, info=PolicyInfo())
