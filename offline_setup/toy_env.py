from offline_setup.toy_setup import ToyEnv
from offline_setup.random_policy import RandomPolicy
from offline_setup.base_offline_env import BaseOfflineEnv

class ToyOfflineEnv(BaseOfflineEnv):

    def __init__(self, path, horizon=5, n_interactions=int(1e5)):
        self.env_cls = lambda: ToyEnv()
        self.test_env_cls = lambda: ToyEnv()

        def data_policy_fn():
            test_env = self.env_cls()
            test_env.action_space
            data_policy = RandomPolicy(test_env.action_space)
            return data_policy

        super().__init__(path, self.env_cls, data_policy_fn, horizon, n_interactions)