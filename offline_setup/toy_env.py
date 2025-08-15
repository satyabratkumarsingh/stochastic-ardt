from offline_setup.toy_setup import ToyEnv
from offline_setup.kuhn_poker_env import KuhnPokerEnv
from offline_setup.random_policy import RandomPolicy
from offline_setup.base_offline_env import BaseOfflineEnv
from offline_setup.leduc_poker_env import LeducPokerEnv
class ToyOfflineEnv(BaseOfflineEnv):
    """
    A flexible offline environment that can be configured for either
    Kuhn Poker or Leduc Poker.
    """
    def __init__(self, game="kuhn_poker", path="", horizon=5, n_interactions=int(1e5)):
        # Determine which environment to use based on the env_type parameter
        if game == "kuhn_poker":
            env_cls_to_use = KuhnPokerEnv
        elif game == "leduc_poker":
            env_cls_to_use = LeducPokerEnv
        else:
            raise ValueError(f"Unknown game type: {game}. Use 'kuhn_poker' or 'leduc_poker'.")

        self.env_cls = lambda: env_cls_to_use()
        self.test_env_cls = lambda: env_cls_to_use()

        def data_policy_fn():
            # The test environment is now dynamically created based on env_type
            test_env = self.test_env_cls()
            data_policy = RandomPolicy(test_env.action_space)
            return data_policy

        # Call the parent constructor with the selected environment class
        super().__init__(path, self.env_cls, data_policy_fn, horizon, n_interactions)