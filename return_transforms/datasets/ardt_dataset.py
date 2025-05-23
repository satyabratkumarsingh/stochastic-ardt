import torch
from torch.utils.data import IterableDataset
from return_transforms.utils.utils import return_labels


class ARDTDataset(IterableDataset):
    """
    This class provides an iterable dataset for handling trajectories and preparing
    batches for model training of adversarial trajectories.

    Args:
        trajs (list): List of trajectory data, where each Trajectory has obs, actions, rewards,
                      adv_actions, adv_rewards as arrays, and infos as list of dicts.
        horizon (int): The maximum length of a trajectory.
        gamma (float): Discount factor for return calculation. Default is 1.
        act_type (str): Type of actions ('discrete' or 'continuous'). Default is 'discrete'.
        epoch_len (int): Number of iterations (or samples) per epoch. Default is 1e5.
        new_rewards (bool): Whether to use new rewards when calculating returns. Default is False.
    """

    def __init__(
            self, 
            trajs: list, 
            horizon: int,
            gamma: float = 1, 
            act_type: str = 'discrete', 
            epoch_len: float = 1e5, 
            new_rewards: bool = False, 
    ):
        if not trajs:
            raise ValueError("Trajectory list is empty")
        
        self.trajs = trajs
        self.rets = [return_labels(traj, gamma, new_rewards) for traj in self.trajs]
        
        # Dynamically determine n_actions and n_adv_actions from the first trajectory
        self.n_actions = trajs[0].actions.shape[-1]
        self.n_adv_actions = trajs[0].adv_actions.shape[-1]
        
        # Validate that all trajectories have consistent action dimensions
        for traj in trajs[1:]:
            if traj.actions.shape[-1] != self.n_actions:
                raise ValueError(f"Inconsistent action dimension: {traj.actions.shape[-1]} vs {self.n_actions}")
            if traj.adv_actions.shape[-1] != self.n_adv_actions:
                raise ValueError(f"Inconsistent adv_action dimension: {traj.adv_actions.shape[-1]} vs {self.n_adv_actions}")
        
        self.horizon = horizon
        self.act_type = act_type
        self.epoch_len = epoch_len
        self.new_rewards = new_rewards

    def segment_generator(self, epoch_len: int):
        """
        Generator function to yield padded segments of trajectory data.

        Args:
            epoch_len (int): Number of samples to generate in this epoch.

        Yields:
            tuple: Padded observations, actions, adversarial actions, returns, and true sequence length.
        """
        for _ in range(epoch_len):
            # Sample a random trajectory
            traj_idx = torch.randint(0, len(self.trajs), (1,), generator=self.rand).item()
            traj = self.trajs[traj_idx]
            rets = self.rets[traj_idx]
            obs = torch.tensor(traj.obs, dtype=torch.float32)

            # Handling different action types
            if self.act_type == 'discrete':
                # Actions and adv_actions are one-hot encoded arrays
                actions = torch.tensor(traj.actions, dtype=torch.float32)
                adv_actions = torch.tensor(traj.adv_actions, dtype=torch.float32)
            else:
                # For continuous actions, assume actions are not one-hot encoded
                actions = torch.tensor(traj.actions, dtype=torch.float32)
                adv_actions = torch.tensor(traj.adv_actions, dtype=torch.float32)

            # Padding the trajectories to the defined horizon length
            padded_obs = torch.zeros((self.horizon, *obs.shape[1:]), dtype=torch.float32)
            padded_acts = torch.zeros((self.horizon, self.n_actions), dtype=torch.float32)
            padded_adv_acts = torch.zeros((self.horizon, self.n_adv_actions), dtype=torch.float32)
            padded_rets = torch.zeros(self.horizon, dtype=torch.float32)
            true_seq_length = obs.shape[0]

            # Fill padded tensors
            padded_obs[:obs.shape[0]] = obs
            padded_acts[:actions.shape[0]] = actions
            padded_adv_acts[:adv_actions.shape[0]] = adv_actions
            padded_rets[:len(rets)] = torch.tensor(rets, dtype=torch.float32)

            # Yield the padded trajectory segments as tensors
            yield (
                padded_obs,
                padded_acts,
                padded_adv_acts,
                padded_rets,
                torch.tensor(true_seq_length, dtype=torch.int32)
            )

    def __len__(self):
        return int(self.epoch_len)

    def __iter__(self):
        """
        Returns an iterator for the dataset.

        If using multiple workers for data loading, each worker gets a split of the data.
        """
        worker_info = torch.utils.data.get_worker_info()
        self.rand = torch.Generator()  # Use torch.Generator for random sampling
        
        if worker_info is None:
            # Single-worker setup
            gen = self.segment_generator(int(self.epoch_len))
        else:
            # Multi-worker setup: Split the workload across workers
            per_worker_time_steps = int(self.epoch_len / float(worker_info.num_workers))
            gen = self.segment_generator(per_worker_time_steps)
        
        return gen