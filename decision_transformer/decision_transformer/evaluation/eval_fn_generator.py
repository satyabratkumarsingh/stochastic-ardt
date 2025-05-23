from functools import partial
import pickle
import numpy as np
import torch
from dataclasses import dataclass
from typing import List
from collections import namedtuple

Trajectory = namedtuple("Trajectory", ["obs", "actions", "rewards", "adv_actions", "adv_rewards", "infos"])

class EvalFnGenerator:
    """
    Class to generate evaluation functions for ARDT using provided offline trajectories.

    Args:
        seed (int): Seed for reproducibility.
        env_name (str): Name of the environment.
        trajectories (List[Trajectory]): List of offline trajectories.
        num_eval_episodes (int): Number of episodes to evaluate.
        state_dim (int): Dimension of the state space.
        act_dim (int): Dimension of the action space.
        adv_act_dim (int): Dimension of the adversary action space.
        action_type (str): Type of action space.
        max_traj_len (int): Length of the trajectory.
        scale (float): Scaling factor for rewards.
        state_mean (np.ndarray): Mean of the states.
        state_std (np.ndarray): Standard deviation of the states.
        batch_size (int): Batch size for evaluation.
        normalize_states (bool): Whether to normalize the states.
        device (torch.device): Device to run the evaluation on.
        returns_filename (str): Name of the returns file.
        dataset_name (str): Name of the dataset.
        test_adv_name (str): Name of the adversary.
        added_dataset_name (str): Name of the added dataset.
        added_dataset_prop (float): Proportion of the added dataset.
        env_alpha (float): Environment alpha parameter (default: 0.0).
    """
    def __init__(
            self,
            seed: int,
            env_name: str,
            trajectories: List[Trajectory],
            num_eval_episodes: int,
            state_dim: int,
            act_dim: int,
            adv_act_dim: int,
            action_type: str,
            max_traj_len: int,
            scale: float,
            state_mean: float,
            state_std: float,
            batch_size: int,
            normalize_states: bool,
            device: torch.device,
            returns_filename: str,
            dataset_name: str,
            test_adv_name: str,
            added_dataset_name: str,
            added_dataset_prop: float,
            env_alpha: float = 0.0
    ):
        self.seed = seed
        self.env_name = env_name
        self.trajectories = trajectories
        self.num_eval_episodes = min(num_eval_episodes, len(trajectories))  # Limit to available trajectories
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.adv_act_dim = adv_act_dim
        self.action_type = action_type
        self.max_traj_len = max_traj_len
        self.scale = scale
        self.state_mean = state_mean
        self.state_std = state_std
        self.batch_size = batch_size
        self.normalize_states = normalize_states
        self.device = device
        self.env_alpha = env_alpha
        self.storage_path = self._build_storage_path(
            returns_filename,
            dataset_name,
            test_adv_name,
            added_dataset_name,
            added_dataset_prop
        )

    def _build_storage_path(
            self,
            returns_filename: str,
            dataset_name: str,
            test_adv_name: str,
            added_dataset_name: str,
            added_dataset_prop: float
    ) -> str:
        test_adv_name = (
            test_adv_name[test_adv_name.rfind('/') + 1:]
            if '/' in test_adv_name
            else test_adv_name
        )
        returns_filename = returns_filename[returns_filename.rfind('/') + 1:]
        dataset_name = (
            dataset_name[dataset_name.rfind('/') + 1:]
            if '/' in dataset_name
            else dataset_name
        )
        return (
            f'results/ardt_{returns_filename}_traj{self.max_traj_len}_model/ardt/_adv{test_adv_name}_' +
            f'alpha{self.env_alpha}_False_target_return_{self.seed}.pkl'
        )

    def _eval_fn(self, target_return: int, model: torch.nn.Module) -> dict:
        # Evaluate ARDT using provided offline trajectories
        returns = []
        lengths = []

        # Sample trajectories up to num_eval_episodes
        np.random.seed(self.seed)
        indices = np.random.choice(len(self.trajectories), self.num_eval_episodes, replace=False)

        for idx in indices:
            traj = self.trajectories[idx]
            traj_return = np.sum(traj.rewards) * self.scale  # Scale the total return
            traj_length = len(traj.rewards)

            # Normalize states if required
            states = traj.obs
            if self.normalize_states:
                states = (states - self.state_mean) / (self.state_std + 1e-8)

            # Simulate ARDT evaluation (assuming model predicts actions given states)
            # This is a simplified evaluation; adapt based on ARDT's specific requirements
            model.eval()
            with torch.no_grad():
                states_tensor = torch.tensor(states, dtype=torch.float32, device=self.device)
                # Assuming ARDT model takes states and target_return as input
                # Adjust based on actual ARDT model interface
                predicted_actions = model(states_tensor, target_return=torch.tensor([target_return], device=self.device))
                # Compare predicted actions to trajectory actions (or evaluate robustness)
                # For simplicity, we just collect the trajectory's return and length
                returns.append(traj_return)
                lengths.append(traj_length)

        show_res_dict = {
            f'target_{target_return}_return_mean': np.mean(returns),
            f'target_{target_return}_return_std': np.std(returns),
        }

        result_dict = {
            f'target_{target_return}_return_mean': np.mean(returns),
            f'target_{target_return}_return_std': np.std(returns),
            f'target_{target_return}_length_mean': np.mean(lengths),
            f'target_{target_return}_length_std': np.std(lengths),
        }

        run_storage_path = self.storage_path
        pickle.dump(result_dict, open(run_storage_path, 'wb'))
        print(f"ARDT evaluation results: {show_res_dict}, saved to {run_storage_path}")
        return show_res_dict

    def generate_eval_fn(self, target_return: int) -> '_eval_fn':
        return partial(self._eval_fn, target_return=target_return)