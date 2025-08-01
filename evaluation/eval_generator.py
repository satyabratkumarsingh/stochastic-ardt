import pickle
from pathlib import Path

import gym
import numpy as np
import yaml
from data_class.trajectory import Trajectory
from offline_setup.base_offline_env import BaseOfflineEnv
import torch
import os
from functools import partial
from evaluation.episodes_evaluator import evaluate


class EvalFnGenerator:
    def __init__(
            self, seed: int, env_name: str, task: BaseOfflineEnv, num_eval_episodes: int,
            state_dim: int, act_dim: int, adv_act_dim: int, action_type: str,
            max_traj_len: int, scale: float, state_mean: np.ndarray, state_std: np.ndarray,
            batch_size: int, normalize_states: bool, device: torch.device,
            algo_name: str, returns_filename: str, dataset_name: str,
            test_adv_name: str, added_dataset_name: str, added_dataset_prop: float,
            # NEW PARAMETER: Array of target returns to iterate through
            target_returns_to_evaluate: np.ndarray
        ):
        self.seed = seed
        self.env_name = env_name
        self.task = task
        self.num_eval_episodes = num_eval_episodes
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
        
        # Store the array of target returns for full evaluation
        self.target_returns_to_evaluate = target_returns_to_evaluate

        # Build a template path that can be filled with specific target_return and model_type
        self.storage_path_template = self._build_storage_path_template(
            algo_name, returns_filename, dataset_name, test_adv_name, added_dataset_name, added_dataset_prop
        )
        os.makedirs(os.path.dirname(self.storage_path_template), exist_ok=True)

    def _build_storage_path_template(
            self, algo_name: str, returns_filename: str, dataset_name: str,
            test_adv_name: str, added_dataset_name: str, added_dataset_prop: float
        ) -> str:
        env_instance = self.task
        env_alpha = env_instance.env_alpha if hasattr(env_instance, 'env_alpha') else 0.0
        
        results_dir = 'results'
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)

        test_adv_name = (
            test_adv_name[test_adv_name.rfind('/') + 1:] 
            if '/' in test_adv_name
            else test_adv_name
        )
        
        # Create a path with placeholders for model_type and target_return
        # These placeholders will be replaced in `_eval_fn` for individual files
        # and in `run_full_evaluation` for the summary file.
        base_path = ""
        if algo_name != 'dt':
            returns_filename = returns_filename[returns_filename.rfind('/') + 1:]
            dataset_name = (
                dataset_name[dataset_name.rfind('/') + 1:] 
                if '/' in dataset_name
                else dataset_name
            )
            base_path = (
                f'{results_dir}/{returns_filename}_traj{self.max_traj_len}_MODEL_TYPE_' +
                f'_adv{test_adv_name}_alpha{env_alpha}_False_TARGET_RETURN_{self.seed}.pkl'
            )
        else:
            base_path = (
                f'{results_dir}/{algo_name}_original_{dataset_name}_{added_dataset_name}_' +
                f'{added_dataset_prop}_traj{self.max_traj_len}_MODEL_TYPE_' + 
                f'_adv{test_adv_name}_alpha{env_alpha}_False_TARGET_RETURN_{self.seed}.pkl'
            )
        # Use clear placeholders like 'MODEL_TYPE' and 'TARGET_RETURN' for replacement
        return base_path

    def _eval_fn(self, target_return: float, model: torch.nn.Module, model_type: str) -> dict:
        """
        Evaluates the model for a single target return and saves its results.
        """
        # Call the global `evaluate` function
        returns, lengths = evaluate(
            self.env_name, self.task, self.num_eval_episodes, self.state_dim, 
            self.act_dim, self.adv_act_dim, self.action_type, model, model_type,
            self.max_traj_len, self.scale, self.state_mean, self.state_std,
            target_return, batch_size=self.batch_size, normalize_states=self.normalize_states,
            device=self.device
        )
        
        # Calculate statistics
        return_mean = np.mean(returns)
        return_std = np.std(returns)
        length_mean = np.mean(lengths)
        length_std = np.std(lengths)

        show_res_dict = {
            f'target_{target_return}_return_mean': return_mean,
            f'target_{target_return}_return_std': return_std,
        }

        result_dict = {
            f'target_{target_return}_return_mean': return_mean,
            f'target_{target_return}_return_std': return_std,
            f'target_{target_return}_length_mean': length_mean,
            f'target_{target_return}_length_std': length_std,
        }

        # Replace placeholders in the template for the specific file name
        run_storage_path = self.storage_path_template.replace('MODEL_TYPE', model_type) \
                                                     .replace('TARGET_RETURN', str(target_return))
        
        # Ensure directory exists for the specific file
        os.makedirs(os.path.dirname(run_storage_path), exist_ok=True)
        
        with open(run_storage_path, 'wb') as f:
            pickle.dump(result_dict, f)
            
        print(f"Evaluation results for target {target_return}: {show_res_dict} saved to {run_storage_path}")
        return show_res_dict

    # The `generate_eval_fn` method is no longer needed with the new `run_full_evaluation`
    # def generate_eval_fn(self, target_return: float) -> '_eval_fn':
    #     return partial(self._eval_fn, target_return=target_return)

    # NEW METHOD: Orchestrates evaluation across all specified target returns
    def run_full_evaluation(self, model: torch.nn.Module, model_type: str) -> dict:
        """
        Runs the full evaluation process across all target returns specified during initialization.
        Collects and prints aggregated results, and saves a summary file.
        """
        overall_targets = []
        overall_return_means = []
        overall_return_stds = []
        overall_length_means = []
        overall_length_stds = [] # To store length statistics too if needed for summary

        print(f"\n--- Starting full evaluation for {model_type} across multiple targets ---")

        for target in self.target_returns_to_evaluate:
            # Call the existing _eval_fn for each target
            current_target_results = self._eval_fn(target, model, model_type)
            
            # Extract and store the results for the overall summary
            overall_targets.append(target)
            overall_return_means.append(current_target_results[f'target_{target}_return_mean'])
            overall_return_stds.append(current_target_results[f'target_{target}_return_std'])

        print("\n--- Full Evaluation Summary ---")
        print("Targets:", np.array(overall_targets))
        print("Return Means:", np.array(overall_return_means))
        print("Return Stds:", np.array(overall_return_stds))
        
        # Compile all results into a single dictionary for the overall summary
        overall_results_summary = {
            'targets': np.array(overall_targets),
            'return_means': np.array(overall_return_means),
            'return_stds': np.array(overall_return_stds),
        }
        
        # Generate a summary filename
        summary_filename = self.storage_path_template.replace('MODEL_TYPE', model_type) \
                                                     .replace(f'_TARGET_RETURN_{self.seed}.pkl', f'_summary_{self.seed}.pkl')
        
        # Ensure summary directory exists
        os.makedirs(os.path.dirname(summary_filename), exist_ok=True)
        
        with open(summary_filename, 'wb') as f:
            pickle.dump(overall_results_summary, f)
        print(f"Overall evaluation summary saved to: {summary_filename}")

        return overall_results_summary