

from data_class.trajectory import Trajectory
import torch 
import pickle
from pathlib import Path
import gym
import numpy as np
import yaml
from tqdm import tqdm
import os
import torch.nn.functional as F
from core_models.decision_transformer.decision_transformer import DecisionTransformer
from core_models.dataset.ardt_dataset import ARDTDataset
from offline_setup.base_offline_env import BaseOfflineEnv
from utils.trajectory_utils import get_relabeled_trajectories, get_action_dim
from typing import Dict, List, Optional, Tuple, Any
from utils.saved_names import dt_model_name, behaviour_cloning_model_name
from core_models.behaviour_cloning.behaviour_cloning import MLPBCModel
from evaluation.model_loader import ModelLoader
from evaluation.stochastic_ardt_evaluator import ARDTEvaluator, ARDTValidator
#from evaluation.ardt_evaluator_fixed import test_model_architecture

class ModelEvaluator:
    """
    A class to orchestrate the entire model evaluation pipeline, including
    loading datasets, auto-detecting parameters, and running validation tests.
    """

    def __init__(self, seed: int, game_name: str, config_path: str, device: str = 'cpu', dt_model = None):
        self.seed = seed
        self.game_name = game_name
        self.config_path = config_path
        self.device = device
        self.config_file_args = yaml.safe_load(Path(config_path).read_text())
        self.dt_model = dt_model

    def auto_detect_dataset_params(self, trajectories: List[Any]) -> Dict:
        """
        Automatically detect dataset parameters from trajectories.
        
        Args:
            trajectories (List[Any]): A list of trajectory objects (or dictionaries).
        
        Returns:
            Dict: A dictionary containing the auto-detected parameters.
        """
        print("🔍 Auto-detecting dataset parameters...")
        
        if not trajectories:
            raise ValueError("Empty trajectory list provided")
        
        first_traj = trajectories[0]
        
        # Helper function to safely access attributes or dictionary keys
        def get_attr_or_item(obj, key):
            if isinstance(obj, dict):
                return obj.get(key)
            return getattr(obj, key, None)

        # Extract basic dimensions
        first_obs = get_attr_or_item(first_traj, 'obs')
        first_actions = get_attr_or_item(first_traj, 'actions')
        if first_obs is None or first_actions is None:
            raise ValueError("Trajectory object must have 'obs' and 'actions'")

        state_dim = len(first_obs[0])
        act_dim = get_action_dim(first_actions)
        
        # Detect action type
        sample_action = first_actions[0]
        if isinstance(sample_action, (list, np.ndarray)) and len(sample_action) > 1:
            if all(x in [0.0, 1.0] for x in sample_action) and sum(sample_action) == 1.0:
                action_type = 'discrete'
            else:
                action_type = 'continuous'
        else:
            action_type = 'discrete' if isinstance(sample_action, (int, np.integer)) else 'continuous'
        
        # Calculate max trajectory length
        max_traj_len = max([len(get_attr_or_item(traj, 'obs')) for traj in trajectories])
        
        # Collect all states for statistics
        all_states = np.concatenate([get_attr_or_item(traj, 'obs') for traj in trajectories])
        state_mean = np.mean(all_states, axis=0)
        state_std = np.std(all_states, axis=0)
        state_std = np.where(state_std == 0, 1.0, state_std)
        
        # Calculate scale from rewards
        all_rewards = np.concatenate([get_attr_or_item(traj, 'rewards') for traj in trajectories])
        reward_range = np.max(all_rewards) - np.min(all_rewards)
        scale = max(reward_range, 1.0)
        scale = 1
        # Auto-detect adversarial action dimension
        adv_act_dim = act_dim
        for traj in trajectories:
            adv_actions = get_attr_or_item(traj, 'adv_actions')
            if adv_actions is not None and len(adv_actions) > 0:
                adv_act_dim = get_action_dim(adv_actions)
                break
        
        # Generate reasonable target returns
        all_episode_returns = [sum(get_attr_or_item(traj, 'rewards')) for traj in trajectories]
        min_return = np.min(all_episode_returns)
        max_return = np.max(all_episode_returns)
        padding = (max_return - min_return) * 0.2
        target_returns = np.linspace(min_return - padding, max_return + padding, num=5)
        target_returns = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]

        params = {
            'state_dim': state_dim,
            'act_dim': act_dim,
            'adv_act_dim': adv_act_dim,
            'action_type': action_type,
            'max_ep_len': max_traj_len,
            'scale': scale,
            'state_mean': state_mean,
            'state_std': state_std,
            'target_returns': target_returns,
            'dataset_stats': {
                'num_trajectories': len(trajectories),
                'min_return': min_return,
                'max_return': max_return,
                'mean_return': np.mean(all_episode_returns),
                'std_return': np.std(all_episode_returns)
            }
        }
        
        print(f"✅ Auto-detected parameters:")
        print(f"   State dim: {state_dim}")
        print(f"   Action dim: {act_dim}")
        print(f"   Adversarial action dim: {adv_act_dim}")
        print(f"   Action type: {action_type}")
        print(f"   Max episode length: {max_traj_len}")
        print(f"   Scale: {scale:.2f}")
        print(f"   Target returns: {target_returns}")
        
        return params

    def run_validation_pipeline(
        self,
        env_instance,
        method: str,
        dt_model: torch.nn.Module,
        baseline_model: Optional[torch.nn.Module],
        dataset_params: Dict,
        num_episodes: int = 500
    ) -> Dict:
        if dt_model is None:
            print("❌ Error: No ARDT model provided for validation.")
            return {}

        # Create the evaluator with auto-detected parameters
        evaluator = ARDTEvaluator(
            env_name=env_instance.name if hasattr(env_instance, 'name') else self.game_name,
            env_instance=env_instance,
            state_dim=dataset_params['state_dim'],
            act_dim=dataset_params['act_dim'],
            action_type=dataset_params['action_type'],
            max_ep_len=dataset_params['max_ep_len'],
            scale=dataset_params['scale'],
            state_mean=dataset_params['state_mean'],
            state_std=dataset_params['state_std'],
            adv_act_dim=dataset_params['adv_act_dim'],
            normalize_states=True,
            device=self.device
        )
        plot_file_name = f"Runs/plot_returns_{method}"
        evaluator.comprehensive_dual_evaluation(
           model=dt_model,
           target_returns=[-2.0, -1.0, 0.0, 1.0, 2.0],
           num_episodes_per_target=num_episodes,
           save_path= plot_file_name
        )

        # evaluator.analyze_action_distribution(dt_model, model_type='dt', num_episodes=100)
        # evaluator.analyze_action_distribution(baseline_model, model_type='bc', num_episodes=100)

        # #Create separate evaluation functions for each model type
        # dt_eval_fn = evaluator.create_eval_function(model_type='dt', worst_case=False)
        # bc_eval_fn = evaluator.create_eval_function(model_type='bc', worst_case=False)
        
        # # Create the validator
        # validator = ARDTValidator(device=self.device)
        
        # # Run the validation suite and return results
        # return validator.run_validation(
        #     ardt_model=dt_model,
        #     dt_eval_fn=dt_eval_fn,
        #     baseline_model=baseline_model,
        #     bc_eval_fn=bc_eval_fn,
        #     target_returns=dataset_params['target_returns'],
        #     num_episodes=num_episodes
        # )

    def evaluate_models(self, env_instance, method: str, num_episodes: int = 500):
        """
        Main orchestration method to run the evaluation.
        """
        # 1. Load trajectories
        trajectories, _ = get_relabeled_trajectories(self.seed, self.game_name, method=method)
        
        # 2. Auto-detect parameters from trajectories
        dataset_params = self.auto_detect_dataset_params(trajectories=trajectories)
        
        # 3. Load models
        model_loader = ModelLoader(self.seed, self.game_name, self.config_path, self.device)
        bc_model, dt_model = model_loader.load_models(method=method)
        if self.dt_model:
            dt_model = self.dt_model
        if not dt_model:
            print("❌ ARDT model could not be loaded. Exiting.")
            return {}
        
        # 4. Run the validation pipeline with the loaded models and detected parameters
        print("✅ Both DT and BC models loaded. Running full validation suite...")
        return self.run_validation_pipeline(
            env_instance = env_instance,
            method = method,
            dt_model=dt_model,
            baseline_model=bc_model,
            dataset_params=dataset_params,
            num_episodes=num_episodes
        )