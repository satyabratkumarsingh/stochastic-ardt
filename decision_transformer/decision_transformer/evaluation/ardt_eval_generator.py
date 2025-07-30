"""
Integration script showing how to use ARDT with your existing EvalFnGenerator framework.
"""

import numpy as np
import torch
import gym
from typing import List, Dict, Any
import pickle

# Import your existing modules
from functools import partial
from collections import namedtuple


class ARDTEvalFnGenerator:
    """
    Extended EvalFnGenerator specifically for ARDT models.
    This integrates seamlessly with your existing evaluation framework.
    """
    
    def __init__(
        self,
        seed: int,
        env_name: str,
        task: 'BaseOfflineEnv',
        num_eval_episodes: int,
        state_dim: int,
        act_dim: int,
        adv_act_dim: int,
        action_type: str,
        max_traj_len: int,
        scale: float,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        batch_size: int,
        normalize_states: bool,
        device: torch.device,
        ardt_prompt_value: float,  # NEW: The computed minimax return
        dataset_name: str = "kuhn_poker_ardt",
        test_adv_name: str = "minimax_adversary"
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
        self.ardt_prompt_value = ardt_prompt_value
        
        # Build storage path for ARDT results
        self.storage_path = self._build_ardt_storage_path(dataset_name, test_adv_name)
    
    def _build_ardt_storage_path(self, dataset_name: str, test_adv_name: str) -> str:
        """Build storage path for ARDT evaluation results."""
        env = self.task.test_env_cls()
        env_alpha = env.env_alpha if hasattr(env, 'env_alpha') else 0.0
        
        return (
            f'results/ardt_{dataset_name}_traj{self.max_traj_len}_' +
            f'adv{test_adv_name}_alpha{env_alpha}_seed{self.seed}_target_return_.pkl'
        )
    
    def _eval_fn(
        self, 
        target_return: float, 
        model: torch.nn.Module, 
        model_type: str = 'dt'
    ) -> Dict[str, float]:
        """
        Evaluation function compatible with your existing evaluate() function.
        
        Args:
            target_return: The target return for conditioning (minimax return from ARDT)
            model: The trained ARDT model (ARDTModelWrapper)
            model_type: Should be 'dt' for Decision Transformer
        """
        
        # Import your evaluate function
        from decision_transformer.decision_transformer.evaluation.evaluate_episodes import evaluate
        
        # Run evaluation using your existing framework
        returns, lengths = evaluate(
            self.env_name,
            self.task,
            self.num_eval_episodes,
            self.state_dim,
            self.act_dim,
            self.adv_act_dim,
            self.action_type,
            model,  # This is the ARDTModelWrapper
            model_type,  # 'dt'
            self.max_traj_len,
            self.scale,
            self.state_mean,
            self.state_std,
            target_return,  # This is the key - use minimax return for conditioning!
            batch_size=self.batch_size,
            normalize_states=self.normalize_states,
            device=self.device
        )
        
        # Process results
        mean_return = np.mean(returns)
        std_return = np.std(returns)
        mean_length = np.mean(lengths)
        std_length = np.std(lengths)
        
        # Create result dictionaries
        show_res_dict = {
            f'ardt_target_{target_return:.1f}_return_mean': mean_return,
            f'ardt_target_{target_return:.1f}_return_std': std_return,
        }
        
        result_dict = {
            f'ardt_target_{target_return:.1f}_return_mean': mean_return,
            f'ardt_target_{target_return:.1f}_return_std': std_return,
            f'ardt_target_{target_return:.1f}_length_mean': mean_length,
            f'ardt_target_{target_return:.1f}_length_std': std_length,
            'ardt_prompt_value': self.ardt_prompt_value,
            'target_return_used': target_return,
            'raw_returns': returns,
            'raw_lengths': lengths
        }
        
        # Save results
        run_storage_path = self.storage_path.replace('_target_return_', f'_{target_return:.1f}_')
        pickle.dump(result_dict, open(run_storage_path, 'wb'))
        
        print(f"ARDT Evaluation results: {show_res_dict}")
        print(f"Results saved to: {run_storage_path}")
        
        return show_res_dict
    
    def generate_eval_fn(self, target_return: float = None):
        """
        Generate evaluation function for ARDT model.
        
        Args:
            target_return: Target return for conditioning. If None, uses the computed prompt value.
        """
        if target_return is None:
            target_return = self.ardt_prompt_value
        
        return partial(self._eval_fn, target_return=target_return)


def complete_ardt_pipeline(
    kuhn_poker_episodes: List[Dict[str, Any]],
    task: 'BaseOfflineEnv',
    train_args: Dict[str, Any],
    evaluation_args: Dict[str, Any],
    device: str = 'cpu'
) -> Dict[str, Any]:
    """
    Complete ARDT training and evaluation pipeline that integrates with your framework.
    
    Args:
        kuhn_poker_episodes: Your Kuhn Poker episode data
        task: Your BaseOfflineEnv task object
        train_args: Training configuration
        evaluation_args: Evaluation configuration 
        device: Device to run on
    
    Returns:
        Dictionary with all evaluation results
    """
    
    print("=" * 60)
    print("COMPLETE ARDT PIPELINE")
    print("=" * 60)
    
    # Step 1: Train ARDT model
    print("\n1. Training ARDT model...")
    
    action_space = gym.spaces.Discrete(2)  # Pass/Bet for Kuhn Poker
    adv_action_space = gym.spaces.Discrete(2)
    
    # Import and run ARDT training
    from ardt_kuhn_poker_fixed import maxmin_kuhn_poker
    
    learned_returns, prompt_value, trained_model = maxmin_kuhn_poker(
        kuhn_poker_episodes=kuhn_poker_episodes,
        action_space=action_space,
        adv_action_space=adv_action_space,
        train_args=train_args,
        device=device,
        is_simple_model=train_args.get('is_simple_model', False)
    )
    
    print(f"✓ ARDT training complete