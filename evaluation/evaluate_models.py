

import os
import json
import yaml
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from pathlib import Path
from typing import Tuple, List, Optional, Callable, Dict, Any
from evaluation.stochastic_ardt_evaluator import ARDTEvaluator
from evaluation.model_loader import ModelLoader
from evaluation.stochastic_ardt_evaluator import ARDTValidator, ARDTModelLoader

TARGET_RETURNS = [-2.0, -1.5, -1.0,-0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
NUM_EPISODES = 1000
SCALE = 1.0
STATE_MEAN = None
STATE_STD = None

def load_and_evaluate_models(
    seed: int,
    game_name: str,
    method: str,
    config_path: str,
    env_instance,
    device: str = 'cpu'
) -> Dict:
    print("🚀 Starting Model Loading and Evaluation Pipeline...")
    print("=" * 70)
    
    # Load models
    model_loader = ARDTModelLoader(seed, game_name, config_path, device)
    models = model_loader.load_both_models(method)
    
    if models['minimax'] is None or models['original'] is None:
        raise ValueError("Failed to load one or both models. Please check model paths.")
    
    minimax_model, minimax_params = models['minimax']
    original_model, original_params = models['original']
    
    # Verify both models have compatible parameters
    if minimax_params['obs_size'] != original_params['obs_size']:
        raise ValueError("Model parameter mismatch: different observation sizes")
    if minimax_params['action_size'] != original_params['action_size']:
        raise ValueError("Model parameter mismatch: different action sizes")
    
    # Create evaluator using model parameters
    evaluator = ARDTEvaluator.from_model_params(
        env_name=game_name,
        env_instance=env_instance,
        model_params=minimax_params,  # Both models should have same dimensions
        scale=SCALE,
        state_mean=STATE_MEAN,
        state_std=STATE_STD,
        device=device
    )
    
    # Run comprehensive evaluation
    evaluation_results = evaluator.comprehensive_model_evaluation(
        minimax_model=minimax_model,
        method=method,
        target_returns=TARGET_RETURNS,
        num_episodes_per_target=NUM_EPISODES,
        save_path=f"evaluation_results_{game_name}_{method}"
    )
    
    # Run validation
    validator = ARDTValidator(device)
    
    eval_fn_normal = evaluator.create_eval_function(case_type="normal")
    eval_fn_worst = evaluator.create_eval_function(case_type="worst") 
    
    validation_results = validator.run_comprehensive_validation(
        minimax_model=minimax_model,
        original_model=original_model,
        eval_fn_normal=eval_fn_normal,
        eval_fn_worst=eval_fn_worst,
        target_returns=TARGET_RETURNS or [-2.0, -1.0, 0.0, 1.0, 2.0],
        num_episodes=100
    )
    
    # Combine results
    combined_results = {
        'evaluation': evaluation_results,
        'validation': validation_results,
        'model_info': {
            'minimax_params': minimax_params,
            'original_params': original_params,
            'game_name': game_name,
            'method': method,
            'seed': seed
        }
    }
    
    print("\n🎉 Pipeline Complete! Check the generated plots and saved results.")
    
    return combined_results


def quick_model_comparison(
    seed: int,
    game_name: str,
    method: str,
    config_path: str,
    env_instance,
    device: str = 'cpu'
) -> None:
    """
    Quick comparison function for a single target return.
    Model parameters are automatically extracted from saved checkpoints.
    """
    
    print(f"🔍 Quick Model Comparison (Target Return: {TARGET_RETURNS})")
    print("=" * 50)
    
    # Load models
    model_loader = ARDTModelLoader(seed, game_name, config_path, device)
    models = model_loader.load_both_models(method)
    
    if models['minimax'] is None or models['original'] is None:
        raise ValueError("Failed to load one or both models.")
    
    minimax_model, minimax_params = models['minimax']
    original_model, original_params = models['original']
    
    # Create evaluator using model parameters
    evaluator = ARDTEvaluator.from_model_params(
        env_name=game_name,
        env_instance=env_instance,
        model_params=minimax_params,
        scale=SCALE,
        state_mean=STATE_MEAN,
        state_std=STATE_STD,
        device=device
    )
    
    # Quick evaluation
    eval_fn_normal = evaluator.create_eval_function(worst_case=False)
    eval_fn_worst = evaluator.create_eval_function(worst_case=True)
    
    print("\n--- Normal Case ---")
    minimax_returns_normal, _ = eval_fn_normal(minimax_model, TARGET_RETURNS, NUM_EPISODES)
    original_returns_normal, _ = eval_fn_normal(original_model, TARGET_RETURNS, NUM_EPISODES)
    
    print(f"Minimax Model: {np.mean(minimax_returns_normal):.3f} ± {np.std(minimax_returns_normal):.3f}")
    print(f"Original Model: {np.mean(original_returns_normal):.3f} ± {np.std(original_returns_normal):.3f}")
    print(f"Improvement: {np.mean(minimax_returns_normal) - np.mean(original_returns_normal):.3f}")
    
    print("\n--- Worst Case ---")
    minimax_returns_worst, _ = eval_fn_worst(minimax_model, TARGET_RETURNS, NUM_EPISODES)
    original_returns_worst, _ = eval_fn_worst(original_model, TARGET_RETURNS, NUM_EPISODES)
    
    print(f"Minimax Model: {np.mean(minimax_returns_worst):.3f} ± {np.std(minimax_returns_worst):.3f}")
    print(f"Original Model: {np.mean(original_returns_worst):.3f} ± {np.std(original_returns_worst):.3f}")
    print(f"Improvement: {np.mean(minimax_returns_worst) - np.mean(original_returns_worst):.3f}")
    
    print("=" * 50)


def get_model_info(seed: int, game_name: str, method: str, config_path: str) -> Dict:
    """
    Get model information without loading the full models
    """
    model_loader = ARDTModelLoader(seed, game_name, config_path, 'cpu')
    
    try:
        minimax_info = model_loader.get_model_info(method, "minimax")
        original_info = model_loader.get_model_info(method, "original")
        
        return {
            'minimax': minimax_info,
            'original': original_info,
            'compatible': (
                minimax_info['obs_size'] == original_info['obs_size'] and
                minimax_info['action_size'] == original_info['action_size']
            )
        }
    except Exception as e:
        print(f"❌ Failed to get model info: {e}")
        return None