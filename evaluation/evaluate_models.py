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
from evaluation.stochastic_ardt_evaluator import ARDTModelLoader

#TARGET_RETURNS = [-2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0]
TARGET_RETURNS = [-0.04, -0.02, 0.0, 0.02, 0.05, 0.10, 0.5]
#TARGET_RETURNS = [-1.5, -1.0,  -0.5, 0.0, 0.5, 1.0, 1.5 ]
NUM_EPISODES = 1000
SCALE = 1.0
STATE_MEAN = None
STATE_STD = None

def load_and_evaluate_model(
    seed: int,
    game_name: str,
    method: str,
    config_path: str,
    env_instance,
    device: str = 'cpu'
) -> Dict:
    """Load and evaluate the minimax Decision Transformer model"""
    print("Starting Model Loading and Evaluation Pipeline...")
    print("=" * 70)
    
    # Load minimax model only
    model_loader = ARDTModelLoader(seed, game_name, config_path, device)
    
    try:
        model, model_params = model_loader.load_model(method, "minimax")
        if model is None:
            raise ValueError("Failed to load minimax model")
    except Exception as e:
        print(f"Failed to load minimax model: {e}")
        raise ValueError(f"Model loading failed: {e}")
    
    print(f"Successfully loaded minimax model with parameters:")
    print(f"  - Observation size: {model_params['obs_size']}")
    print(f"  - Action size: {model_params['action_size']}")
    print(f"  - Action type: {model_params['action_type']}")
    print(f"  - Horizon: {model_params['horizon']}")
    
    # Create evaluator using model parameters
    evaluator = ARDTEvaluator.from_model_params(
        env_name=game_name,
        env_instance=env_instance,
        model_params=model_params,
        scale=SCALE,
        state_mean=STATE_MEAN,
        state_std=STATE_STD,
        device=device
    )
    
    # Run comprehensive evaluation
    print("\nRunning comprehensive evaluation...")
    evaluation_results = evaluator.comprehensive_model_evaluation(
        minimax_model=model,
        method=method,
        target_returns=TARGET_RETURNS,
        num_episodes_per_target=NUM_EPISODES,
        save_path=f"evaluation_results_{game_name}_{method}"
    )


def get_model_info(seed: int, game_name: str, method: str, config_path: str) -> Dict:
    """Get model information without loading the full model"""
    model_loader = ARDTModelLoader(seed, game_name, config_path, 'cpu')
    
    try:
        model_info = model_loader.get_model_info(method, "minimax")
        return {
            'model_type': 'minimax',
            'info': model_info,
            'available': True
        }
    except Exception as e:
        print(f"Failed to get model info: {e}")
        return {
            'model_type': 'minimax', 
            'info': None,
            'available': False,
            'error': str(e)
        }

def quick_model_test(
    seed: int,
    game_name: str, 
    method: str,
    config_path: str,
    device: str = 'cpu'
) -> bool:
    """Quick test to verify model can be loaded successfully"""
    try:
        model_loader = ARDTModelLoader(seed, game_name, config_path, device)
        model, params = model_loader.load_model(method, "minimax")
        
        if model is None:
            print("Model loading returned None")
            return False
            
        print(f"Model test successful:")
        print(f"  - Model type: {type(model).__name__}")
        print(f"  - Parameters: {params}")
        
        # Quick forward pass test
        batch_size = 1
        seq_len = params.get('horizon', 5)
        obs_size = params['obs_size']
        action_size = params['action_size']
        
        # Create dummy inputs
        dummy_obs = torch.zeros(batch_size, seq_len, obs_size)
        dummy_actions = torch.zeros(batch_size, seq_len, action_size)
        dummy_rewards = torch.zeros(batch_size, seq_len, 1)
        dummy_rtg = torch.zeros(batch_size, seq_len, 1)
        dummy_timesteps = torch.arange(seq_len).unsqueeze(0)
        dummy_mask = torch.ones(batch_size, seq_len).bool()
        
        # Test forward pass
        with torch.no_grad():
            model.eval()
            state_preds, action_preds, return_preds = model(
                states=dummy_obs,
                actions=dummy_actions, 
                rewards=dummy_rewards,
                returns_to_go=dummy_rtg,
                timesteps=dummy_timesteps,
                attention_mask=dummy_mask
            )
            
        print(f"  - Forward pass successful")
        print(f"  - Action predictions shape: {action_preds.shape}")
        
        return True
        
    except Exception as e:
        print(f"Model test failed: {e}")
        return False

def evaluate_at_target_return(
    seed: int,
    game_name: str,
    method: str, 
    config_path: str,
    env_instance,
    target_return: float,
    num_episodes: int = 100,
    device: str = 'cpu'
) -> Dict:
    """Evaluate model performance at a specific target return"""
    
    # Load model
    model_loader = ARDTModelLoader(seed, game_name, config_path, device)
    model, model_params = model_loader.load_model(method, "minimax")
    
    if model is None:
        raise ValueError("Failed to load model for evaluation")
    
    # Create evaluator
    evaluator = ARDTEvaluator.from_model_params(
        env_name=game_name,
        env_instance=env_instance,
        model_params=model_params,
        scale=SCALE,
        state_mean=STATE_MEAN,
        state_std=STATE_STD,
        device=device
    )
    
    # Run evaluation at specific return
    print(f"Evaluating at target return: {target_return}")
    results = evaluator.evaluate_at_return_level(
        model=model,
        target_return=target_return,
        num_episodes=num_episodes
    )
    
    return results

# Backwards compatibility function (deprecated)
def load_and_evaluate_models(*args, **kwargs):
    """
    Deprecated: Use load_and_evaluate_model() instead
    This function is kept for backwards compatibility
    """
    print("Warning: load_and_evaluate_models() is deprecated. Use load_and_evaluate_model() instead.")
    return load_and_evaluate_model(*args, **kwargs)