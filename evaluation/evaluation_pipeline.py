
import torch
import yaml
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import os
import pickle
from tqdm import tqdm

from core_models.decision_transformer.decision_transformer import DecisionTransformer
from core_models.dataset.ardt_dataset import ARDTDataset
from training.decision_tranformer_trainer import train_both_models, get_action_dim
from utils.trajectory_utils import get_relabeled_trajectories
from evaluation.new_simplified_evaluator import run_ardt_validation_pipeline

class IntegratedARDTPipeline:
    """
    Integrated pipeline for training Decision Transformers and running comprehensive evaluations
    """
    
    def __init__(self, 
                 ret_file: str,
                 run_implicit: bool,
                 config_path: str,
                 device: str = 'cuda',
                 n_cpu: int = 4,
                 seed: int = 42):
        
        self.config_path = config_path
        self.device = device
        self.n_cpu = n_cpu
        self.seed = seed
        self.ret_file = ret_file
        # Load configuration
        self.config = yaml.safe_load(Path(config_path).read_text())
        self.dt_train_args = self.config['train_args']
        
        # Initialize models storage
        self.baseline_dt_model = None
        self.ardt_model = None
        self.minimax_model = None
        
        # Initialize evaluators
        self.evaluator = None
        self.enhanced_evaluator = None
        relabeled_trajs, prompt_value = get_relabeled_trajectories(ret_file, run_implicit=run_implicit)

        self.relabeled_trajs = relabeled_trajs
        self.prompt_value = prompt_value
        self.run_implicit = run_implicit
        print(f"🚀 Initialized ARDT Pipeline with device: {device}")

    def setup_evaluators_from_trajectories(self, 
                                          trajectories: List,
                                          env_name: str,
                                          task,
                                          num_eval_episodes: int = 100,
                                          batch_size: int = 1,
                                          normalize_states: bool = True,
                                          algo_name: str = 'ardt',
                                          returns_filename: str = 'minimax_returns',
                                          dataset_name: str = 'kuhn_poker',
                                          test_adv_name: str = 'worst_case',
                                          added_dataset_name: str = 'none',
                                          added_dataset_prop: float = 0.0):
        """
        Setup both standard and enhanced evaluators using auto-detected parameters from trajectories
        """
        # Auto-detect parameters from trajectories
        auto_params = self.auto_detect_dataset_params(trajectories)
        
        # Setup standard evaluator
        self.evaluator = EvalFnGenerator(
            seed=self.seed,
            env_name=env_name,
            task=task,
            num_eval_episodes=num_eval_episodes,
            state_dim=auto_params['state_dim'],
            act_dim=auto_params['act_dim'],
            adv_act_dim=auto_params['adv_act_dim'],
            action_type=auto_params['action_type'],
            max_traj_len=auto_params['max_traj_len'],
            scale=auto_params['scale'],
            state_mean=auto_params['state_mean'],
            state_std=auto_params['state_std'],
            batch_size=batch_size,
            normalize_states=normalize_states,
            device=torch.device(self.device),
            algo_name=algo_name,
            returns_filename=returns_filename,
            dataset_name=dataset_name,
            test_adv_name=test_adv_name,
            added_dataset_name=added_dataset_name,
            added_dataset_prop=added_dataset_prop,
            target_returns_to_evaluate=auto_params['target_returns_to_evaluate']
        )
        
        # Setup enhanced evaluator
        self.enhanced_evaluator = EnhancedEvalFnGenerator(
            seed=self.seed,
            env_name=env_name,
            task=task,
            num_eval_episodes=num_eval_episodes,
            state_dim=auto_params['state_dim'],
            act_dim=auto_params['act_dim'],
            adv_act_dim=auto_params['adv_act_dim'],
            action_type=auto_params['action_type'],
            max_traj_len=auto_params['max_traj_len'],
            scale=auto_params['scale'],
            state_mean=auto_params['state_mean'],
            state_std=auto_params['state_std'],
            batch_size=batch_size,
            normalize_states=normalize_states,
            device=torch.device(self.device),
            algo_name=algo_name,
            returns_filename=returns_filename,
            dataset_name=dataset_name,
            test_adv_name=test_adv_name,
            added_dataset_name=added_dataset_name,
            added_dataset_prop=added_dataset_prop,
            target_returns_to_evaluate=auto_params['target_returns_to_evaluate']
        )
        
        print("✅ Evaluators setup complete with auto-detected parameters")
        return auto_params
    
    def auto_detect_dataset_params(self, trajectories: List) -> Dict:
        """
        Automatically detect dataset parameters from trajectories
        """
        print("🔍 Auto-detecting dataset parameters...")
        
        if not trajectories:
            raise ValueError("Empty trajectory list provided")
        
        first_traj = trajectories[0]
        
        # Extract basic dimensions
        state_dim = len(first_traj.obs[0]) if hasattr(first_traj, 'obs') else len(first_traj['obs'][0])
        act_dim = get_action_dim(first_traj.actions if hasattr(first_traj, 'actions') else first_traj['actions'])
        
        # Detect action type
        sample_action = (first_traj.actions[0] if hasattr(first_traj, 'actions') 
                        else first_traj['actions'][0])
        
        if isinstance(sample_action, (list, np.ndarray)) and len(sample_action) > 1:
            if all(x in [0.0, 1.0] for x in sample_action) and sum(sample_action) == 1.0:
                action_type = 'discrete'
            else:
                action_type = 'continuous'
        else:
            action_type = 'discrete' if isinstance(sample_action, (int, np.integer)) else 'continuous'
        
        # Calculate max trajectory length
        max_traj_len = max([
            len(traj.obs if hasattr(traj, 'obs') else traj['obs']) 
            for traj in trajectories
        ])
        
        # Collect all states for statistics
        all_states = []
        for traj in trajectories:
            traj_obs = traj.obs if hasattr(traj, 'obs') else traj['obs']
            all_states.extend(traj_obs)
        
        all_states = np.array(all_states)
        state_mean = np.mean(all_states, axis=0)
        state_std = np.std(all_states, axis=0)
        # Avoid division by zero
        state_std = np.where(state_std == 0, 1.0, state_std)
        
        # Calculate scale from rewards
        all_rewards = []
        for traj in trajectories:
            traj_rewards = traj.rewards if hasattr(traj, 'rewards') else traj['rewards']
            all_rewards.extend(traj_rewards)
        
        reward_range = np.max(all_rewards) - np.min(all_rewards)
        scale = max(reward_range, 1.0)  # Ensure scale is at least 1.0
        
        # Auto-detect adversarial action dimension
        adv_act_dim = act_dim  # Default assumption
        
        # Try to detect from adversarial actions if available
        for traj in trajectories:
            if hasattr(traj, 'adv_actions'):
                adv_actions = traj.adv_actions
            elif 'adv_actions' in (traj if isinstance(traj, dict) else {}):
                adv_actions = traj['adv_actions']
            else:
                continue
                
            if adv_actions is not None and len(adv_actions) > 0:
                adv_act_dim = get_action_dim(adv_actions)
                break
        
        # Generate reasonable target returns based on actual returns
        all_episode_returns = []
        for traj in trajectories:
            traj_rewards = traj.rewards if hasattr(traj, 'rewards') else traj['rewards']
            episode_return = sum(traj_rewards)
            all_episode_returns.append(episode_return)
        
        min_return = np.min(all_episode_returns)
        max_return = np.max(all_episode_returns)
        mean_return = np.mean(all_episode_returns)
        
        # Create target returns spanning the range with some padding
        padding = (max_return - min_return) * 0.2
        target_returns = np.linspace(
            min_return - padding, 
            max_return + padding, 
            num=5
        )
        
        params = {
            'state_dim': state_dim,
            'act_dim': act_dim,
            'adv_act_dim': adv_act_dim,
            'action_type': action_type,
            'max_traj_len': max_traj_len,
            'scale': scale,
            'state_mean': state_mean,
            'state_std': state_std,
            'target_returns_to_evaluate': target_returns,
            'dataset_stats': {
                'num_trajectories': len(trajectories),
                'min_return': min_return,
                'max_return': max_return,
                'mean_return': mean_return,
                'std_return': np.std(all_episode_returns)
            }
        }
        
        print(f"✅ Auto-detected parameters:")
        print(f"   State dim: {state_dim}")
        print(f"   Action dim: {act_dim}")
        print(f"   Adversarial action dim: {adv_act_dim}")
        print(f"   Action type: {action_type}")
        print(f"   Max trajectory length: {max_traj_len}")
        print(f"   Scale: {scale:.2f}")
        print(f"   Return range: [{min_return:.2f}, {max_return:.2f}]")
        print(f"   Target returns: {target_returns}")
        
        return params
    
    def setup_evaluators(self, 
                        env_name: str,
                        task,
                        num_eval_episodes: int,
                        state_dim: int,
                        act_dim: int,
                        adv_act_dim: int,
                        action_type: str,
                        max_traj_len: int,
                        scale: float,
                        state_mean: np.ndarray,
                        state_std: np.ndarray,
                        batch_size: int = 1,
                        normalize_states: bool = True,
                        algo_name: str = 'ardt',
                        returns_filename: str = 'minimax_returns',
                        dataset_name: str = 'kuhn_poker',
                        test_adv_name: str = 'worst_case',
                        added_dataset_name: str = 'none',
                        added_dataset_prop: float = 0.0,
                        target_returns_to_evaluate: Optional[np.ndarray] = None):
        """
        Setup both standard and enhanced evaluators (manual parameter specification)
        """
        if target_returns_to_evaluate is None:
            target_returns_to_evaluate = np.array([-2.0, -1.0, 0.0, 1.0, 2.0])
        
        # Setup standard evaluator
        self.evaluator = EvalFnGenerator(
            seed=self.seed,
            env_name=env_name,
            task=task,
            num_eval_episodes=num_eval_episodes,
            state_dim=state_dim,
            act_dim=act_dim,
            adv_act_dim=adv_act_dim,
            action_type=action_type,
            max_traj_len=max_traj_len,
            scale=scale,
            state_mean=state_mean,
            state_std=state_std,
            batch_size=batch_size,
            normalize_states=normalize_states,
            device=torch.device(self.device),
            algo_name=algo_name,
            returns_filename=returns_filename,
            dataset_name=dataset_name,
            test_adv_name=test_adv_name,
            added_dataset_name=added_dataset_name,
            added_dataset_prop=added_dataset_prop,
            target_returns_to_evaluate=target_returns_to_evaluate
        )
        
        # Setup enhanced evaluator
        self.enhanced_evaluator = EnhancedEvalFnGenerator(
            seed=self.seed,
            env_name=env_name,
            task=task,
            num_eval_episodes=num_eval_episodes,
            state_dim=state_dim,
            act_dim=act_dim,
            adv_act_dim=adv_act_dim,
            action_type=action_type,
            max_traj_len=max_traj_len,
            scale=scale,
            state_mean=state_mean,
            state_std=state_std,
            batch_size=batch_size,
            normalize_states=normalize_states,
            device=torch.device(self.device),
            algo_name=algo_name,
            returns_filename=returns_filename,
            dataset_name=dataset_name,
            test_adv_name=test_adv_name,
            added_dataset_name=added_dataset_name,
            added_dataset_prop=added_dataset_prop,
            target_returns_to_evaluate=target_returns_to_evaluate
        )
        
        print("✅ Evaluators setup complete")

  
    def run_comprehensive_evaluation(self, 
                                   test_trajectories: Optional[List] = None,
                                   run_standard_eval: bool = True,
                                   run_enhanced_eval: bool = True,
                                   skip_minimax_validation: bool = True) -> Dict:
        """
        Run comprehensive evaluation using both evaluators
        """
        if self.evaluator is None or self.enhanced_evaluator is None:
            raise ValueError("Evaluators not setup. Call setup_evaluators() first.")
        
        if self.ardt_model is None:
            raise ValueError("ARDT model not trained. Call train_both_models() first.")
        
        print("\n" + "="*80)
        print("🎯 RUNNING COMPREHENSIVE ARDT EVALUATION")  
        print("="*80)
        
        all_results = {}
        
        # 1. Standard Evaluation
        if run_standard_eval:
            print("\n📊 Running Standard Evaluation...")
            try:
                standard_results = self.evaluator.run_full_evaluation(self.ardt_model, 'dt')
                all_results['standard_evaluation'] = standard_results
                print("✅ Standard evaluation complete")
            except Exception as e:
                print(f"❌ Standard evaluation failed: {e}")
                all_results['standard_evaluation'] = {'error': str(e)}
        
        # 2. Enhanced Evaluation with Validation Suite
        if run_enhanced_eval:
            print("\n🔬 Running Enhanced Evaluation with ARDT Validation...")
            try:
                # Only pass minimax_model and test_trajectories if not skipping minimax validation
                minimax_model_param = None if skip_minimax_validation else self.minimax_model
                test_trajectories_param = None if skip_minimax_validation else test_trajectories
                
                enhanced_results = self.enhanced_evaluator.run_full_ardt_evaluation(
                    ardt_model=self.ardt_model,
                    baseline_dt_model=self.baseline_dt_model,
                    minimax_model=minimax_model_param,
                    test_trajectories=test_trajectories_param
                )
                all_results['enhanced_evaluation'] = enhanced_results
                print("✅ Enhanced evaluation complete")
            except Exception as e:
                print(f"❌ Enhanced evaluation failed: {e}")
                all_results['enhanced_evaluation'] = {'error': str(e)}
        
        # 3. Save comprehensive results
        self._save_comprehensive_results(all_results)
        
        return all_results

    def _save_comprehensive_results(self, results: Dict):
        """Save comprehensive evaluation results"""
        results_dir = Path('comprehensive_results')
        results_dir.mkdir(exist_ok=True)
        
        filename = results_dir / f'comprehensive_ardt_evaluation_{self.seed}.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n💾 Comprehensive results saved to: {filename}")

    def run_full_pipeline(self,
                         env_name: str,
                         task,
                         num_eval_episodes: int = 100,
                         test_trajectories: Optional[List] = None,
                         skip_minimax_validation: bool = True,
                         **evaluator_kwargs) -> Dict:
        """
        Run the complete pipeline: training + evaluation with auto-detected parameters
        """
        print("\n" + "="*80)
        print("🚀 STARTING COMPLETE ARDT PIPELINE WITH AUTO-DETECTION")
        print("="*80)
        
        # Step 1: Train models (this loads trajectories)
        print("\n🏋️ Step 1: Training models...")
        self.minimax_model, self.ardt_model = train_both_models(self.dt_train_args, relabeled_trajs= self.relabeled_trajs, run_implicit=self.run_implicit)
    
        all_states = []
        all_rewards = []
        for traj in self.relabeled_trajs:
            traj_obs = traj.obs if hasattr(traj, 'obs') else traj['obs']
            episode_reward = np.sum(traj.rewards)
            all_states.extend(traj_obs)
            all_rewards.append(episode_reward)
        
        all_states = np.array(all_states)
        state_mean = np.mean(all_states, axis=0)
        state_std = np.std(all_states, axis=0)
        # Avoid division by zero
        state_std = np.where(state_std == 0, 1.0, state_std)
         # Setup scaling
        scale = self.config.get('scale', 1.0)
        if np.std(all_rewards) > 0:
            scale = max(1.0, np.std(all_rewards) / 1.0)
        print(f"========= Using scale: {scale:.4f}")


        results = run_ardt_validation_pipeline(
                     ardt_model=self.ardt_model,
                    env_name='kuhn_poker',
                    env_instance=task,
                    state_dim=12,
                    act_dim=2,
                    action_type='discrete',
                     max_ep_len=10,
                    scale=scale,
                    state_mean=state_mean,
                    state_std=state_std,
                     baseline_model= self.minimax_model,
                     num_episodes=100
                )

        # # Step 2: Setup evaluators using auto-detected parameters from trajectories
        # print("\n📋 Step 2: Setting up evaluators with auto-detected parameters...")
        # auto_params = self.setup_evaluators_from_trajectories(
        #     trajectories=self.relabeled_trajs,
        #     env_name=env_name,
        #     task=task,
        #     num_eval_episodes=num_eval_episodes
        # )
        
        # # Step 3: Run comprehensive evaluation
        # print("\n🔍 Step 3: Running comprehensive evaluation...")
        # results = self.run_comprehensive_evaluation(
        #     test_trajectories=test_trajectories,
        #     skip_minimax_validation=skip_minimax_validation
        # )
        
        # # Add auto-detected parameters to results
        # results['auto_detected_params'] = auto_params
        
        # print("\n" + "="*80)
        # print("🎉 COMPLETE ARDT PIPELINE FINISHED SUCCESSFULLY")
        # print("="*80)
        
        return results
