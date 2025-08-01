
import numpy as np
import torch
from typing import Dict, List, Tuple
import pickle
import os
from pathlib import Path
from return_transforms.eval_function import EvalFnGenerator

class ARDTValidationSuite:
    """
    Integration with your existing evaluation framework to add ARDT-specific validation
    """
    
    def __init__(self, eval_fn_generator):
        self.eval_fn_generator = eval_fn_generator
        self.validation_results = {}
    
    def validate_minimax_returns_prediction(self, minimax_model, test_trajectories: List[Dict]) -> Dict:
        """
        Validate that the minimax returns prediction model is working correctly
        """
        print("🔍 Validating Minimax Returns Prediction...")
        
        prediction_errors = []
        trajectory_errors = []
        
        for traj in test_trajectories:
            # Extract trajectory data
            states = np.array(traj['obs'])
            actions = np.array(traj['actions'])
            rewards = np.array(traj['rewards'])
            true_minimax_rtg = np.array(traj['minimax_returns_to_go'])
            
            # Predict minimax returns using your model
            with torch.no_grad():
                states_tensor = torch.FloatTensor(states)
                actions_tensor = torch.FloatTensor(actions)
                
                # This would depend on your minimax model interface
                predicted_minimax_rtg = minimax_model.predict(states_tensor, actions_tensor)
                predicted_minimax_rtg = predicted_minimax_rtg.cpu().numpy()
            
            # Calculate errors
            traj_error = np.mean(np.abs(predicted_minimax_rtg - true_minimax_rtg))
            trajectory_errors.append(traj_error)
            prediction_errors.extend(np.abs(predicted_minimax_rtg - true_minimax_rtg))
        
        results = {
            'mean_absolute_error': np.mean(prediction_errors),
            'std_error': np.std(prediction_errors),
            'max_error': np.max(prediction_errors),
            'trajectory_errors': trajectory_errors,
            'prediction_quality': 'good' if np.mean(prediction_errors) < 0.1 else 'poor'
        }
        
        print(f"   MAE: {results['mean_absolute_error']:.4f}")
        print(f"   Max Error: {results['max_error']:.4f}")
        print(f"   Quality: {results['prediction_quality']}")
        
        return results
    
    def test_return_conditioning_accuracy(self, model, model_type: str) -> Dict:
        """
        Test if the model achieves different returns when conditioned on different targets
        """
        print("🎯 Testing Return Conditioning Accuracy...")
        
        # Test with minimax returns from your dataset
        test_targets = [-2.0, -1.0, 0.0, 1.0, 2.0]  # Range based on your Kuhn Poker rewards
        conditioning_results = {}
        
        for target in test_targets:
            print(f"   Testing target return: {target:.2f}")
            
            # Use your existing evaluation function
            returns, lengths = self.eval_fn_generator.evaluate(
                env_name=self.eval_fn_generator.env_name,
                task=self.eval_fn_generator.task,
                num_eval_episodes=50,  # Smaller number for testing
                state_dim=self.eval_fn_generator.state_dim,
                act_dim=self.eval_fn_generator.act_dim,
                adv_act_dim=self.eval_fn_generator.adv_act_dim,
                action_type=self.eval_fn_generator.action_type,
                model=model,
                model_type=model_type,
                max_ep_len=self.eval_fn_generator.max_traj_len,
                scale=self.eval_fn_generator.scale,
                state_mean=self.eval_fn_generator.state_mean,
                state_std=self.eval_fn_generator.state_std,
                target_return=target,
                batch_size=self.eval_fn_generator.batch_size,
                normalize_states=self.eval_fn_generator.normalize_states,
                device=self.eval_fn_generator.device
            )
            
            mean_return = np.mean(returns)
            target_error = abs(mean_return - target)
            achievement_rate = np.mean([abs(r - target) < 0.2 for r in returns])
            
            conditioning_results[target] = {
                'achieved_return': mean_return,
                'target_error': target_error,
                'achievement_rate': achievement_rate,
                'return_std': np.std(returns),
                'all_returns': returns
            }
            
            print(f"     Achieved: {mean_return:.3f}, Error: {target_error:.3f}, Success Rate: {achievement_rate:.3f}")
        
        return conditioning_results
    
    def test_worst_case_robustness(self, model, model_type: str, num_episodes: int = 200) -> Dict:
        """
        Test robustness against progressively stronger adversaries
        """
        print("🛡️ Testing Worst-Case Robustness...")
        
        robustness_results = {}
        
        # Test against different adversary strengths
        # In your case, you have a deterministic worst-case adversary, 
        # but we can test with different target returns to see robustness
        
        adversary_configs = {
            'conservative_target': -0.5,  # Conservative play
            'balanced_target': 0.0,       # Balanced play
            'aggressive_target': 0.5      # Aggressive play
        }
        
        for config_name, target_return in adversary_configs.items():
            print(f"   Testing against {config_name} (target: {target_return})...")
            
            returns, lengths = self.eval_fn_generator.evaluate(
                env_name=self.eval_fn_generator.env_name,
                task=self.eval_fn_generator.task,
                num_eval_episodes=num_episodes,
                state_dim=self.eval_fn_generator.state_dim,
                act_dim=self.eval_fn_generator.act_dim,
                adv_act_dim=self.eval_fn_generator.adv_act_dim,
                action_type=self.eval_fn_generator.action_type,
                model=model,
                model_type=model_type,
                max_ep_len=self.eval_fn_generator.max_traj_len,
                scale=self.eval_fn_generator.scale,
                state_mean=self.eval_fn_generator.state_mean,
                state_std=self.eval_fn_generator.state_std,
                target_return=target_return,
                batch_size=self.eval_fn_generator.batch_size,
                normalize_states=self.eval_fn_generator.normalize_states,
                device=self.eval_fn_generator.device
            )
            
            mean_return = np.mean(returns)
            worst_case_return = np.min(returns)
            robustness_score = np.percentile(returns, 10)  # 10th percentile as robustness metric
            
            robustness_results[config_name] = {
                'mean_return': mean_return,
                'worst_case_return': worst_case_return,
                'robustness_score': robustness_score,
                'return_std': np.std(returns),
                'all_returns': returns
            }
            
            print(f"     Mean: {mean_return:.3f}, Worst: {worst_case_return:.3f}, 10th percentile: {robustness_score:.3f}")
        
        # Calculate overall robustness metric
        overall_robustness = np.mean([result['robustness_score'] for result in robustness_results.values()])
        
        robustness_results['overall_robustness'] = overall_robustness
        print(f"   Overall Robustness Score: {overall_robustness:.3f}")
        
        return robustness_results
    
    def compare_with_baseline_dt(self, ardt_model, baseline_dt_model, num_episodes: int = 100) -> Dict:
        """
        Compare ARDT model with baseline Decision Transformer
        """
        print("📊 Comparing ARDT vs Baseline DT...")
        
        comparison_results = {}
        test_targets = [-1.0, 0.0, 1.0]  # Representative targets
        
        for target in test_targets:
            print(f"   Testing target return: {target:.2f}")
            
            # Test ARDT
            ardt_returns, _ = self.eval_fn_generator.evaluate(
                env_name=self.eval_fn_generator.env_name,
                task=self.eval_fn_generator.task,
                num_eval_episodes=num_episodes,
                state_dim=self.eval_fn_generator.state_dim,
                act_dim=self.eval_fn_generator.act_dim,
                adv_act_dim=self.eval_fn_generator.adv_act_dim,
                action_type=self.eval_fn_generator.action_type,
                model=ardt_model,
                model_type='dt',
                max_ep_len=self.eval_fn_generator.max_traj_len,
                scale=self.eval_fn_generator.scale,
                state_mean=self.eval_fn_generator.state_mean,
                state_std=self.eval_fn_generator.state_std,
                target_return=target,
                batch_size=self.eval_fn_generator.batch_size,
                normalize_states=self.eval_fn_generator.normalize_states,
                device=self.eval_fn_generator.device
            )
            
            # Test Baseline DT
            dt_returns, _ = self.eval_fn_generator.evaluate(
                env_name=self.eval_fn_generator.env_name,
                task=self.eval_fn_generator.task,
                num_eval_episodes=num_episodes,
                state_dim=self.eval_fn_generator.state_dim,
                act_dim=self.eval_fn_generator.act_dim,
                adv_act_dim=self.eval_fn_generator.adv_act_dim,
                action_type=self.eval_fn_generator.action_type,
                model=baseline_dt_model,
                model_type='dt',
                max_ep_len=self.eval_fn_generator.max_traj_len,
                scale=self.eval_fn_generator.scale,
                state_mean=self.eval_fn_generator.state_mean,
                state_std=self.eval_fn_generator.state_std,
                target_return=target,
                batch_size=self.eval_fn_generator.batch_size,
                normalize_states=self.eval_fn_generator.normalize_states,
                device=self.eval_fn_generator.device
            )
            
            # Calculate comparison metrics
            ardt_mean = np.mean(ardt_returns)
            dt_mean = np.mean(dt_returns)
            ardt_worst = np.min(ardt_returns)
            dt_worst = np.min(dt_returns)
            
            comparison_results[target] = {
                'ardt_mean': ardt_mean,
                'dt_mean': dt_mean,
                'ardt_worst': ardt_worst,
                'dt_worst': dt_worst,
                'mean_improvement': ardt_mean - dt_mean,
                'worst_case_improvement': ardt_worst - dt_worst,
                'ardt_returns': ardt_returns,
                'dt_returns': dt_returns
            }
            
            print(f"     ARDT Mean: {ardt_mean:.3f}, DT Mean: {dt_mean:.3f}")
            print(f"     ARDT Worst: {ardt_worst:.3f}, DT Worst: {dt_worst:.3f}")
            print(f"     Improvement (Mean): {ardt_mean - dt_mean:.3f}")
            print(f"     Improvement (Worst): {ardt_worst - dt_worst:.3f}")
        
        # Calculate overall improvement
        overall_mean_improvement = np.mean([result['mean_improvement'] for result in comparison_results.values()])
        overall_worst_improvement = np.mean([result['worst_case_improvement'] for result in comparison_results.values()])
        
        comparison_results['overall_metrics'] = {
            'mean_improvement': overall_mean_improvement,
            'worst_case_improvement': overall_worst_improvement,
            'better_mean_performance': overall_mean_improvement > 0,
            'better_worst_case': overall_worst_improvement > 0
        }
        
        print(f"   Overall Mean Improvement: {overall_mean_improvement:.3f}")
        print(f"   Overall Worst-Case Improvement: {overall_worst_improvement:.3f}")
        
        return comparison_results
    
    def validate_trajectory_relabeling(self, original_trajectories: List[Dict], 
                                     relabeled_trajectories: List[Dict]) -> Dict:
        """
        Validate that trajectory relabeling with minimax returns is consistent
        """
        print("🔄 Validating Trajectory Relabeling...")
        
        relabeling_results = {
            'original_return_stats': {},
            'relabeled_return_stats': {},
            'consistency_metrics': {}
        }
        
        # Extract original and relabeled returns-to-go
        original_rtgs = []
        relabeled_rtgs = []
        
        for orig_traj, relabeled_traj in zip(original_trajectories, relabeled_trajectories):
            # Assuming original trajectories have 'returns_to_go' and relabeled have 'minimax_returns_to_go'
            if 'returns_to_go' in orig_traj:
                original_rtgs.extend(orig_traj['returns_to_go'])
            if 'minimax_returns_to_go' in relabeled_traj:
                relabeled_rtgs.extend(relabeled_traj['minimax_returns_to_go'])
        
        original_rtgs = np.array(original_rtgs)
        relabeled_rtgs = np.array(relabeled_rtgs)
        
        # Calculate statistics
        relabeling_results['original_return_stats'] = {
            'mean': np.mean(original_rtgs),
            'std': np.std(original_rtgs),
            'min': np.min(original_rtgs),
            'max': np.max(original_rtgs)
        }
        
        relabeling_results['relabeled_return_stats'] = {
            'mean': np.mean(relabeled_rtgs),
            'std': np.std(relabeled_rtgs),
            'min': np.min(relabeled_rtgs),
            'max': np.max(relabeled_rtgs)
        }
        
        # Consistency metrics
        if len(original_rtgs) == len(relabeled_rtgs):
            correlation = np.corrcoef(original_rtgs, relabeled_rtgs)[0, 1]
            mean_difference = np.mean(relabeled_rtgs - original_rtgs)
            
            relabeling_results['consistency_metrics'] = {
                'correlation': correlation,
                'mean_difference': mean_difference,
                'relabeling_conservative': mean_difference < 0,  # Minimax should generally be more conservative
            }
            
            print(f"   Correlation: {correlation:.3f}")  
            print(f"   Mean Difference: {mean_difference:.3f}")
            print(f"   Conservative Relabeling: {mean_difference < 0}")
        
        print(f"   Original RTG - Mean: {relabeling_results['original_return_stats']['mean']:.3f}")
        print(f"   Relabeled RTG - Mean: {relabeling_results['relabeled_return_stats']['mean']:.3f}")
        
        return relabeling_results
    
    def run_ardt_validation_suite(self, ardt_model, baseline_dt_model=None, 
                                 minimax_model=None, test_trajectories=None) -> Dict:
        """
        Run complete ARDT validation suite
        """
        print("🚀 Running Complete ARDT Validation Suite\n")
        
        all_results = {}
        
        # 1. Minimax Returns Prediction Validation
        if minimax_model and test_trajectories:
            all_results['minimax_prediction'] = self.validate_minimax_returns_prediction(
                minimax_model, test_trajectories
            )
        
        # 2. Return Conditioning Test
        all_results['return_conditioning'] = self.test_return_conditioning_accuracy(
            ardt_model, 'dt'
        )
        
        # 3. Worst-Case Robustness Test
        all_results['robustness'] = self.test_worst_case_robustness(
            ardt_model, 'dt'
        )
        
        # 4. Baseline Comparison
        if baseline_dt_model:
            all_results['baseline_comparison'] = self.compare_with_baseline_dt(
                ardt_model, baseline_dt_model
            )
        
        # 5. Generate validation report
        self._generate_validation_report(all_results)
        
        # 6. Save results
        self._save_validation_results(all_results)
        
        return all_results
    
    def _generate_validation_report(self, results: Dict):
        """Generate comprehensive validation report"""
        print("\n" + "="*80)
        print("🎯 ARDT VALIDATION REPORT")
        print("="*80)
        
        # Return Conditioning Results
        if 'return_conditioning' in results:
            print("\n📊 RETURN CONDITIONING VALIDATION:")
            conditioning = results['return_conditioning']
            avg_achievement = np.mean([r['achievement_rate'] for r in conditioning.values()])
            avg_error = np.mean([r['target_error'] for r in conditioning.values()])
            
            print(f"   Average Achievement Rate: {avg_achievement:.3f}")
            print(f"   Average Target Error: {avg_error:.3f}")
            print(f"   Conditioning Quality: {'✅ GOOD' if avg_achievement > 0.7 else '❌ POOR'}")
        
        # Robustness Results  
        if 'robustness' in results:
            print("\n🛡️ ROBUSTNESS VALIDATION:")
            robustness = results['robustness']
            overall_robustness = robustness.get('overall_robustness', 0)
            
            print(f"   Overall Robustness Score: {overall_robustness:.3f}")
            print(f"   Robustness Quality: {'✅ GOOD' if overall_robustness > -0.5 else '❌ POOR'}")
        
        # Baseline Comparison Results
        if 'baseline_comparison' in results:
            print("\n📈 BASELINE COMPARISON:")
            comparison = results['baseline_comparison']
            overall_metrics = comparison.get('overall_metrics', {})
            
            mean_improvement = overall_metrics.get('mean_improvement', 0)
            worst_improvement = overall_metrics.get('worst_case_improvement', 0)
            
            print(f"   Mean Performance Improvement: {mean_improvement:.3f}")
            print(f"   Worst-Case Improvement: {worst_improvement:.3f}")
            print(f"   Better than Baseline: {'✅ YES' if mean_improvement > 0 else '❌ NO'}")
            print(f"   More Robust: {'✅ YES' if worst_improvement > 0 else '❌ NO'}")
        
        # Overall Assessment
        print("\n🏆 OVERALL ARDT VALIDATION:")
        
        passes = 0
        total_tests = 0
        
        if 'return_conditioning' in results:
            total_tests += 1
            if np.mean([r['achievement_rate'] for r in results['return_conditioning'].values()]) > 0.7:
                passes += 1
                print("   Return Conditioning: ✅ PASS")
            else:
                print("   Return Conditioning: ❌ FAIL")
        
        if 'robustness' in results:
            total_tests += 1
            if results['robustness'].get('overall_robustness', -999) > -0.5:
                passes += 1
                print("   Robustness: ✅ PASS")
            else:
                print("   Robustness: ❌ FAIL")
        
        if 'baseline_comparison' in results:
            total_tests += 1
            overall_metrics = results['baseline_comparison'].get('overall_metrics', {})
            if overall_metrics.get('worst_case_improvement', -999) > 0:
                passes += 1
                print("   Baseline Improvement: ✅ PASS")
            else:
                print("   Baseline Improvement: ❌ FAIL")
        
        print(f"\n   Validation Score: {passes}/{total_tests}")
        
        if passes >= total_tests * 0.8:
            print("   🎉 ARDT MODEL VALIDATION SUCCESSFUL!")
        elif passes >= total_tests * 0.6:
            print("   ⚠️ ARDT model shows promise but needs improvement")
        else:
            print("   ❌ ARDT model validation failed - significant issues detected")
        
        print("="*80)
    
    def _save_validation_results(self, results: Dict):
        """Save validation results to file"""
        results_dir = Path('validation_results')
        results_dir.mkdir(exist_ok=True)
        
        filename = results_dir / f'ardt_validation_{self.eval_fn_generator.seed}.pkl'
        
        with open(filename, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"\n💾 Validation results saved to: {filename}")



class EnhancedEvalFnGenerator(EvalFnGenerator):
    """
    Enhanced version of your EvalFnGenerator with ARDT validation capabilities
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.validation_suite = ARDTValidationSuite(self)
    
    def run_full_ardt_evaluation(self, ardt_model: torch.nn.Module, 
                                baseline_dt_model: torch.nn.Module = None,
                                minimax_model: torch.nn.Module = None,
                                test_trajectories: List[Dict] = None) -> Dict:
        """
        Run both standard evaluation and ARDT-specific validation
        """
        print("🚀 Running Full ARDT Evaluation (Standard + Validation)\n")
        
        # 1. Run standard evaluation across target returns
        standard_results = self.run_full_evaluation(ardt_model, 'dt')
        
        # 2. Run ARDT-specific validation
        validation_results = self.validation_suite.run_ardt_validation_suite(
            ardt_model=ardt_model,
            baseline_dt_model=baseline_dt_model,
            minimax_model=minimax_model,
            test_trajectories=test_trajectories
        )
        
        # 3. Combine results
        combined_results = {
            'standard_evaluation': standard_results,
            'ardt_validation': validation_results
        }
        
        # 4. Save combined results
        combined_filename = self.storage_path_template.replace('MODEL_TYPE', 'dt') \
                                                     .replace(f'_TARGET_RETURN_{self.seed}.pkl', 
                                                            f'_full_ardt_evaluation_{self.seed}.pkl')
        
        os.makedirs(os.path.dirname(combined_filename), exist_ok=True)
        with open(combined_filename, 'wb') as f:
            pickle.dump(combined_results, f)
        
        print(f"\n💾 Full ARDT evaluation saved to: {combined_filename}")
        
        return combined_results