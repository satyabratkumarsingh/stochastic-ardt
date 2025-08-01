
import numpy as np
import torch
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
from pathlib import Path
import json

class SimplifiedKuhnPokerValidator:
    """
    Simplified validation suite for Kuhn Poker Decision Transformer with minimax returns
    """
    
    def __init__(self, device='cuda'):
        self.device = device
        self.results = {}
        
    def validate_return_conditioning(self, model, eval_function, target_returns=None, num_episodes=50):
        """
        Test if the model achieves different returns when conditioned on different targets
        """
        print("🎯 Testing Return Conditioning...")
        
        if target_returns is None:
            # Based on your Kuhn Poker data, reasonable targets
            target_returns = [-2.0, -1.0, 0.0, 1.0, 2.0]
        
        conditioning_results = {}
        
        for target in target_returns:
            print(f"   Testing target return: {target:.2f}")
            
            # Run evaluation with this target return
            returns, lengths = eval_function(
                model=model,
                target_return=target,
                num_episodes=num_episodes
            )
            
            achieved_return = np.mean(returns)
            target_error = abs(achieved_return - target)
            success_rate = np.mean([abs(r - target) < 0.3 for r in returns])  # Within 0.3 of target
            
            conditioning_results[target] = {
                'achieved_return': achieved_return,
                'target_error': target_error,
                'success_rate': success_rate,
                'return_std': np.std(returns),
                'all_returns': returns.tolist()
            }
            
            print(f"     → Achieved: {achieved_return:.3f}, Error: {target_error:.3f}, Success: {success_rate:.3f}")
        
        # Overall conditioning quality
        avg_error = np.mean([r['target_error'] for r in conditioning_results.values()])
        avg_success = np.mean([r['success_rate'] for r in conditioning_results.values()])
        
        conditioning_results['summary'] = {
            'avg_target_error': avg_error,
            'avg_success_rate': avg_success,
            'conditioning_quality': 'GOOD' if avg_success > 0.6 else 'POOR'
        }
        
        print(f"   📊 Overall: Avg Error = {avg_error:.3f}, Avg Success = {avg_success:.3f}")
        print(f"   📊 Quality: {conditioning_results['summary']['conditioning_quality']}")
        
        return conditioning_results
    
    def compare_with_baseline(self, ardt_model, baseline_model, eval_function, 
                            target_returns=None, num_episodes=100):
        """
        Compare ARDT model with baseline Decision Transformer
        """
        print("📊 Comparing ARDT vs Baseline...")
        
        if target_returns is None:
            target_returns = [-1.0, 0.0, 1.0]  # Representative targets
        
        comparison_results = {}
        
        for target in target_returns:
            print(f"   Testing target return: {target:.2f}")
            
            # Test ARDT model
            ardt_returns, _ = eval_function(
                model=ardt_model,
                target_return=target,
                num_episodes=num_episodes
            )
            
            # Test baseline model
            baseline_returns, _ = eval_function(
                model=baseline_model,
                target_return=target,
                num_episodes=num_episodes
            )
            
            # Calculate comparison metrics
            ardt_mean = np.mean(ardt_returns)
            baseline_mean = np.mean(baseline_returns)
            ardt_worst = np.min(ardt_returns)
            baseline_worst = np.min(baseline_returns)
            ardt_best = np.max(ardt_returns)
            baseline_best = np.max(baseline_returns)
            
            comparison_results[target] = {
                'ardt_mean': ardt_mean,
                'baseline_mean': baseline_mean,
                'ardt_worst': ardt_worst,
                'baseline_worst': baseline_worst,
                'ardt_best': ardt_best,
                'baseline_best': baseline_best,
                'mean_improvement': ardt_mean - baseline_mean,
                'worst_case_improvement': ardt_worst - baseline_worst,
                'ardt_returns': ardt_returns.tolist(),
                'baseline_returns': baseline_returns.tolist()
            }
            
            print(f"     ARDT:     Mean={ardt_mean:.3f}, Worst={ardt_worst:.3f}, Best={ardt_best:.3f}")
            print(f"     Baseline: Mean={baseline_mean:.3f}, Worst={baseline_worst:.3f}, Best={baseline_best:.3f}")
            print(f"     Improvement: Mean={ardt_mean - baseline_mean:.3f}, Worst={ardt_worst - baseline_worst:.3f}")
        
        # Overall comparison
        overall_mean_improvement = np.mean([r['mean_improvement'] for r in comparison_results.values()])
        overall_worst_improvement = np.mean([r['worst_case_improvement'] for r in comparison_results.values()])
        
        comparison_results['summary'] = {
            'overall_mean_improvement': overall_mean_improvement,
            'overall_worst_improvement': overall_worst_improvement,
            'better_mean_performance': overall_mean_improvement > 0,
            'better_worst_case': overall_worst_improvement > 0,
            'significant_improvement': overall_mean_improvement > 0.1  # At least 0.1 improvement
        }
        
        print(f"   📈 Overall Mean Improvement: {overall_mean_improvement:.3f}")
        print(f"   📈 Overall Worst-Case Improvement: {overall_worst_improvement:.3f}")
        print(f"   📈 Better Performance: {comparison_results['summary']['better_mean_performance']}")
        
        return comparison_results
    
    def test_minimax_consistency(self, model, eval_function, expected_minimax_value=None, num_episodes=200):
        """
        Test if the model's performance is consistent with minimax expectations
        """
        print("🛡️ Testing Minimax Consistency...")
        
        # Test with minimax-optimal target (should be around the game value)
        if expected_minimax_value is None:
            expected_minimax_value = 0.0  # Kuhn Poker game value is typically around 0
            
        returns, lengths = eval_function(
            model=model,
            target_return=expected_minimax_value,
            num_episodes=num_episodes
        )
        
        mean_return = np.mean(returns)
        worst_return = np.min(returns)
        robustness_score = np.percentile(returns, 10)  # 10th percentile
        consistency_score = 1.0 - np.std(returns) / (np.abs(mean_return) + 1e-6)  # Lower std is better
        
        minimax_results = {
            'expected_minimax_value': expected_minimax_value,
            'achieved_mean': mean_return,
            'worst_case': worst_return,
            'robustness_score': robustness_score,
            'consistency_score': consistency_score,
            'return_std': np.std(returns),
            'all_returns': returns.tolist(),
            'is_robust': robustness_score > expected_minimax_value - 0.5,  # Within reasonable bound
            'is_consistent': np.std(returns) < 0.5  # Reasonable consistency
        }
        
        print(f"   Expected minimax: {expected_minimax_value:.3f}")
        print(f"   Achieved mean: {mean_return:.3f}")
        print(f"   Worst case: {worst_return:.3f}")
        print(f"   Robustness (10th percentile): {robustness_score:.3f}")
        print(f"   Consistency (low std is good): {np.std(returns):.3f}")
        print(f"   Is Robust: {minimax_results['is_robust']}")
        print(f"   Is Consistent: {minimax_results['is_consistent']}")
        
        return minimax_results
    
    def validate_trajectory_relabeling(self, original_trajectories, relabeled_trajectories):
        """
        Validate that minimax return relabeling is reasonable
        """
        print("🔄 Validating Trajectory Relabeling...")
        
        original_episode_returns = []
        minimax_episode_returns = []
        
        for orig_traj, relabeled_traj in zip(original_trajectories, relabeled_trajectories):
            # Calculate original episode return
            orig_return = sum(orig_traj['rewards'])
            original_episode_returns.append(orig_return)
            
            # Get minimax return (first minimax_returns_to_go value)
            minimax_return = relabeled_traj['minimax_returns_to_go'][0]
            minimax_episode_returns.append(minimax_return)
        
        original_episode_returns = np.array(original_episode_returns)
        minimax_episode_returns = np.array(minimax_episode_returns)
        
        # Calculate statistics
        relabeling_results = {
            'original_stats': {
                'mean': np.mean(original_episode_returns),
                'std': np.std(original_episode_returns),
                'min': np.min(original_episode_returns),
                'max': np.max(original_episode_returns)
            },
            'minimax_stats': {
                'mean': np.mean(minimax_episode_returns),
                'std': np.std(minimax_episode_returns),
                'min': np.min(minimax_episode_returns),
                'max': np.max(minimax_episode_returns)
            },
            'correlation': np.corrcoef(original_episode_returns, minimax_episode_returns)[0, 1],
            'mean_difference': np.mean(minimax_episode_returns - original_episode_returns),
            'reasonable_relabeling': True  # Will be updated based on checks
        }
        
        print(f"   Original returns: Mean={relabeling_results['original_stats']['mean']:.3f}, "
              f"Range=[{relabeling_results['original_stats']['min']:.2f}, {relabeling_results['original_stats']['max']:.2f}]")
        print(f"   Minimax returns:  Mean={relabeling_results['minimax_stats']['mean']:.3f}, "
              f"Range=[{relabeling_results['minimax_stats']['min']:.2f}, {relabeling_results['minimax_stats']['max']:.2f}]")
        print(f"   Correlation: {relabeling_results['correlation']:.3f}")
        print(f"   Mean difference (minimax - original): {relabeling_results['mean_difference']:.3f}")
        
        # Check if relabeling is reasonable
        correlation_ok = relabeling_results['correlation'] > 0.3  # Some positive correlation expected
        range_reasonable = (relabeling_results['minimax_stats']['max'] - 
                          relabeling_results['minimax_stats']['min']) > 0.5  # Some spread
        
        relabeling_results['reasonable_relabeling'] = correlation_ok and range_reasonable
        print(f"   Relabeling Quality: {'GOOD' if relabeling_results['reasonable_relabeling'] else 'QUESTIONABLE'}")
        
        return relabeling_results
    
    def run_complete_validation(self, ardt_model, baseline_model, eval_function,
                              original_trajectories=None, relabeled_trajectories=None,
                              target_returns=None, num_episodes=100):
        """
        Run complete validation suite
        """
        print("\n" + "="*60)
        print("🚀 KUHN POKER ARDT VALIDATION SUITE")
        print("="*60)
        
        results = {}
        
        # 1. Return Conditioning Test
        print("\n1️⃣ RETURN CONDITIONING TEST")
        print("-" * 40)
        results['return_conditioning'] = self.validate_return_conditioning(
            ardt_model, eval_function, target_returns, num_episodes
        )
        
        # 2. Baseline Comparison
        if baseline_model is not None:
            print("\n2️⃣ BASELINE COMPARISON")
            print("-" * 40)
            results['baseline_comparison'] = self.compare_with_baseline(
                ardt_model, baseline_model, eval_function, target_returns, num_episodes
            )
        
        # 3. Minimax Consistency Test
        print("\n3️⃣ MINIMAX CONSISTENCY TEST")
        print("-" * 40)
        results['minimax_consistency'] = self.test_minimax_consistency(
            ardt_model, eval_function, num_episodes=num_episodes*2
        )
        
        # 4. Trajectory Relabeling Validation
        if original_trajectories and relabeled_trajectories:
            print("\n4️⃣ TRAJECTORY RELABELING VALIDATION")
            print("-" * 40)
            results['relabeling_validation'] = self.validate_trajectory_relabeling(
                original_trajectories, relabeled_trajectories
            )
        
        # 5. Generate Summary Report
        print("\n5️⃣ VALIDATION SUMMARY")
        print("-" * 40)
        summary = self._generate_summary_report(results)
        results['summary'] = summary
        
        # 6. Save results
        self._save_results(results)
        
        return results
    
    def _generate_summary_report(self, results):
        """Generate validation summary"""
        summary = {
            'tests_passed': 0,
            'total_tests': 0,
            'overall_score': 0.0,
            'recommendations': []
        }
        
        # Check return conditioning
        if 'return_conditioning' in results:
            summary['total_tests'] += 1
            conditioning_quality = results['return_conditioning']['summary']['conditioning_quality']
            if conditioning_quality == 'GOOD':
                summary['tests_passed'] += 1
                print("   ✅ Return Conditioning: PASS")
            else:
                print("   ❌ Return Conditioning: FAIL")
                summary['recommendations'].append("Improve return conditioning - model not achieving target returns reliably")
        
        # Check baseline comparison
        if 'baseline_comparison' in results:
            summary['total_tests'] += 1
            better_performance = results['baseline_comparison']['summary']['better_mean_performance']
            if better_performance:
                summary['tests_passed'] += 1
                print("   ✅ Baseline Comparison: PASS (Better performance)")
            else:
                print("   ❌ Baseline Comparison: FAIL (Worse than baseline)")
                summary['recommendations'].append("Model performs worse than baseline - check training data and hyperparameters")
        
        # Check minimax consistency
        if 'minimax_consistency' in results:
            summary['total_tests'] += 1
            is_robust = results['minimax_consistency']['is_robust']
            is_consistent = results['minimax_consistency']['is_consistent']
            if is_robust and is_consistent:
                summary['tests_passed'] += 1
                print("   ✅ Minimax Consistency: PASS")
            else:
                print("   ❌ Minimax Consistency: FAIL")
                if not is_robust:
                    summary['recommendations'].append("Model lacks robustness - worst-case performance too poor")
                if not is_consistent:
                    summary['recommendations'].append("Model inconsistent - high variance in returns")
        
        # Check relabeling validation
        if 'relabeling_validation' in results:
            summary['total_tests'] += 1
            reasonable_relabeling = results['relabeling_validation']['reasonable_relabeling']
            if reasonable_relabeling:
                summary['tests_passed'] += 1
                print("   ✅ Relabeling Validation: PASS")
            else:
                print("   ❌ Relabeling Validation: FAIL")
                summary['recommendations'].append("Trajectory relabeling may be incorrect - check minimax computation")
        
        # Calculate overall score
        if summary['total_tests'] > 0:
            summary['overall_score'] = summary['tests_passed'] / summary['total_tests']
        
        print(f"\n   📊 Overall Score: {summary['tests_passed']}/{summary['total_tests']} ({summary['overall_score']:.1%})")
        
        if summary['overall_score'] >= 0.8:
            print("   🎉 EXCELLENT: ARDT model validation successful!")
            summary['grade'] = 'EXCELLENT'
        elif summary['overall_score'] >= 0.6:
            print("   ✅ GOOD: ARDT model shows promise with minor issues")
            summary['grade'] = 'GOOD'
        elif summary['overall_score'] >= 0.4:
            print("   ⚠️ FAIR: ARDT model has significant issues requiring attention")
            summary['grade'] = 'FAIR'
        else:
            print("   ❌ POOR: ARDT model validation failed - major issues detected")
            summary['grade'] = 'POOR'
        
        if summary['recommendations']:
            print("\n   🔧 RECOMMENDATIONS:")
            for i, rec in enumerate(summary['recommendations'], 1):
                print(f"      {i}. {rec}")
        
        return summary
    
    def _save_results(self, results):
        """Save validation results"""
        results_dir = Path('kuhn_poker_validation_results')
        results_dir.mkdir(exist_ok=True)
        
        filename = results_dir / 'validation_results.json'
        
        with open(filename, 'w') as f:
            json.dump(results, f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")
    
    def plot_validation_results(self, results, save_plots=True):
        """Create validation plots"""
        if 'return_conditioning' not in results:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Kuhn Poker ARDT Validation Results', fontsize=16)
        
        # Plot 1: Return Conditioning
        conditioning = results['return_conditioning']
        targets = [k for k in conditioning.keys() if k != 'summary']
        achieved = [conditioning[t]['achieved_return'] for t in targets]
        errors = [conditioning[t]['target_error'] for t in targets]
        
        axes[0, 0].plot(targets, targets, 'k--', label='Perfect conditioning', alpha=0.7)
        axes[0, 0].scatter(targets, achieved, color='blue', s=100, alpha=0.7)
        axes[0, 0].set_xlabel('Target Return')
        axes[0, 0].set_ylabel('Achieved Return')
        axes[0, 0].set_title('Return Conditioning Accuracy')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Plot 2: Target Errors
        axes[0, 1].bar(range(len(targets)), errors, color='orange', alpha=0.7)
        axes[0, 1].set_xlabel('Target Return')
        axes[0, 1].set_ylabel('Absolute Error')
        axes[0, 1].set_title('Target Achievement Errors')
        axes[0, 1].set_xticks(range(len(targets)))
        axes[0, 1].set_xticklabels([f'{t:.1f}' for t in targets])
        axes[0, 1].grid(True, alpha=0.3)
        
        # Plot 3: Baseline Comparison (if available)
        if 'baseline_comparison' in results:
            comparison = results['baseline_comparison']
            comp_targets = [k for k in comparison.keys() if k != 'summary']
            ardt_means = [comparison[t]['ardt_mean'] for t in comp_targets]
            baseline_means = [comparison[t]['baseline_mean'] for t in comp_targets]
            
            x = np.arange(len(comp_targets))
            width = 0.35
            
            axes[1, 0].bar(x - width/2, ardt_means, width, label='ARDT', color='blue', alpha=0.7)
            axes[1, 0].bar(x + width/2, baseline_means, width, label='Baseline', color='red', alpha=0.7)
            axes[1, 0].set_xlabel('Target Return')
            axes[1, 0].set_ylabel('Achieved Return')
            axes[1, 0].set_title('ARDT vs Baseline Comparison')
            axes[1, 0].set_xticks(x)
            axes[1, 0].set_xticklabels([f'{t:.1f}' for t in comp_targets])
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
        else:
            axes[1, 0].text(0.5, 0.5, 'No baseline comparison\navailable', 
                           ha='center', va='center', transform=axes[1, 0].transAxes)
            axes[1, 0].set_title('Baseline Comparison')
        
        # Plot 4: Return Distribution for Minimax Consistency
        if 'minimax_consistency' in results:
            minimax_results = results['minimax_consistency']
            returns = minimax_results['all_returns']
            
            axes[1, 1].hist(returns, bins=20, alpha=0.7, color='green', edgecolor='black')
            axes[1, 1].axvline(minimax_results['achieved_mean'], color='red', linestyle='--', 
                              label=f'Mean: {minimax_results["achieved_mean"]:.2f}')
            axes[1, 1].axvline(minimax_results['robustness_score'], color='orange', linestyle='--',
                              label=f'10th percentile: {minimax_results["robustness_score"]:.2f}')
            axes[1, 1].set_xlabel('Return')
            axes[1, 1].set_ylabel('Frequency')
            axes[1, 1].set_title('Return Distribution (Minimax Consistency)')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
        else:
            axes[1, 1].text(0.5, 0.5, 'No minimax consistency\ntest available', 
                           ha='center', va='center', transform=axes[1, 1].transAxes)
            axes[1, 1].set_title('Minimax Consistency')
        
        plt.tight_layout()
        
        if save_plots:
            results_dir = Path('kuhn_poker_validation_results')
            results_dir.mkdir(exist_ok=True)
            plt.savefig(results_dir / 'validation_plots.png', dpi=300, bbox_inches='tight')
            print("📈 Plots saved to: kuhn_poker_validation_results/validation_plots.png")
        
        plt.show()