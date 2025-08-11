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
import gym
from core_models.decision_transformer.decision_transformer import DecisionTransformer
from core_models.behaviour_cloning.behaviour_cloning import MLPBCModel
from utils.saved_names import dt_model_name, behaviour_cloning_model_name
import matplotlib.pyplot as plt


class ARDTEvaluator:
    
    def __init__(
        self,
        env_name: str,
        env_instance,
        state_dim: int,
        act_dim: int,
        action_type: str,
        max_ep_len: int,
        scale: float,
        state_mean: np.ndarray,
        state_std: np.ndarray,
        adv_act_dim: Optional[int] = None,
        normalize_states: bool = True,
        device: str = 'cpu'
    ):
        self.env_name = env_name
        self.env_instance = env_instance
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.action_type = action_type
        self.max_ep_len = max_ep_len
        self.scale = scale
        self.adv_act_dim = adv_act_dim or act_dim
        self.normalize_states = normalize_states
        self.device = device
        
        self.state_mean = torch.from_numpy(state_mean).float().to(device)
        self.state_std = torch.from_numpy(state_std).float().to(device)

    def _decode_kuhn_state(self, state: np.ndarray) -> str:
        if self.env_name != "kuhn_poker":
            return "N/A"
        
        if len(state) >= 12:
            active_pos = np.where(state == 1.0)[0]
            if len(active_pos) > 0:
                pos = active_pos[0]
                if pos < 3:
                    return f"P0_card_{pos}"
                elif pos < 6:
                    return f"P1_card_{pos-3}"
                elif pos < 9:
                    return f"History_{pos-6}"
                else:
                    return f"Other_{pos}"
        return "Unknown_state"

    def _get_env(self):
        if hasattr(self.env_instance, 'reset'):
            return self.env_instance
        try:
            return self.env_instance()
        except TypeError:
            return self.env_instance

    def _worst_case_env_step(self, action: Any, env) -> Tuple:
        action = int(np.argmax(action) if isinstance(action, (np.ndarray, torch.Tensor)) and action.size > 1 else action)
        if self.env_name == "kuhn_poker" and env.player_turn == 0:
            new_state, reward, terminated, truncated, info = env.step(action)
            if not (terminated or truncated) and env.player_turn == 1:
                adv_action = np.random.choice([0, 1])
                new_state, reward_adv, terminated_adv, truncated_adv, info_adv = env.step(adv_action)
                reward += reward_adv
                terminated = terminated_adv
                truncated = truncated_adv
                info.update(info_adv)
                info["adv_action"] = adv_action
            return new_state, reward, terminated, truncated, info
        else:
            return env.step(action)

            
    def _normal_case_env_step(self, action: Any, env) -> Tuple:
        """Standard environment step."""
        action = int(np.argmax(action) if isinstance(action, (np.ndarray, torch.Tensor)) and action.size > 1 else action)
        step_result = env.step(action)
        if len(step_result) == 4:
            next_state, reward, done, info = step_result
            return next_state, reward, done, done, info
        else:
            return step_result

    # 
    def evaluate_single_episode(
    self,
    model: torch.nn.Module,
    target_return: float,
    step_fn: Callable,
    debug: bool = False
    ) -> tuple[float, list[int]]:
        debug = False
        env = self._get_env()
        env_state = env.reset()
        if isinstance(env_state, tuple):
            state, _ = env_state
        else:
            state = env_state

        episode_return = 0.0
        episode_actions = []

        # Buffers for states, actions, returns-to-go
        states = torch.zeros(1, self.max_ep_len, self.state_dim, device=self.device)
        actions = torch.zeros(1, self.max_ep_len, self.act_dim, device=self.device)
        returns_to_go = torch.zeros(1, self.max_ep_len, 1, device=self.device)

        # Initialize first return-to-go with scaled target_return
        returns_to_go[:, 0, 0] = target_return * self.scale

        if debug:
            print(f"\n🎯 Episode Start: target={target_return:.3f}")

        model.eval()

        for step in range(self.max_ep_len):
            if debug:
                print(f"\n--- Step {step} ---")

            # Current state -> store in buffer
            states[:, step, :] = torch.from_numpy(state.astype(np.float32)).to(self.device)

            # Calculate returns-to-go for current step: target_return - cum_return so far
            if step > 0:
                returns_to_go[:, step, 0] = (target_return - episode_return) * self.scale

            # Prepare shifted actions input: shift right by 1 timestep, first is zeros
            if step == 0:
                shifted_actions = torch.zeros_like(actions[:, :step+1, :])
            else:
                shifted_actions = torch.zeros(1, step+1, self.act_dim, device=self.device)
                shifted_actions[:, 1:, :] = actions[:, :step, :]

            # Build attention mask (all ones for current length)
            attention_mask = torch.ones(1, step+1, dtype=torch.bool, device=self.device)

            with torch.no_grad():
                try:
                    _, action_preds, _ = model(
                        states=states[:, :step+1, :],
                        actions=shifted_actions,
                        returns_to_go=returns_to_go[:, :step+1, :],
                        timesteps=torch.tensor([[step]], device=self.device, dtype=torch.long),
                        attention_mask=attention_mask
                    )

                    action_logits = action_preds[0, -1, :]  # last token's prediction
                    action_probs = F.softmax(action_logits, dim=0)
                    action_idx = Categorical(action_probs).sample().item()

                except Exception as e:
                    print(f"🚨 Model forward pass failed: {e}")
                    action_idx = np.random.choice(self.act_dim)

            # Convert discrete action index to one-hot
            action_onehot = torch.zeros(self.act_dim, device=self.device)
            action_onehot[action_idx] = 1.0

            # Store action in buffer
            actions[:, step, :] = action_onehot

            episode_actions.append(action_idx)

            # Step env
            next_state, reward, terminated, truncated, _ = step_fn(action_idx, env)
            done = terminated or truncated
            episode_return += reward

            if done:
                if debug:
                    print(f"Episode finished: length={step+1}, return={episode_return:.3f}")
                break

            state = next_state

        return episode_return, episode_actions


    def create_eval_function(self, model_type: str, worst_case: bool = False, debug: bool = False) -> Callable:
        """
        Creates a single, unified evaluation function for a given model type and case.
        """
        step_fn = self._worst_case_env_step if worst_case else self._normal_case_env_step

        def eval_fn(model: torch.nn.Module, target_return: float, num_episodes: int = 50):
            model.eval()
            episode_returns = []
            episode_actions = []
            
            with torch.no_grad():
                for ep in tqdm(range(num_episodes), desc=f"Evaluating R={target_return} ({'Worst' if worst_case else 'Normal'})"):
                    episode_debug = debug and ep < 3
                    
                    episode_return, actions = self.evaluate_single_episode(
                        model, target_return, step_fn, debug=episode_debug
                    )
                    episode_returns.append(episode_return)
                    episode_actions.append(actions)
            
            return episode_returns, episode_actions
        return eval_fn

    def comprehensive_dual_evaluation(
        self,
        model: torch.nn.Module,
        target_returns: List[float] = None,
        num_episodes_per_target: int = 500,
        save_path: str = "evaluation_results"
    ) -> Dict:
        """
        Runs comprehensive evaluation on both normal and worst-case scenarios.
        Saves and plots the results.
        """
        if target_returns is None:
            target_returns = [-2.0, -1.0, 0.0, 1.0, 2.0]
        
        print("🚀 Starting Comprehensive Dual ARDT Evaluation...")
        print("=" * 60)
        
        results = {'normal': {}, 'worst_case': {}}
        
        # --- Normal Case Evaluation ---
        print("\n** Evaluating Normal Case **")
        eval_fn_normal = self.create_eval_function(model_type='dt', worst_case=False)
        for target in target_returns:
            returns, _ = eval_fn_normal(model, target, num_episodes=num_episodes_per_target)
            results['normal'][f'target_{target}'] = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'returns': returns
            }
        
        # --- Worst-Case Evaluation ---
        print("\n** Evaluating Worst-Case **")
        eval_fn_worst = self.create_eval_function(model_type='dt', worst_case=True)
        for target in target_returns:
            returns, _ = eval_fn_worst(model, target, num_episodes=num_episodes_per_target)
            results['worst_case'][f'target_{target}'] = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'returns': returns
            }

        # --- Save and Plot Results ---
        self._save_results(results, save_path)
        self._plot_results(results, target_returns, save_path)
        
        print("\n** Dual evaluation complete. Results saved and plotted. **")
        
        return results

    def _save_results(self, results: Dict, save_path: str):
        """Saves evaluation results to a JSON file."""
        os.makedirs(save_path, exist_ok=True)
        filename = Path(save_path) / 'dual_evaluation_results.json'
        
        # Convert numpy types to native Python types for JSON serialization
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            return obj
            
        with open(filename, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
        print(f"💾 Results saved to {filename}")

    def _plot_results(self, results: Dict, target_returns: List[float], save_path: str):
        """
        Plots the results for both normal and worst-case scenarios with standard deviation.
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        target_returns = np.array(target_returns)

        # Plot Normal Case
        normal_means = np.array([results['normal'][f'target_{t}']['mean'] for t in target_returns])
        normal_stds = np.array([results['normal'][f'target_{t}']['std'] for t in target_returns])
        ax.plot(target_returns, normal_means, 'o-', color='green', label='Normal Case')
        ax.fill_between(target_returns, normal_means - normal_stds, normal_means + normal_stds, color='green', alpha=0.2)

        # Plot Worst-Case
        worst_means = np.array([results['worst_case'][f'target_{t}']['mean'] for t in target_returns])
        worst_stds = np.array([results['worst_case'][f'target_{t}']['std'] for t in target_returns])
        ax.plot(target_returns, worst_means, 'x-', color='red', label='Worst-Case')
        ax.fill_between(target_returns, worst_means - worst_stds, worst_means + worst_stds, color='red', alpha=0.2)
        
        # Plot ideal line
        ax.plot(target_returns, target_returns, '--', color='gray', label='Ideal')
        
        # Add labels, title, and legend
        ax.set_xlabel("Target Return")
        ax.set_ylabel("Achieved Mean Return")
        ax.set_title("ARDT Performance: Normal vs. Worst-Case")
        ax.legend()
        ax.grid(True)
        
        # Save and show the plot
        plot_filename = Path(save_path) / 'dual_evaluation_plot.png'
        plt.savefig(plot_filename)
        print(f"🖼️ Plot saved to {plot_filename}")
        plt.show()

class ARDTValidator:
    """
    A suite of validation tests for ARDT models.
    This class analyzes the performance results generated by an evaluator.
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
    def validate_return_conditioning(self, 
                                   model: torch.nn.Module,
                                   eval_fn: Callable,
                                   target_returns: List[float] = None,
                                   num_episodes: int = 50) -> Dict:
        """Tests if the model can achieve a range of target returns."""
        print("🎯 Testing Return Conditioning...")
        
        target_returns = target_returns or [-2.0, -1.0, 0.0, 1.0, 2.0]
        results = {}
        errors = []
        success_rates = []
        
        for target in target_returns:
            print(f"   Target: {target:.2f}")
            returns, _ = eval_fn(model, target, num_episodes)
            achieved = np.mean(returns)
            error = abs(achieved - target)
            success_rate = np.mean([abs(r - target) < 0.3 for r in returns])
            
            results[f'target_{target}'] = {
                'achieved': float(achieved),
                'error': float(error),
                'success_rate': float(success_rate),
                'std': float(np.std(returns))
            }
            
            errors.append(error)
            success_rates.append(success_rate)
            print(f"     → Achieved: {achieved:.3f}, Error: {error:.3f}, Success: {success_rate:.3f}")
        
        avg_error = np.mean(errors)
        avg_success = np.mean(success_rates)
        passed = avg_success > 0.6 and avg_error < 0.5
        
        results['summary'] = {
            'avg_error': float(avg_error),
            'avg_success_rate': float(avg_success),
            'passed': bool(passed),
            'grade': 'PASS' if passed else 'FAIL'
        }
        
        print(f"   📊 Avg Error: {avg_error:.3f}, Avg Success: {avg_success:.3f}")
        print(f"   📊 Result: {results['summary']['grade']}")
        
        return results
    
    def validate_robustness(self,
                           model: torch.nn.Module,
                           eval_fn: Callable,
                           target_return: float = 0.0,
                           num_episodes: int = 100) -> Dict:
        """Tests the model's robustness and consistency."""
        print("🛡️ Testing Robustness...")
        
        returns, _ = eval_fn(model, target_return, num_episodes)
        
        mean_return = np.mean(returns)
        worst_return = np.min(returns)
        std_return = np.std(returns)
        robustness_score = np.percentile(returns, 10)
        
        is_robust = robustness_score > target_return - 1.0
        is_consistent = std_return < 1.0
        passed = is_robust and is_consistent
        
        results = {
            'mean_return': float(mean_return),
            'worst_return': float(worst_return),
            'std_return': float(std_return),
            'robustness_score': float(robustness_score),
            'is_robust': bool(is_robust),
            'is_consistent': bool(is_consistent),
            'passed': bool(passed),
            'grade': 'PASS' if passed else 'FAIL'
        }
        
        print(f"   Mean: {mean_return:.3f}, Worst: {worst_return:.3f}, Std: {std_return:.3f}")
        print(f"   Robustness (10th %): {robustness_score:.3f}")
        print(f"   Result: {results['grade']}")
        
        return results
    
    def compare_models(self,
                      ardt_model: torch.nn.Module,
                      baseline_model: torch.nn.Module,
                      eval_fn: Callable,
                      target_returns: List[float] = None,
                      num_episodes: int = 50) -> Dict:
        """Compares the ARDT model's performance against a baseline model."""
        print("📊 Comparing Models...")
        
        target_returns = target_returns or [-1.0, 0.0, 1.0]
        improvements = []
        results = {}
        
        for target in target_returns:
            print(f"   Target: {target:.2f}")
            
            ardt_returns, _ = eval_fn(ardt_model, target, num_episodes)
            baseline_returns, _ = eval_fn(baseline_model, target, num_episodes)
            
            ardt_mean = np.mean(ardt_returns)
            baseline_mean = np.mean(baseline_returns)
            improvement = ardt_mean - baseline_mean
            
            results[f'target_{target}'] = {
                'ardt_mean': float(ardt_mean),
                'baseline_mean': float(baseline_mean),
                'improvement': float(improvement)
            }
            
            improvements.append(improvement)
            print(f"     ARDT: {ardt_mean:.3f}, Baseline: {baseline_mean:.3f}")
            print(f"     Improvement: {improvement:.3f}")
        
        avg_improvement = np.mean(improvements)
        passed = avg_improvement > 0.05
        
        results['summary'] = {
            'avg_improvement': float(avg_improvement),
            'passed': bool(passed),
            'grade': 'PASS' if passed else 'FAIL'
        }
        
        print(f"   📈 Avg Improvement: {avg_improvement:.3f}")
        print(f"   📈 Result: {results['summary']['grade']}")
        
        return results
    
    def run_validation(
        self,
        ardt_model: torch.nn.Module,
        dt_eval_fn: Callable,
        baseline_model: Optional[torch.nn.Module],
        bc_eval_fn: Optional[Callable],
        target_returns: List[float],
        num_episodes: int
    ) -> Dict:
        """
        Runs the validation for both ARDT and baseline models.
        """
        print(f"🚀 Running validation for {self.device} device...")
         
        results = {'dt': {}, 'bc': {}}
        
        for target_return in target_returns:
            print(f"\n--- Evaluating ARDT Model with target_return: {target_return} ---")
            dt_returns, dt_actions = dt_eval_fn(ardt_model, target_return, num_episodes)

            results['dt'][f'target_return_{target_return}'] = {
                'returns': np.mean(dt_returns),
                'std': np.std(dt_returns)
            }
            
            if baseline_model and bc_eval_fn:
                print(f"\n--- Evaluating Baseline (BC) Model with target_return: {target_return} ---")
                bc_returns, bc_actions = bc_eval_fn(baseline_model, target_return, num_episodes)
                results['bc'][f'target_return_{target_return}'] = {
                    'returns': np.mean(bc_returns),
                    'std': np.std(bc_returns)
                }
        
        return results
    
    def _generate_summary(self, test_results: Dict) -> Dict:
        """Generates a summary of all test results."""
        passed_tests = sum(
            1 for test_result in test_results.values()
            if test_result.get('passed', False) or test_result.get('summary', {}).get('passed', False)
        )
        total_tests = len(test_results)
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        print(f"   ✅ Tests Passed: {passed_tests}/{total_tests}")
        print(f"   📊 Success Rate: {success_rate:.1%}")

        if success_rate >= 0.8:
            grade, emoji = "EXCELLENT", "🎉"
        elif success_rate >= 0.6:
            grade, emoji = "GOOD", "✅"
        elif success_rate >= 0.4:
            grade, emoji = "FAIR", "⚠️"
        else:
            grade, emoji = "POOR", "❌"
            
        print(f"   {emoji} Grade: {grade}")
        
        return {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': success_rate,
            'grade': grade,
            'overall_passed': success_rate >= 0.6
        }

    def _save_results(self, results: Dict):
        """Saves the validation results to a JSON file."""
        results_dir = Path('ardt_validation_results')
        results_dir.mkdir(exist_ok=True)
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating, np.integer, bool)):
                return obj.item()
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        filename = results_dir / 'validation_results.json'
        with open(filename, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")