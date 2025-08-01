import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from tqdm import tqdm
from typing import Tuple, List, Optional, Callable, Dict, Any
import json
from pathlib import Path


class CleanARDTEvaluator:
    """
    Clean, integrated ARDT evaluation pipeline
    Combines evaluation logic with validation in a simple interface
    """
    
    def __init__(self, 
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
                 device: str = 'cpu'):
    
        self.env_name = env_name
        self.env_instance = env_instance
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.action_type = action_type
        self.max_ep_len = max_ep_len
        self.scale = scale
        self.state_mean = torch.from_numpy(state_mean).float()
        self.state_std = torch.from_numpy(state_std).float()
        self.adv_act_dim = adv_act_dim or act_dim
        self.normalize_states = normalize_states
        self.device = device
        
        # Move tensors to device
        self.state_mean = self.state_mean.to(device)
        self.state_std = self.state_std.to(device)
        
    def _worst_case_env_step(self, state: np.ndarray, action: Any, timestep: int, env) -> Tuple:
        """Handle worst-case adversary for Kuhn Poker"""
        # Convert action to int if needed
        if isinstance(action, np.ndarray):
            action = np.argmax(action) if action.size > 1 else int(action.item())
        elif isinstance(action, (np.integer, int, torch.Tensor)):
            action = int(action)

        if self.env_name == "kuhn_poker":
            current_player_turn = env.player_turn
            
            if current_player_turn == 0:  # Agent's turn
                # Agent takes action
                new_state, reward, terminated, truncated, info = env.step(action)
                
                # If adversary's turn next and game not done
                if not (terminated or truncated) and env.player_turn == 1:
                    # Adversary takes worst-case action (always bet/call = 1)
                    adv_action = 1
                    new_state, reward, terminated, truncated, info_adv = env.step(adv_action)
                    info.update(info_adv)
                    
                info["adv_action"] = info.get("adv_action", -1)
                return new_state, reward, terminated, truncated, info
                
            else:
                # Shouldn't happen in normal flow
                print(f"WARNING: Unexpected player turn {current_player_turn}")
                return env.step(action)
        else:
            # For other environments, just step normally
            return env.step(action)


    def evaluate_single_episode(self, 
                             model: torch.nn.Module,
                             model_type: str,
                             target_return: float,
                             worst_case: bool = True) -> Tuple[float, int]:
        model.eval()
        # FIX: Ensure the model's parameters are the same data type as the inputs.
        model.float().to(self.device)
        
        # Get environment instance
        if hasattr(self.env_instance, 'reset'):
            env = self.env_instance
        else:
            try:
                env = self.env_instance()
            except TypeError:
                env = self.env_instance

        # Reset environment
        state = env.reset()
        if isinstance(state, tuple):
            state, _ = state
        
        episode_return = 0.0
        episode_length = 0
        
        # FIX: Scale the target return to match the scaled rewards.
        # This is a critical logical fix for conditioning the model correctly.
        scaled_target_return = target_return / self.scale
        
        # CRITICAL FIX 2: Initialize ALL history tensors with a sequence length of 1.
        states = torch.from_numpy(state.astype(np.float32)).to(self.device).reshape(1, 1, self.state_dim)
        actions = torch.zeros((1, 1, self.act_dim), device=self.device, dtype=torch.float32)
        adv_actions = torch.zeros((1, 1, self.adv_act_dim), device=self.device, dtype=torch.float32)
        # Reverting to the diminishing return-to-go, but with a scaled initial value.
        returns_to_go = torch.tensor([[scaled_target_return]], device=self.device, dtype=torch.float32).reshape(1, 1, 1)
        timesteps = torch.tensor([[0]], device=self.device, dtype=torch.long).reshape(1, 1)
        
        with torch.no_grad():
            for t in range(self.max_ep_len):
                # The Decision Transformer expects a fixed-size sequence.
                # We pad the current history to the max episode length.
                
                states_padded = torch.cat([states, torch.zeros((1, self.max_ep_len - states.shape[1], self.state_dim), device=self.device)], dim=1)
                actions_padded = torch.cat([actions, torch.zeros((1, self.max_ep_len - actions.shape[1], self.act_dim), device=self.device)], dim=1)
                returns_to_go_padded = torch.cat([returns_to_go, torch.zeros((1, self.max_ep_len - returns_to_go.shape[1], 1), device=self.device)], dim=1)
                timesteps_padded = torch.cat([timesteps, torch.zeros((1, self.max_ep_len - timesteps.shape[1]), device=self.device, dtype=torch.long)], dim=1)
                
                # Normalize states before passing to the model
                if self.normalize_states:
                    states_normalized = (states_padded - self.state_mean.float().unsqueeze(0)) / self.state_std.float().unsqueeze(0)
                else:
                    states_normalized = states_padded
                
                # Get action from model
                if model_type == 'dt':
                    _, action_preds, _ = model(
                        states=states_normalized,
                        actions=actions_padded,
                        returns_to_go=returns_to_go_padded,
                        timesteps=timesteps_padded
                    )
                    action = action_preds[0, t] 
                    
                elif model_type == 'adt':
                    adv_actions_padded = torch.cat([adv_actions, torch.zeros((1, self.max_ep_len - adv_actions.shape[1], self.adv_act_dim), device=self.device)], dim=1)
                    action_preds = model(
                        states=states_normalized,
                        actions=actions_padded,
                        adv_actions=adv_actions_padded,
                        returns_to_go=returns_to_go_padded,
                        timesteps=timesteps_padded
                    )
                    action = action_preds[0, t]
                    
                elif model_type == 'bc':
                    action_preds = model(states=states_normalized[:, t, :])
                    action = action_preds[0]
                
                # Handle actions (discrete vs. continuous)
                if self.action_type == 'discrete':
                    action_probs = F.softmax(action, dim=-1)
                    action_idx = Categorical(probs=action_probs).sample()
                    action_for_step = action_idx.item()
                    action_for_history = F.one_hot(action_idx, num_classes=self.act_dim).float()
                else:
                    action_for_step = action.detach().cpu().numpy()
                    action_for_history = action
                
                # Step environment using the agent's action
                if worst_case and self.env_name in ["kuhn_poker", "gambling", "toy", "mstoy", "new_mstoy"]:
                    next_state, reward, terminated, truncated, info = self._worst_case_env_step(
                        state, action_for_step, t, env
                    )
                else:
                    step_result = env.step(action_for_step)
                    if len(step_result) == 4:
                        next_state, reward, terminated, info = step_result
                        truncated = False
                    else:
                        next_state, reward, terminated, truncated, info = step_result
                
                done = terminated or truncated
                
                # Handle adversarial action from info
                adv_action = info.get("adv_action", -1)
                if adv_action != -1 and adv_action is not None:
                    if self.action_type == 'discrete':
                        adv_action_for_history = F.one_hot(
                            torch.tensor(adv_action), num_classes=self.adv_act_dim
                        ).float().to(self.device)
                    else:
                        adv_action_for_history = torch.tensor([adv_action]).float().to(self.device)
                else:
                    adv_action_for_history = torch.zeros(self.adv_act_dim, device=self.device)
                
                # The next RTG is the current RTG minus the scaled reward received.
                # We now use a temporary variable for the next RTG calculation.
                next_rtg_for_model = (returns_to_go[0, t] - (reward / self.scale)).squeeze()
                
                # Update histories with new data
                state = next_state
                next_state_tensor = torch.from_numpy(next_state.astype(np.float32)).to(self.device).reshape(1, 1, self.state_dim)
                
                states = torch.cat([states, next_state_tensor], dim=1)
                actions = torch.cat([actions, action_for_history.reshape(1, 1, self.act_dim)], dim=1)
                adv_actions = torch.cat([adv_actions, adv_action_for_history.reshape(1, 1, self.adv_act_dim)], dim=1)
                
                returns_to_go = torch.cat([
                    returns_to_go,
                    torch.tensor([[[next_rtg_for_model]]], device=self.device, dtype=torch.float32)
                ], dim=1)
                
                timesteps = torch.cat([timesteps, torch.tensor([[t + 1]], device=self.device)], dim=1)
                
                episode_return += reward
                episode_length += 1
                
                if done:
                    break
        
        return episode_return, episode_length


    
    def evaluate_model(self, 
                      model: torch.nn.Module,
                      model_type: str,
                      target_return: float,
                      num_episodes: int,
                      worst_case: bool = True) -> Tuple[np.ndarray, np.ndarray]:
      
        returns = []
        lengths = []
        
        desc = f"Evaluating {model_type} (target: {target_return:.2f})"
        for _ in tqdm(range(num_episodes), desc=desc):
            episode_return, episode_length = self.evaluate_single_episode(
                model, model_type, target_return, worst_case
            )
            returns.append(episode_return)
            lengths.append(episode_length)
        
        return np.array(returns), np.array(lengths)
    
    def create_eval_function(self, model_type: str = 'dt', worst_case: bool = True) -> Callable:
       
        def eval_function(model: torch.nn.Module, 
                         target_return: float, 
                         num_episodes: int) -> Tuple[np.ndarray, np.ndarray]:
            return self.evaluate_model(
                model=model,
                model_type=model_type,
                target_return=target_return,
                num_episodes=num_episodes,
                worst_case=worst_case
            )
        
        return eval_function
    
class SimpleARDTValidator:
    """
    Simplified ARDT validation suite
    """
    
    def __init__(self, device: str = 'cpu'):
        self.device = device
        
    def validate_return_conditioning(self, 
                                   model: torch.nn.Module,
                                   eval_fn: Callable,
                                   target_returns: List[float] = None,
                                   num_episodes: int = 50) -> Dict:
        """Test return conditioning accuracy"""
        print("🎯 Testing Return Conditioning...")
        
        if target_returns is None:
            target_returns = [-2.0, -1.0, 0.0, 1.0, 2.0]
            
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
                'achieved': achieved,
                'error': error,
                'success_rate': success_rate,
                'std': np.std(returns)
            }
            
            errors.append(error)
            success_rates.append(success_rate)
            print(f"     → Achieved: {achieved:.3f}, Error: {error:.3f}, Success: {success_rate:.3f}")
        
        avg_error = np.mean(errors)
        avg_success = np.mean(success_rates)
        passed = avg_success > 0.6 and avg_error < 0.5
        
        results['summary'] = {
            'avg_error': avg_error,
            'avg_success_rate': avg_success,
            'passed': passed,
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
        """Test model robustness"""
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
            'mean_return': mean_return,
            'worst_return': worst_return,
            'std_return': std_return,
            'robustness_score': robustness_score,
            'is_robust': is_robust,
            'is_consistent': is_consistent,
            'passed': passed,
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
        """Compare ARDT vs baseline"""
        print("📊 Comparing Models...")
        
        if target_returns is None:
            target_returns = [-1.0, 0.0, 1.0]
            
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
                'ardt_mean': ardt_mean,
                'baseline_mean': baseline_mean,
                'improvement': improvement
            }
            
            improvements.append(improvement)
            print(f"     ARDT: {ardt_mean:.3f}, Baseline: {baseline_mean:.3f}")
            print(f"     Improvement: {improvement:.3f}")
        
        avg_improvement = np.mean(improvements)
        passed = avg_improvement > 0.05
        
        results['summary'] = {
            'avg_improvement': avg_improvement,
            'passed': passed,
            'grade': 'PASS' if passed else 'FAIL'
        }
        
        print(f"   📈 Avg Improvement: {avg_improvement:.3f}")
        print(f"   📈 Result: {results['summary']['grade']}")
        
        return results
    
    def run_validation(self,
                      ardt_model: torch.nn.Module,
                      eval_fn: Callable,
                      baseline_model: Optional[torch.nn.Module] = None,
                      target_returns: Optional[List[float]] = None,
                      num_episodes: int = 50) -> Dict:
        """Run complete validation suite"""
        print("\n" + "="*60)
        print("🚀 ARDT VALIDATION SUITE")
        print("="*60)
        
        results = {'tests': {}}
        
        # Test 1: Return Conditioning
        print("\n1️⃣ RETURN CONDITIONING")
        print("-" * 30)
        results['tests']['return_conditioning'] = self.validate_return_conditioning(
            ardt_model, eval_fn, target_returns, num_episodes
        )
        
        # Test 2: Robustness
        print("\n2️⃣ ROBUSTNESS")
        print("-" * 30)
        results['tests']['robustness'] = self.validate_robustness(
            ardt_model, eval_fn, num_episodes=num_episodes * 2
        )
        
        # Test 3: Model Comparison
        if baseline_model is not None:
            print("\n3️⃣ MODEL COMPARISON")
            print("-" * 30)
            results['tests']['model_comparison'] = self.compare_models(
                ardt_model, baseline_model, eval_fn, target_returns, num_episodes
            )
        
        # Generate summary
        print("\n4️⃣ OVERALL RESULTS")
        print("-" * 30)
        results['summary'] = self._generate_summary(results['tests'])
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _generate_summary(self, test_results: Dict) -> Dict:
        """Generate validation summary"""
        passed_tests = 0
        total_tests = 0
        
        for test_name, test_result in test_results.items():
            total_tests += 1
            if test_result.get('passed', False) or test_result.get('summary', {}).get('passed', False):
                passed_tests += 1
                print(f"   ✅ {test_name.replace('_', ' ').title()}: PASS")
            else:
                print(f"   ❌ {test_name.replace('_', ' ').title()}: FAIL")
        
        success_rate = passed_tests / total_tests if total_tests > 0 else 0
        
        if success_rate >= 0.8:
            grade, emoji = "EXCELLENT", "🎉"
        elif success_rate >= 0.6:
            grade, emoji = "GOOD", "✅"
        elif success_rate >= 0.4:
            grade, emoji = "FAIR", "⚠️"
        else:
            grade, emoji = "POOR", "❌"
        
        summary = {
            'passed_tests': passed_tests,
            'total_tests': total_tests,
            'success_rate': success_rate,
            'grade': grade,
            'overall_passed': success_rate >= 0.6
        }
        
        print(f"\n   📊 Score: {passed_tests}/{total_tests} ({success_rate:.1%})")
        print(f"   {emoji} Grade: {grade}")
        
        return summary
    
    def _save_results(self, results: Dict):
        """Save results to JSON"""
        results_dir = Path('ardt_validation_results')
        results_dir.mkdir(exist_ok=True)
        
        def convert_numpy(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(item) for item in obj]
            return obj
        
        filename = results_dir / 'validation_results.json'
        with open(filename, 'w') as f:
            json.dump(convert_numpy(results), f, indent=2)
        
        print(f"\n💾 Results saved to: {filename}")

def run_ardt_validation_pipeline(
    ardt_model: torch.nn.Module,
    env_name: str,
    env_instance,
    state_dim: int,
    act_dim: int,
    action_type: str,
    max_ep_len: int,
    scale: float,
    state_mean: np.ndarray,
    state_std: np.ndarray,
    baseline_model: Optional[torch.nn.Module] = None,
    adv_act_dim: Optional[int] = None,
    normalize_states: bool = True,
    target_returns: Optional[List[float]] = None,
    num_episodes: int = 50,
    device: str = 'cpu'
) -> Dict:
   
    # Create evaluator
    evaluator = CleanARDTEvaluator(
        env_name=env_name,
        env_instance=env_instance,
        state_dim=state_dim,
        act_dim=act_dim,
        action_type=action_type,
        max_ep_len=max_ep_len,
        scale=scale,
        state_mean=state_mean,
        state_std=state_std,
        adv_act_dim=adv_act_dim,
        normalize_states=normalize_states,
        device=device
    )
    
    # Create evaluation function
    eval_fn = evaluator.create_eval_function(model_type='dt', worst_case=False)
    
    # Create validator and run validation
    validator = SimpleARDTValidator(device=device)
    
    return validator.run_validation(
        ardt_model=ardt_model,
        eval_fn=eval_fn,
        baseline_model=baseline_model,
        target_returns=target_returns,
        num_episodes=num_episodes
    )