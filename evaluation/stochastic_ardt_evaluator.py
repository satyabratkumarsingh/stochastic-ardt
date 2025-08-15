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
from utils.saved_names import dt_model_name
import matplotlib.pyplot as plt


class ARDTModelLoader:
    """
    Handles loading of saved ARDT models (minimax and original)
    """
    
    def __init__(self, seed: int, game_name: str, config_path: str, device: str = 'cpu'):
        self.seed = seed
        self.game_name = game_name
        self.device = device
        
        # Load config for model architecture parameters
        config = yaml.safe_load(Path(config_path).read_text())
        self.dt_train_args = config
    
    def load_model(self, method: str, model_type: str = "minimax") -> Tuple[torch.nn.Module, Dict]:
        """
        Load a specific DT model
        
        Args:
            method: Training method used
            model_type: Either "minimax" or "original"
        """
        # Create model path with suffix
        base_path = dt_model_name(seed=self.seed, game=self.game_name, method=method)
        
        if base_path.endswith('.pth'):
            model_path = base_path.replace('.pth', f'_{model_type}.pth')
        else:
            model_path = f"{base_path}_{model_type}.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        model_params = checkpoint["model_params"]
        
        # Extract parameters from saved model_params
        obs_size = model_params['obs_size']
        action_size = model_params['action_size']
        action_type = model_params['action_type']
        horizon = model_params['horizon']
        effective_max_ep_len = model_params['effective_max_ep_len']
        
        # Recreate model using saved parameters
        dt_model = DecisionTransformer(
            state_dim=obs_size,
            act_dim=action_size,
            hidden_size=32,          # Match the checkpoint's hidden size
            max_length=horizon,
            max_ep_len=effective_max_ep_len,
            action_tanh=(action_type == 'continuous'),
            action_type=action_type,
            n_layer=2,               # Match the number of layers from the checkpoint
            n_head=1,                # Match the number of heads
            n_inner=256,             # Match the inner size from the checkpoint
            dropout=0.1
        ).to(self.device)
        
        # Load state dict
        dt_model.load_state_dict(checkpoint["model_state_dict"])
        dt_model.eval()
        
        print(f"✅ Loaded {model_type} DT model from {model_path}")
        print(f"   Model params: state_dim={obs_size}, act_dim={action_size}, action_type={action_type}")
        
        return dt_model, model_params
    
    def load_both_models(self, method: str) -> Dict[str, Tuple[torch.nn.Module, Dict]]:
        """
        Load both minimax and original models
        
        Returns:
            Dictionary with 'minimax' and 'original' keys containing (model, params) tuples
        """
        models = {}
        
        try:
            minimax_model, minimax_params = self.load_model(method, "minimax")
            models['minimax'] = (minimax_model, minimax_params)
            print("✅ Minimax model loaded successfully")
        except FileNotFoundError as e:
            print(f"❌ Failed to load minimax model: {e}")
            models['minimax'] = None
        except Exception as e:
            print(f"❌ Failed to load minimax model: {e}")
            models['minimax'] = None
        
        try:
            original_model, original_params = self.load_model(method, "original")
            models['original'] = (original_model, original_params)
            print("✅ Original model loaded successfully")
        except FileNotFoundError as e:
            print(f"❌ Failed to load original model: {e}")
            models['original'] = None
        except Exception as e:
            print(f"❌ Failed to load original model: {e}")
            models['original'] = None
        
        return models
    
    def get_model_info(self, method: str, model_type: str = "minimax") -> Dict:
        """
        Get model information without loading the full model
        """
        base_path = dt_model_name(seed=self.seed, game=self.game_name, method=method)
        
        if base_path.endswith('.pth'):
            model_path = base_path.replace('.pth', f'_{model_type}.pth')
        else:
            model_path = f"{base_path}_{model_type}.pth"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        checkpoint = torch.load(model_path, map_location='cpu')
        return checkpoint["model_params"]


class ARDTEvaluator:
    
    def __init__(
        self,
        env_name: str,
        env_instance,
        model_params: Dict,
        scale: float = 1.0,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        normalize_states: bool = True,
        device: str = 'cpu'
    ):
        """
        Initialize evaluator using model parameters from saved checkpoint
        
        Args:
            env_name: Name of the environment
            env_instance: Environment instance
            model_params: Model parameters dictionary from saved checkpoint
            scale: Scaling factor for returns
            state_mean: State normalization mean (optional)
            state_std: State normalization std (optional)
            normalize_states: Whether to normalize states
            device: Device to run evaluation on
        """
        self.env_name = env_name
        self.env_instance = env_instance
        self.normalize_states = normalize_states
        self.device = device
        self.scale = scale
        
        # Extract parameters from model_params
        self.state_dim = model_params['obs_size']
        self.act_dim = model_params['action_size']
        self.action_type = model_params['action_type']
        self.max_ep_len = model_params['effective_max_ep_len']
        
        # Set up state normalization if provided
        if state_mean is not None and state_std is not None:
            self.state_mean = torch.from_numpy(state_mean).float().to(device)
            self.state_std = torch.from_numpy(state_std).float().to(device)
        else:
            self.state_mean = None
            self.state_std = None
            self.normalize_states = False
        
        print(f"✅ ARDTEvaluator initialized:")
        print(f"   Environment: {env_name}")
        print(f"   State dim: {self.state_dim}, Action dim: {self.act_dim}")
        print(f"   Action type: {self.action_type}, Max episode length: {self.max_ep_len}")
        print(f"   Scale: {scale}, Normalize states: {self.normalize_states}")

    @classmethod
    def from_model_params(
        cls,
        env_name: str,
        env_instance,
        model_params: Dict,
        scale: float = 1.0,
        state_mean: Optional[np.ndarray] = None,
        state_std: Optional[np.ndarray] = None,
        normalize_states: bool = True,
        device: str = 'cpu'
    ):
        """
        Factory method to create evaluator from model parameters
        """
        return cls(
            env_name=env_name,
            env_instance=env_instance,
            model_params=model_params,
            scale=scale,
            state_mean=state_mean,
            state_std=state_std,
            normalize_states=normalize_states,
            device=device
        )

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

    def create_eval_function(self, case_type: str = "normal", debug: bool = False) -> Callable:
        """
        Creates a single, unified evaluation function for a given case.
        
        Args:
            case_type: "normal" or "worst"
        """
        if case_type == "worst":
            step_fn = self._worst_case_env_step
        else:
            step_fn = self._normal_case_env_step

        def eval_fn(model: torch.nn.Module, target_return: float, num_episodes: int = 50):
            model.eval()
            episode_returns = []
            episode_actions = []
            
            with torch.no_grad():
                for ep in tqdm(range(num_episodes), desc=f"Evaluating R={target_return} ({case_type.capitalize()})"):
                    episode_debug = debug and ep < 3
                    
                    episode_return, actions = self.evaluate_single_episode(
                        model, target_return, step_fn, debug=episode_debug
                    )
                    episode_returns.append(episode_return)
                    episode_actions.append(actions)
            
            return episode_returns, episode_actions
        return eval_fn

    def comprehensive_model_evaluation(
        self,
        minimax_model: torch.nn.Module,
        method: str = "minimax",
        target_returns: List[float] = None,
        num_episodes_per_target: int = 500,
        save_path: str = "evaluation_results"
    ) -> Dict:
        """
        Runs comprehensive evaluation for minimax model on normal and worst-case scenarios.
        """
        if target_returns is None:
            target_returns = [-2.0, -1.0, 0.0, 1.0, 2.0]
        
        print("🚀 Starting Comprehensive Minimax Model Evaluation...")
        print("=" * 70)
        
        results = {
            'minimax_normal': {},
            'minimax_worst': {}
        }
        
        # Create evaluation functions
        eval_fn_normal = self.create_eval_function(case_type="normal")
        eval_fn_worst = self.create_eval_function(case_type="worst")
        
        # --- Minimax Model Evaluation ---
        print("\n** Evaluating Minimax Model - Normal Case **")
        for target in target_returns:
            returns, _ = eval_fn_normal(minimax_model, target, num_episodes=num_episodes_per_target)
            results['minimax_normal'][f'target_{target}'] = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'returns': returns
            }
        
        print("\n** Evaluating Minimax Model - Worst Case **")
        for target in target_returns:
            returns, _ = eval_fn_worst(minimax_model, target, num_episodes=num_episodes_per_target)
            results['minimax_worst'][f'target_{target}'] = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'returns': returns
            }

        # --- Save and Plot Results ---
        self._save_results(results, save_path, method)
        self._plot_comparison_results(results, target_returns, save_path, method)
        
        if self.env_name == "kuhn_poker":
            self._plot_kuhn_poker_results(results, target_returns, save_path, method)
        
        print("\n** Minimax model evaluation complete. Results saved and plotted. **")
        
        return results

    def _save_results(self, results: Dict, save_path: str, method: str):
        """Saves evaluation results to a JSON file."""
        os.makedirs(save_path, exist_ok=True)
        filename = Path(save_path) / f'{method}_evaluation_results.json'
        
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

    def _plot_comparison_results(self, results: Dict, target_returns: List[float], save_path: str, method: str):
        """
        Plots single comprehensive comparison for model across different scenarios.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        
        target_returns = np.array(target_returns)

        # Get data from results
        minimax_normal_means = np.array([results['minimax_normal'][f'target_{t}']['mean'] for t in target_returns])
        minimax_normal_stds = np.array([results['minimax_normal'][f'target_{t}']['std'] for t in target_returns])
        minimax_worst_means = np.array([results['minimax_worst'][f'target_{t}']['mean'] for t in target_returns])
        minimax_worst_stds = np.array([results['minimax_worst'][f'target_{t}']['std'] for t in target_returns])

        ax.plot(target_returns, minimax_normal_means, 'o-', color='blue', label='Normal Case', linewidth=2, markersize=8)
        ax.fill_between(target_returns, minimax_normal_means - minimax_normal_stds, 
                        minimax_normal_means + minimax_normal_stds, color='blue', alpha=0.2)
        
        ax.plot(target_returns, minimax_worst_means, 'x-', color='red', label='Worst Case', linewidth=2, markersize=10)
        ax.fill_between(target_returns, minimax_worst_means - minimax_worst_stds, 
                        minimax_worst_means + minimax_worst_stds, color='red', alpha=0.2)
        
        ax.plot(target_returns, target_returns, '--', color='gray', label='Ideal Performance', linewidth=2)
        
        ax.set_xlabel("Target Return", fontsize=14)
        ax.set_ylabel("Achieved Mean Return", fontsize=14)
        ax.set_title(f"{method.capitalize()} Model Performance: All Scenarios", fontsize=16, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Enhance the plot appearance
        ax.tick_params(axis='both', which='major', labelsize=12)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = Path(save_path) / f'{method}_evaluation_plot.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"🖼️ Plot saved to {plot_filename}")
        plt.show()

    def _plot_kuhn_poker_results(self, results: Dict, target_returns: List[float], save_path: str, method: str):
        """
        Plots a single graph for Kuhn Poker comparing model performance to the Nash Equilibrium.
        """
        fig, ax = plt.subplots(figsize=(12, 8))
        target_returns = np.array(target_returns)
        
        # Kuhn Poker Nash Equilibrium value for Player 1
        NASH_EQUILIBRIUM_RETURN = -1/21  # Approximately -0.0476
        
        # Get data from results
        minimax_normal_means = np.array([results['minimax_normal'][f'target_{t}']['mean'] for t in target_returns])
        minimax_worst_means = np.array([results['minimax_worst'][f'target_{t}']['mean'] for t in target_returns])

        # Plotting model performance
        ax.plot(target_returns, minimax_normal_means, 'o-', color='blue', label=f'{method.capitalize()} Model (Normal Case)', linewidth=2)
        ax.plot(target_returns, minimax_worst_means, 'x-', color='red', label=f'{method.capitalize()} Model (Worst Case)', linewidth=2)
        
        # Plotting the Nash Equilibrium line
        ax.axhline(y=NASH_EQUILIBRIUM_RETURN, color='purple', linestyle='--', 
                  label=f'Nash Equilibrium Return ({NASH_EQUILIBRIUM_RETURN:.4f})', linewidth=2)
        
        # Plotting the ideal performance line
        ax.plot(target_returns, target_returns, '--', color='gray', label='Ideal Performance (Target = Achieved)', linewidth=2)

        ax.set_xlabel("Target Return", fontsize=12)
        ax.set_ylabel("Achieved Mean Return", fontsize=12)
        ax.set_title(f"Kuhn Poker: {method.capitalize()} Model Performance vs. Nash Equilibrium", fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save the plot
        plot_filename = Path(save_path) / f'kuhn_poker_{method}_results.png'
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        print(f"🖼️ Kuhn Poker specific plot saved to {plot_filename}")
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
    
    def validate_scenario_performance(self,
                                    model: torch.nn.Module,
                                    eval_fn_normal: Callable,
                                    eval_fn_worst: Callable,
                                    target_returns: List[float] = None,
                                    num_episodes: int = 50) -> Dict:
        """Validates model performance across different scenarios."""
        print("📊 Testing Scenario Performance...")
        
        target_returns = target_returns or [-1.0, 0.0, 1.0]
        improvements = []
        results = {}
        
        for target in target_returns:
            print(f"   Target: {target:.2f}")
            
            normal_returns, _ = eval_fn_normal(model, target, num_episodes)
            worst_returns, _ = eval_fn_worst(model, target, num_episodes)
            
            normal_mean = np.mean(normal_returns)
            worst_mean = np.mean(worst_returns)
            
            # Calculate robustness as the difference between normal and worst case
            robustness = normal_mean - worst_mean
            
            results[f'target_{target}'] = {
                'normal_mean': float(normal_mean),
                'worst_mean': float(worst_mean),
                'robustness': float(robustness)
            }
            
            improvements.append(robustness)
            print(f"     Normal: {normal_mean:.3f}, Worst: {worst_mean:.3f}")
            print(f"     Robustness: {robustness:.3f}")
        
        avg_robustness = np.mean([results[f'target_{t}']['robustness'] for t in target_returns])
        passed = avg_robustness > 0.1  # Model should perform better in normal vs worst case
        
        results['summary'] = {
            'avg_robustness': float(avg_robustness),
            'passed': bool(passed),
            'grade': 'PASS' if passed else 'FAIL'
        }
        
        print(f"   📈 Avg Robustness: {avg_robustness:.3f}")
        print(f"   📈 Result: {results['summary']['grade']}")
        
        return results
    
    def run_comprehensive_validation(
        self,
        minimax_model: torch.nn.Module,
        eval_fn_normal: Callable,
        eval_fn_worst: Callable,
        target_returns: List[float],
        num_episodes: int = 50
    ) -> Dict:
        """
        Runs comprehensive validation for minimax model across different scenarios.
        """
        print(f"🚀 Running comprehensive validation for minimax model...")
        print("=" * 60)
         
        results = {
            'normal_case': {},
            'worst_case': {},
            'scenario_comparison': {}
        }
        
        # Test return conditioning for each scenario
        print("\n--- Testing Minimax Model (Normal Case) ---")
        results['normal_case']['return_conditioning'] = self.validate_return_conditioning(
            minimax_model, eval_fn_normal, target_returns, num_episodes
        )
        
        print("\n--- Testing Minimax Model (Worst Case) ---")
        results['worst_case']['return_conditioning'] = self.validate_return_conditioning(
            minimax_model, eval_fn_worst, target_returns, num_episodes
        )
        
        # Test robustness for each scenario
        print("\n--- Testing Robustness (Normal Case) ---")
        results['normal_case']['robustness'] = self.validate_robustness(
            minimax_model, eval_fn_normal, 0.0, num_episodes
        )
        
        print("\n--- Testing Robustness (Worst Case) ---")
        results['worst_case']['robustness'] = self.validate_robustness(
            minimax_model, eval_fn_worst, 0.0, num_episodes
        )
        
        # Compare performance across scenarios
        print("\n--- Comparing Scenarios ---")
        results['scenario_comparison'] = self.validate_scenario_performance(
            minimax_model, eval_fn_normal, eval_fn_worst, target_returns, num_episodes
        )
        
        print("\n✨ Comprehensive validation complete.")
        return results