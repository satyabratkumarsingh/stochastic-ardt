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
import pickle

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
            rtg_seq=True      
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

    # --- Initialization remains same ---
    def __init__(self, env_name: str, env_instance, model_params: Dict,
                 scale: float = 1.0, state_mean: Optional[np.ndarray] = None,
                 state_std: Optional[np.ndarray] = None, normalize_states: bool = True,
                 device: str = 'cpu'):

        self.env_name = env_name
        self.env_instance = env_instance
        self.normalize_states = normalize_states
        self.device = device
        self.scale = scale

        self.state_dim = model_params['obs_size']
        self.act_dim = model_params['action_size']
        self.action_type = model_params['action_type']
        self.max_ep_len = model_params['effective_max_ep_len']

        if state_mean is not None and state_std is not None:
            self.state_mean = torch.from_numpy(state_mean).float().to(device)
            self.state_std = torch.from_numpy(state_std).float().to(device)
        else:
            self.state_mean = None
            self.state_std = None
            self.normalize_states = False

        solver_path = "kuhn_poker_solver_cfr_plus.pkl"
        with open(solver_path, "rb") as f:
            print(f"Loading expert solver from {solver_path}")
            self.solver = pickle.load(f)

        print(f"✅ ARDTEvaluator initialized: Environment={env_name}, State dim={self.state_dim}, Action dim={self.act_dim}")

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
        Factory method to create evaluator from model parameters.
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
        
    def _get_env(self):
        if hasattr(self.env_instance, 'reset'):
            return self.env_instance
        try:
            return self.env_instance()
        except TypeError:
            return self.env_instance

    def _get_cfr_action(self, env, solver):
        info_set = f"{env.opponent_card}{env.history}"
        policy = solver.average_policy()
        strategy = policy.policy_for_key(info_set)
        return np.random.choice([0, 1], p=strategy)

    def _normal_case_env_step(self, action: Any, env) -> Tuple:
        """Normal environment step with no-action fix"""
        if isinstance(action, (np.ndarray, torch.Tensor)) and action.size > 1:
            action = int(np.argmax(action))
            if action == 2:  # no-action placeholder
                action = 0
        return env.step(action)

    def _worst_case_env_step(self, action: Any, env) -> Tuple:
        """Worst-case environment step with CFR opponent"""
        if isinstance(action, (np.ndarray, torch.Tensor)) and action.size > 1:
            action = int(np.argmax(action))
            if action == 2:  # no-action placeholder
                action = 0

        new_state, reward, terminated, truncated, info = env.step(action)

        if not (terminated or truncated) and env.current_player == 1:
            opponent_action = self._get_cfr_action(env, self.solver)
            new_state, reward_opp, terminated_opp, truncated_opp, info_opp = env.step(opponent_action)
            reward += reward_opp
            terminated = terminated_opp
            truncated = truncated_opp
            info.update(info_opp)
            info["opponent_action"] = opponent_action

        return new_state, reward, terminated, truncated, info

    def evaluate_single_episode(self, model: torch.nn.Module, target_return: float, step_fn: Callable, debug: bool = False) -> tuple[float, list[int]]:
        env = self._get_env()
        env_state = env.reset()
        state = env_state[0] if isinstance(env_state, tuple) else env_state

        episode_return = 0.0
        episode_actions = []

        states = torch.zeros(1, self.max_ep_len, self.state_dim, device=self.device)
        actions = torch.zeros(1, self.max_ep_len, self.act_dim, device=self.device)
        returns_to_go = torch.zeros(1, self.max_ep_len, 1, device=self.device)
        rewards = torch.zeros(1, self.max_ep_len, 1, device=self.device)

        model.eval()

        for step in range(self.max_ep_len):
            state_tensor = torch.from_numpy(state.astype(np.float32)).to(self.device)
            if self.normalize_states and self.state_mean is not None:
                state_tensor = (state_tensor - self.state_mean) / self.state_std
            states[:, step, :] = state_tensor

            returns_to_go[:, step, 0] = (target_return - episode_return) * self.scale
            seq_len = step + 1
            timesteps = torch.arange(seq_len, device=self.device).unsqueeze(0)
            action_sequence = torch.zeros(1, seq_len, self.act_dim, device=self.device)
            if step > 0:
                action_sequence[:, 1:, :] = actions[:, :step, :]

            attention_mask = torch.ones(1, seq_len, dtype=torch.bool, device=self.device)

            with torch.no_grad():
                try:
                    _, action_preds, _ = model(
                        states=states[:, :seq_len, :],
                        actions=action_sequence,
                        rewards=rewards[:, :seq_len, :],
                        returns_to_go=returns_to_go[:, :seq_len, :],
                        timesteps=timesteps,
                        attention_mask=attention_mask
                    )
                    action_logits = action_preds[0, -1, :]
                    action_probs = F.softmax(action_logits, dim=0)
                    action_idx = Categorical(action_probs).sample().item()
                except Exception as e:
                    print(f"Model forward pass failed: {e}")
                    action_idx = 0

            # Store action safely
            store_action = action_idx if action_idx < 2 else 0
            action_onehot = torch.zeros(self.act_dim, device=self.device)
            action_onehot[store_action] = 1.0
            actions[:, step, :] = action_onehot
            episode_actions.append(store_action)

            next_state, reward, terminated, truncated, _ = step_fn(action_idx, env)
            episode_return += reward
            rewards[:, step, 0] = reward

            if terminated or truncated:
                break
            state = next_state

        return episode_return, episode_actions

    def create_eval_function(self, case_type: str = "normal", debug: bool = False) -> Callable:
        step_fn = self._worst_case_env_step if case_type == "worst" else self._normal_case_env_step

        def eval_fn(model: torch.nn.Module, target_return: float, num_episodes: int = 50):
            model.eval()
            episode_returns = []
            episode_actions = []
            with torch.no_grad():
                for ep in tqdm(range(num_episodes), desc=f"Evaluating R={target_return} ({case_type.capitalize()})"):
                    ep_return, actions = self.evaluate_single_episode(model, target_return, step_fn, debug=debug and ep < 3)
                    episode_returns.append(ep_return)
                    episode_actions.append(actions)
            return episode_returns, episode_actions

        return eval_fn

    def comprehensive_model_evaluation(self, minimax_model: torch.nn.Module, method: str = "minimax",
                                       target_returns: List[float] = None, num_episodes_per_target: int = 1,
                                       save_path: str = "evaluation_results") -> Dict:

        if target_returns is None:
            target_returns = [-2.0, -1.0, 0.0, 1.0, 2.0]

        results = {'minimax_normal': {}, 'minimax_worst': {}}

        eval_fn_normal = self.create_eval_function("normal")
        eval_fn_worst = self.create_eval_function("worst")

        for target in target_returns:
            returns, _ = eval_fn_normal(minimax_model, target, num_episodes_per_target)
            returns = [0 if r is None else r for r in returns]  # fix any placeholder None
            results['minimax_normal'][f'target_{target}'] = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'returns': returns
            }

            returns, _ = eval_fn_worst(minimax_model, target, num_episodes_per_target)
            returns = [0 if r is None else r for r in returns]
            results['minimax_worst'][f'target_{target}'] = {
                'mean': np.mean(returns),
                'std': np.std(returns),
                'returns': returns
            }

        # Save & plot
        self._save_results(results, save_path, method)
        self._plot_comparison_results(results, target_returns, save_path, method)
        if self.env_name == "kuhn_poker":
            self._plot_kuhn_poker_results(results, target_returns, save_path, method)

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
        
        # Kuhn Poker Nash Equilibrium value for Player 0
        NASH_EQUILIBRIUM_RETURN = 1/18 
        
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
        #ax.plot(target_returns, target_returns, '--', color='gray', label='Ideal Performance (Target = Achieved)', linewidth=2)

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