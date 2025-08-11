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


class ARDTEvaluatorFixed:
    """
    Fixed version of the ARDT Evaluator with comprehensive debugging and proper RTG handling.
    """
    
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
        
        # FIX 1: Ensure state statistics are properly handled
        self.state_mean = torch.from_numpy(state_mean).float().to(device)
        self.state_std = torch.from_numpy(state_std).float().to(device)
        
        # Prevent division by zero in normalization
        self.state_std = torch.clamp(self.state_std, min=1e-8)
        
        print(f"✅ Evaluator initialized for {env_name}")
        print(f"   State dim: {state_dim}, Action dim: {act_dim}")
        print(f"   Scale factor: {scale}")
        print(f"   State mean: {self.state_mean.cpu().numpy()[:5]}...")
        print(f"   State std: {self.state_std.cpu().numpy()[:5]}...")

    def _decode_kuhn_state(self, state: np.ndarray) -> str:
        """Enhanced Kuhn Poker state decoder for better debugging."""
        if self.env_name != "kuhn_poker":
            return "N/A"
        
        if len(state) >= 12:
            # Find all active positions (value = 1.0)
            active_positions = np.where(np.abs(state - 1.0) < 1e-6)[0]
            
            if len(active_positions) > 0:
                pos = active_positions[0]
                if pos < 3:
                    return f"P0_Card{pos}"
                elif pos < 6:
                    return f"P1_Card{pos-3}"
                elif pos < 12:
                    history_pos = pos - 6
                    return f"History_Pos{history_pos}"
            
            # If no clear active position, show the non-zero elements
            non_zero = np.where(np.abs(state) > 1e-6)[0]
            return f"NonZero_at_{non_zero.tolist()}"
        
        return f"Unknown_state_len{len(state)}"

    def debug_model_response(self, model: torch.nn.Module, model_type: str = 'dt') -> Dict:
        """
        FIX 2: Enhanced model response debugging to identify RTG sensitivity issues.
        """
        print("🔍 Enhanced Model Response Debug...")
        
        model.eval()
        with torch.no_grad():
            # Test with a variety of different inputs
            dummy_state = torch.zeros(1, 1, self.state_dim, device=self.device)
            dummy_action = torch.zeros(1, 1, self.act_dim, device=self.device)
            dummy_timestep = torch.zeros(1, 1, device=self.device, dtype=torch.long)
            
            rtg_values = [0.0, 1.0, 2.0, -1.0, -2.0]
            outputs = {}
            
            for rtg_val in rtg_values:
                rtg_tensor = torch.tensor([[[rtg_val * self.scale]]], device=self.device)
                
                try:
                    if model_type == 'dt':
                        _, action_pred, _ = model(
                            states=dummy_state,
                            actions=dummy_action,
                            returns_to_go=rtg_tensor,
                            timesteps=dummy_timestep.squeeze(-1)
                        )
                    else:
                        _, action_pred, _ = model(
                            states=dummy_state,
                            actions=dummy_action,
                            rewards=rtg_tensor
                        )
                    
                    outputs[rtg_val] = action_pred[0, -1, :].cpu().numpy()
                    
                except Exception as e:
                    print(f"❌ Error with RTG {rtg_val}: {e}")
                    outputs[rtg_val] = None
            
            # Analyze sensitivity
            valid_outputs = {k: v for k, v in outputs.items() if v is not None}
            
            if len(valid_outputs) < 2:
                print("❌ Model failed to produce valid outputs")
                return {'error': 'model_failed'}
            
            # Check variance across different RTG values
            output_matrix = np.array(list(valid_outputs.values()))
            rtg_sensitivity = np.var(output_matrix, axis=0).max()
            
            print(f"   RTG Sensitivity (max variance): {rtg_sensitivity:.6f}")
            
            # Check for identical outputs
            identical_outputs = len(set([tuple(v) for v in valid_outputs.values()])) == 1
            
            if identical_outputs:
                print("❌ CRITICAL BUG: Model produces identical outputs for all RTG values!")
                print("   This means the model is completely ignoring RTG inputs.")
            else:
                print("✅ Model responds differently to different RTG values")
            
            # Show action probabilities for each RTG
            print("\n📊 Action probabilities by RTG:")
            for rtg_val, logits in valid_outputs.items():
                probs = F.softmax(torch.from_numpy(logits), dim=0).numpy()
                print(f"   RTG {rtg_val:5.1f}: {probs}")
            
            return {
                'rtg_sensitivity': float(rtg_sensitivity),
                'identical_outputs': bool(identical_outputs),
                'outputs': {str(k): v.tolist() if v is not None else None for k, v in outputs.items()},
                'model_responds_to_rtg': not identical_outputs and rtg_sensitivity > 1e-6
            }

    def evaluate_single_episode_fixed(
        self, 
        model: torch.nn.Module, 
        target_return: float, 
        model_type: str, 
        debug: bool = False,
        temperature: float = 1.0,
        use_stochastic_sampling: bool = True
    ) -> Tuple[float, List[int]]:
        """
        FIX 3: Completely rewritten episode evaluation with proper RTG handling.
        """
        
        # Reset environment
        state = self.env_instance.reset()
        if isinstance(state, tuple):
            state, _ = state
        
        episode_return = 0.0
        episode_actions = []
        
        # Scale target return
        scaled_target_return = target_return * self.scale
        
        # Track the sequence for debugging
        rtg_sequence = [scaled_target_return]
        reward_sequence = []
        state_sequence = [state.copy()]
        
        if debug:
            print(f"\n🎮 Starting episode with target={target_return:.2f} (scaled={scaled_target_return:.3f})")
            print(f"   Initial state: {self._decode_kuhn_state(state)}")
        
        model.eval()
        with torch.no_grad():
            
            for t in range(self.max_ep_len):
                
                # FIX 4: Proper tensor preparation for history
                # Create tensors for the current history length
                seq_len = t + 1
                
                states_tensor = torch.zeros(1, seq_len, self.state_dim, device=self.device)
                actions_tensor = torch.zeros(1, seq_len, self.act_dim, device=self.device)
                rtg_tensor = torch.zeros(1, seq_len, 1, device=self.device)
                timesteps_tensor = torch.arange(seq_len, device=self.device).unsqueeze(0)
                
                # Fill in the history
                for i in range(seq_len):
                    # State
                    states_tensor[0, i] = torch.from_numpy(state_sequence[i]).float()
                    
                    # RTG
                    rtg_tensor[0, i, 0] = rtg_sequence[i]
                    
                    # Actions (for previous timesteps only)
                    if i < len(episode_actions):
                        action_one_hot = F.one_hot(
                            torch.tensor(episode_actions[i]), 
                            num_classes=self.act_dim
                        ).float()
                        actions_tensor[0, i] = action_one_hot
                
                # FIX 5: Proper state normalization
                if self.normalize_states:
                    states_normalized = (states_tensor - self.state_mean.unsqueeze(0).unsqueeze(0)) / self.state_std.unsqueeze(0).unsqueeze(0)
                else:
                    states_normalized = states_tensor
                
                # FIX 6: Model forward pass with proper error handling
                try:
                    if model_type == 'dt':
                        _, action_preds, _ = model(
                            states=states_normalized,
                            actions=actions_tensor,
                            returns_to_go=rtg_tensor,
                            timesteps=timesteps_tensor
                        )
                    elif model_type == 'bc':
                        _, action_preds, _ = model(
                            states=states_normalized,
                            actions=actions_tensor,
                            rewards=rtg_tensor  # Use RTG as reward proxy for BC
                        )
                    else:
                        raise ValueError(f"Unknown model type: {model_type}")
                    
                except Exception as e:
                    print(f"❌ Model forward pass error at step {t}: {e}")
                    print(f"   States shape: {states_normalized.shape}")
                    print(f"   Actions shape: {actions_tensor.shape}")
                    print(f"   RTG shape: {rtg_tensor.shape}")
                    print(f"   Timesteps shape: {timesteps_tensor.shape}")
                    raise e
                
                # FIX 7: Improved action selection with temperature and proper sampling
                action_logits = action_preds[0, -1, :]  # Last timestep prediction
                
                if use_stochastic_sampling:
                    # Apply temperature for exploration
                    action_logits_temp = action_logits / temperature
                    action_probs = F.softmax(action_logits_temp, dim=0)
                    
                    # Sample action
                    action_dist = Categorical(action_probs)
                    action = action_dist.sample().item()
                else:
                    # Deterministic selection
                    action = torch.argmax(action_logits).item()
                
                episode_actions.append(action)
                
                if debug:
                    action_probs_display = F.softmax(action_logits, dim=0)
                    print(f"\n   Step {t}:")
                    print(f"     Current RTG: {rtg_tensor[0, -1, 0].item():.3f} (unscaled: {rtg_tensor[0, -1, 0].item()/self.scale:.3f})")
                    print(f"     State: {self._decode_kuhn_state(state)}")
                    print(f"     Action logits: {action_logits.cpu().numpy()}")
                    print(f"     Action probs: {action_probs_display.cpu().numpy()}")
                    print(f"     Selected action: {action} ({'bet/call' if action == 1 else 'check/fold'})")
                
                # Environment step
                step_result = self.env_instance.step(action)
                
                if len(step_result) == 4:
                    next_state, reward, done, info = step_result
                    terminated = done
                    truncated = False
                else:
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated
                
                episode_return += reward
                reward_sequence.append(reward)
                
                if debug:
                    print(f"     Environment response:")
                    print(f"       Next state: {self._decode_kuhn_state(next_state)}")
                    print(f"       Reward: {reward}")
                    print(f"       Done: {done}")
                    print(f"       Episode return so far: {episode_return}")
                
                if done:
                    break
                
                # FIX 8: Correct RTG update for next timestep
                # RTG should decrease by the reward we just received
                new_rtg = rtg_sequence[-1] - reward * self.scale
                rtg_sequence.append(new_rtg)
                state_sequence.append(next_state.copy())
                state = next_state
                
                if debug:
                    print(f"       Updated RTG for next step: {new_rtg:.3f} (unscaled: {new_rtg/self.scale:.3f})")
        
        if debug:
            print(f"\n🏁 Episode completed:")
            print(f"   Target return: {target_return:.3f}")
            print(f"   Achieved return: {episode_return:.3f}")
            print(f"   Error: {abs(episode_return - target_return):.3f}")
            print(f"   Episode length: {len(episode_actions)}")
            print(f"   Action sequence: {episode_actions}")
            
            # Verify RTG updates were correct
            expected_final_rtg = scaled_target_return - sum(reward_sequence) * self.scale
            actual_final_rtg = rtg_sequence[-1] if len(rtg_sequence) > len(reward_sequence) else "N/A"
            
            if actual_final_rtg != "N/A":
                rtg_error = abs(expected_final_rtg - actual_final_rtg)
                if rtg_error > 1e-6:
                    print(f"   ⚠️ RTG update error: expected {expected_final_rtg:.6f}, got {actual_final_rtg:.6f}")
                else:
                    print(f"   ✅ RTG updates correct")
        
        return episode_return, episode_actions

    def create_eval_function_fixed(
        self, 
        model_type: str, 
        debug: bool = False,
        temperature: float = 1.0,
        use_stochastic_sampling: bool = True
    ) -> Callable:
        """
        FIX 9: Enhanced evaluation function with better error detection.
        """
        def eval_fn(model: torch.nn.Module, target_return: float, num_episodes: int = 50):
            model.eval()
            episode_returns = []
            episode_actions = []
            
            print(f"\n🎯 Evaluating {model_type.upper()} model with target return: {target_return}")
            
            with torch.no_grad():
                for ep in tqdm(range(num_episodes), desc=f"Episodes (target={target_return})"):
                    
                    # Debug first few episodes in detail
                    episode_debug = debug and ep < 3
                    
                    try:
                        episode_return, actions = self.evaluate_single_episode_fixed(
                            model=model,
                            target_return=target_return,
                            model_type=model_type,
                            debug=episode_debug,
                            temperature=temperature,
                            use_stochastic_sampling=use_stochastic_sampling
                        )
                        
                        episode_returns.append(episode_return)
                        episode_actions.append(actions)
                        
                    except Exception as e:
                        print(f"❌ Episode {ep} failed: {e}")
                        continue
                    
                    # Early detection of identical returns
                    if ep >= 2:
                        unique_returns = len(set(episode_returns))
                        if unique_returns == 1:
                            print(f"⚠️ First {ep+1} episodes all returned {episode_returns[0]}")
                            if ep >= 4:  # Stop early if pattern is very clear
                                print("🛑 Stopping early due to identical returns pattern")
                                break
                        elif unique_returns <= 2 and ep >= 9:
                            print(f"⚠️ Only {unique_returns} unique returns in {ep+1} episodes")
            
            # Final analysis
            if len(episode_returns) == 0:
                print("❌ No successful episodes!")
                return [], []
            
            mean_return = np.mean(episode_returns)
            std_return = np.std(episode_returns)
            unique_returns = len(set(episode_returns))
            
            print(f"📊 Results for target {target_return}:")
            print(f"   Episodes: {len(episode_returns)}")
            print(f"   Mean return: {mean_return:.3f}")
            print(f"   Std return: {std_return:.3f}")
            print(f"   Unique returns: {unique_returns}")
            print(f"   Error from target: {abs(mean_return - target_return):.3f}")
            
            # Diagnostic warnings
            if unique_returns == 1:
                print("❌ CRITICAL: All episodes have identical returns!")
            elif unique_returns < 3 and len(episode_returns) > 10:
                print(f"⚠️ WARNING: Very low return diversity ({unique_returns} unique)")
            elif std_return < 0.01:
                print("⚠️ WARNING: Extremely low return variance")
            else:
                print("✅ Return distribution looks reasonable")
            
            return episode_returns, episode_actions
        
        return eval_fn

    def comprehensive_debug_suite(self, model: torch.nn.Module, model_type: str = 'dt') -> Dict:
        """
        FIX 10: Comprehensive debugging suite to identify all possible issues.
        """
        print("\n" + "="*60)
        print("🔧 COMPREHENSIVE DEBUG SUITE")
        print("="*60)
        
        debug_results = {}
        
        # Test 1: Model Response Analysis
        print("\n1️⃣ Model Response Analysis")
        debug_results['model_response'] = self.debug_model_response(model, model_type)
        
        # Test 2: Single Episode Deep Dive
        print("\n2️⃣ Single Episode Deep Dive")
        try:
            test_targets = [0.0, 1.0, -1.0]
            for target in test_targets:
                print(f"\n   Testing target return: {target}")
                episode_return, actions = self.evaluate_single_episode_fixed(
                    model=model,
                    target_return=target,
                    model_type=model_type,
                    debug=True,
                    temperature=1.0,
                    use_stochastic_sampling=True
                )
                
                debug_results[f'single_episode_target_{target}'] = {
                    'return': float(episode_return),
                    'actions': actions,
                    'length': len(actions)
                }
                
        except Exception as e:
            print(f"❌ Single episode test failed: {e}")
            debug_results['single_episode_error'] = str(e)
        
        # Test 3: Action Distribution Analysis
        print("\n3️⃣ Action Distribution Analysis")
        try:
            eval_fn = self.create_eval_function_fixed(model_type, debug=False, use_stochastic_sampling=True)
            _, all_actions = eval_fn(model, target_return=0.0, num_episodes=20)
            
            flat_actions = [action for episode in all_actions for action in episode]
            if flat_actions:
                action_counts = {}
                for action in flat_actions:
                    action_counts[action] = action_counts.get(action, 0) + 1
                
                total_actions = len(flat_actions)
                action_probs = {action: count/total_actions for action, count in action_counts.items()}
                
                debug_results['action_distribution'] = {
                    'counts': action_counts,
                    'probabilities': action_probs,
                    'total_actions': total_actions,
                    'entropy': float(-sum(p * np.log(p + 1e-10) for p in action_probs.values()))
                }
                
                print(f"   Total actions: {total_actions}")
                for action, prob in action_probs.items():
                    action_name = "bet/call" if action == 1 else "check/fold"
                    print(f"   Action {action} ({action_name}): {prob:.3f}")
                    
                entropy = debug_results['action_distribution']['entropy']
                max_entropy = np.log(len(action_probs))
                normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
                
                print(f"   Action entropy: {entropy:.3f} (normalized: {normalized_entropy:.3f})")
                
                if normalized_entropy < 0.1:
                    print("   ❌ CRITICAL: Extremely low action diversity!")
                elif normalized_entropy < 0.3:
                    print("   ⚠️ WARNING: Low action diversity")
                else:
                    print("   ✅ Reasonable action diversity")
            
        except Exception as e:
            print(f"❌ Action distribution test failed: {e}")
            debug_results['action_distribution_error'] = str(e)
        
        # Test 4: Return Conditioning Test
        print("\n4️⃣ Return Conditioning Test")
        try:
            eval_fn = self.create_eval_function_fixed(model_type, debug=False, use_stochastic_sampling=False)
            
            test_targets = [-2.0, -1.0, 0.0, 1.0, 2.0]
            conditioning_results = {}
            
            for target in test_targets:
                returns, _ = eval_fn(model, target, num_episodes=5)
                if returns:
                    mean_return = np.mean(returns)
                    conditioning_results[target] = {
                        'mean_return': float(mean_return),
                        'error': float(abs(mean_return - target)),
                        'returns': [float(r) for r in returns]
                    }
            
            debug_results['return_conditioning'] = conditioning_results
            
            # Check if model actually conditions on returns
            if len(conditioning_results) >= 2:
                returns_by_target = [result['mean_return'] for result in conditioning_results.values()]
                return_variance = np.var(returns_by_target)
                
                print(f"   Return variance across targets: {return_variance:.6f}")
                
                if return_variance < 1e-6:
                    print("   ❌ CRITICAL: Model does not condition on target returns!")
                elif return_variance < 0.01:
                    print("   ⚠️ WARNING: Weak return conditioning")
                else:
                    print("   ✅ Model shows return conditioning behavior")
                
                debug_results['return_conditioning']['variance_across_targets'] = float(return_variance)
                debug_results['return_conditioning']['conditions_on_returns'] = return_variance > 1e-6
            
        except Exception as e:
            print(f"❌ Return conditioning test failed: {e}")
            debug_results['return_conditioning_error'] = str(e)
        
        # Summary
        print("\n" + "="*60)
        print("🏁 DEBUG SUITE SUMMARY")
        print("="*60)
        
        issues_found = []
        
        if not debug_results.get('model_response', {}).get('model_responds_to_rtg', False):
            issues_found.append("Model ignores RTG inputs")
        
        if debug_results.get('return_conditioning', {}).get('conditions_on_returns', True) == False:
            issues_found.append("No return conditioning")
        
        action_entropy = debug_results.get('action_distribution', {}).get('entropy', 1.0)
        if action_entropy < 0.1:
            issues_found.append("Extremely biased actions")
        
        if issues_found:
            print("❌ ISSUES DETECTED:")
            for issue in issues_found:
                print(f"   • {issue}")
        else:
            print("✅ No major issues detected in debug suite")
        
        debug_results['summary'] = {
            'issues_found': issues_found,
            'has_major_issues': len(issues_found) > 0
        }
        
        return debug_results


# Additional utility functions for comprehensive testing

def test_model_architecture(model: torch.nn.Module, device: str = 'cpu') -> Dict:
    """
    Test if the model architecture is properly set up for return conditioning.
    """
    print("\n🏗️ Testing Model Architecture...")
    
    results = {}
    
    # Check if model has expected attributes/methods
    expected_attributes = ['forward', 'state_dict']
    missing_attributes = [attr for attr in expected_attributes if not hasattr(model, attr)]
    
    if missing_attributes:
        print(f"❌ Missing model attributes: {missing_attributes}")
        results['missing_attributes'] = missing_attributes
    else:
        print("✅ Model has expected attributes")
    
    # Test model parameter count
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"   Total parameters: {total_params:,}")
    print(f"   Trainable parameters: {trainable_params:,}")
    
    results.update({
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_reasonable': 1000 < total_params < 10_000_000
    })
    
    # Test gradient flow
    try:
        model.train()
        dummy_input = {
            'states': torch.randn(1, 5, 12, device=device),
            'actions': torch.randn(1, 5, 2, device=device),
            'returns_to_go': torch.randn(1, 5, 1, device=device),
            'timesteps': torch.arange(5, device=device).unsqueeze(0)
        }
        
        output = model(**dummy_input)
        if isinstance(output, tuple) and len(output) >= 2:
            loss = output[1].mean()  # Use action predictions
            loss.backward()
            
            # Check if gradients exist
            has_gradients = any(p.grad is not None for p in model.parameters())
            if has_gradients:
                print("✅ Gradient flow working")
                results['gradient_flow'] = True
            else:
                print("❌ No gradients found")
                results['gradient_flow'] = False
        else:
            print("❌ Unexpected model output format")
            results['output_format_error'] = True
            
    except Exception as e:
        print(f"❌ Gradient test failed: {e}")
        results['gradient_test_error'] = str(e)
    
    return results


def comprehensive_model_validation(
    model: torch.nn.Module,
    evaluator: ARDTEvaluatorFixed,
    model_type: str = 'dt'
) -> Dict:
    """
    Run the complete validation suite on a model.
    """
    print("\n" + "🚀 " + "="*58)
    print("🚀 COMPREHENSIVE MODEL VALIDATION SUITE")
    print("🚀 " + "="*58)
    
    validation_results = {}
    
    # Architecture test
    validation_results['architecture'] = test_model_architecture(model, evaluator.device)
    
    # Debug suite
    validation_results['debug_suite'] = evaluator.comprehensive_debug_suite(model, model_type)
    
    # Performance evaluation
    print("\n5️⃣ Performance Evaluation")
    try:
        eval_fn = evaluator.create_eval_function_fixed(
            model_type=model_type,
            debug=False,
            temperature=1.0,
            use_stochastic_sampling=True
        )
        
        test_targets = [-1.0, 0.0, 1.0]
        performance_results = {}
        
        for target in test_targets:
            returns, actions = eval_fn(model, target, num_episodes=10)
            if returns:
                performance_results[f'target_{target}'] = {
                    'mean_return': float(np.mean(returns)),
                    'std_return': float(np.std(returns)),
                    'error_from_target': float(abs(np.mean(returns) - target)),
                    'num_episodes': len(returns)
                }
        
        validation_results['performance'] = performance_results
        
    except Exception as e:
        print(f"❌ Performance evaluation failed: {e}")
        validation_results['performance_error'] = str(e)
    
    # Final assessment
    print("\n" + "🎯 " + "="*58)
    print("🎯 FINAL ASSESSMENT")
    print("🎯 " + "="*58)
    
    issues = validation_results.get('debug_suite', {}).get('summary', {}).get('issues_found', [])
    
    if not issues:
        grade = "EXCELLENT ✅"
        recommendation = "Model appears to be working correctly!"
    elif len(issues) == 1:
        grade = "GOOD ⚠️"
        recommendation = f"Minor issue detected: {issues[0]}. May need attention."
    elif len(issues) <= 3:
        grade = "FAIR ⚠️"
        recommendation = f"Multiple issues detected: {', '.join(issues)}. Needs investigation."
    else:
        grade = "POOR ❌"
        recommendation = f"Serious issues detected: {', '.join(issues)}. Requires debugging."
    
    print(f"Overall Grade: {grade}")
    print(f"Recommendation: {recommendation}")
    
    validation_results['final_assessment'] = {
        'grade': grade,
        'recommendation': recommendation,
        'issues_count': len(issues),
        'issues': issues
    }
    
    return validation_results