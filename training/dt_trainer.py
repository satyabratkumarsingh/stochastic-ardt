from data_class.trajectory import Trajectory
import torch 
import pickle
from pathlib import Path
import gym
import numpy as np
import yaml
from tqdm import tqdm
import os
import torch.nn.functional as F
from core_models.decision_transformer.decision_transformer import DecisionTransformer
from core_models.dataset.ardt_dataset import ARDTDataset
from offline_setup.base_offline_env import BaseOfflineEnv
from utils.trajectory_utils import get_relabeled_trajectories, get_action_dim
from typing import Dict, List, Optional, Tuple
from utils.saved_names import dt_model_name


class DecisionTransformerTrainer:

    def __init__(self, seed, game_name, config, n_cpu = 1, device = 'cpu'):
        self.seed = seed
        self.game_name = game_name
        config = yaml.safe_load(Path(config).read_text())
        self.dt_train_args = config
        self.n_cpu = n_cpu
        self.device = device
    
    def _create_model_and_optimizer(self, obs_size: int, action_size: int, action_type: str, 
                                   horizon: int, effective_max_ep_len: int) -> Tuple[torch.nn.Module, torch.optim.Optimizer]:
        """Create and return a fresh DT model and optimizer"""
        dt_model = DecisionTransformer(
            state_dim=obs_size,
            act_dim=action_size,
            hidden_size=32,
            max_length=horizon,
            max_ep_len=effective_max_ep_len,
            action_tanh=(action_type == 'continuous'),
            rtg_seq=True
        ).to(self.device)
        
        dt_optimizer = torch.optim.AdamW(
            dt_model.parameters(), 
            lr=self.dt_train_args.get('model_lr', 1e-4),
            weight_decay=self.dt_train_args.get('model_wd', 1e-4)
        )
        
        return dt_model, dt_optimizer

    def _train_single_model(self, relabeled_trajs: List, use_minimax_returns: bool, 
                           obs_size: int, action_size: int, action_type: str, 
                           horizon: int, effective_max_ep_len: int, model_suffix: str) -> Tuple[torch.nn.Module, List]:
        """Train a single DT model with specified return type"""
        
        print(f"\n--- Training DT Model ({model_suffix}) ---")
        print(f"Using minimax returns: {use_minimax_returns}")
        
        # Create fresh model and optimizer
        dt_model, dt_optimizer = self._create_model_and_optimizer(
            obs_size, action_size, action_type, horizon, effective_max_ep_len
        )

        # Create dataset with specified return type
        dt_dataset = ARDTDataset(
            trajs=relabeled_trajs,
            horizon=horizon,
            gamma=self.dt_train_args.get('gamma', 0.99),
            act_type=action_type,
            use_minimax_returns=use_minimax_returns,
        )
        
        dt_dataloader = torch.utils.data.DataLoader(
            dt_dataset,
            batch_size=self.dt_train_args.get('batch_size', 256),
            num_workers=self.n_cpu
        )

        # Training loop
        dt_model.train()
        training_losses = []
        
        num_epochs = self.dt_train_args.get('dt_epochs', 10)
        for epoch in range(num_epochs):
            total_dt_loss = 0
            batch_count = 0
            
            pbar = tqdm(dt_dataloader, desc=f"Epoch {epoch+1}/{num_epochs} ({model_suffix})")
            for batch_idx, batch_data in enumerate(pbar):
                obs, acts, adv_acts, returns_to_go, rewards, seq_len = [d.to(self.device) for d in batch_data]
                
                dt_optimizer.zero_grad()

                batch_size, horizon_len = obs.shape[0], obs.shape[1]
                timesteps = torch.arange(horizon_len, device=self.device).repeat(batch_size, 1)
                attention_mask = (torch.arange(horizon_len, device=self.device)[None, :] < seq_len[:, None]).bool()

                # Forward pass
                state_preds, action_preds_tensor, return_preds = dt_model(
                    states=obs, 
                    actions=acts, 
                    returns_to_go=returns_to_go.unsqueeze(-1),
                    rewards=rewards.unsqueeze(-1),
                    timesteps=timesteps, 
                    attention_mask=attention_mask
                )

                # Calculate loss
                valid_mask = attention_mask
                action_preds_flat = action_preds_tensor[valid_mask]

                if action_type == 'discrete':
                    actions_target_flat = acts[valid_mask]
                    if actions_target_flat.ndim > 1 and actions_target_flat.shape[-1] > 1:
                        actions_target_flat = torch.argmax(actions_target_flat, dim=-1)
                    if action_preds_flat.numel() == 0:
                        loss = torch.tensor(0.0, device=self.device)
                    else:
                        loss = torch.nn.functional.cross_entropy(
                            action_preds_flat, actions_target_flat.long(), reduction='mean'
                        )
                else:
                    actions_target_flat = acts[valid_mask]
                    if action_preds_flat.numel() == 0:
                        loss = torch.tensor(0.0, device=self.device)
                    else:
                        loss = torch.nn.functional.mse_loss(
                            action_preds_flat, actions_target_flat, reduction='mean'
                        )

                # Backward pass
                loss.backward()
                
                # Gradient clipping
                #torch.nn.utils.clip_grad_norm_(dt_model.parameters(), max_norm=1.0)
                
                dt_optimizer.step()
                total_dt_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            # Calculate epoch statistics
            if batch_count > 0:
                avg_loss = total_dt_loss / batch_count
                training_losses.append(avg_loss)
                print(f"Epoch {epoch+1} ({model_suffix}) Avg Loss: {avg_loss:.6f}")
            else:
                print(f"Epoch {epoch+1} ({model_suffix}): No valid batches processed")

        return dt_model, training_losses

    def train(self, relabeled_trajs: List, method: str) -> torch.nn.Module:
        """
        Train Decision Transformer model with minimax returns only
        
        Returns:
            Trained minimax model
        """
        
        print(f"\n=== Training Decision Transformer Model ===")
        
        # Extract model parameters from first trajectory
        first_traj = relabeled_trajs[0]
        obs_size = len(first_traj.obs[0])
        action_size = get_action_dim(first_traj.actions)
        
        # Determine action type
        sample_action = first_traj.actions[0]
        if isinstance(sample_action, (list, np.ndarray)) and len(sample_action) > 1:
            if all(x in [0.0, 1.0] for x in sample_action) and sum(sample_action) == 1.0:
                action_type = 'discrete'
            else:
                action_type = 'continuous'
        else:
            action_type = 'continuous'

        # Calculate effective parameters
        horizon = self.dt_train_args.get('context_size', 5)
        max_ep_len_from_data = max([len(t.obs) for t in relabeled_trajs]) if relabeled_trajs else 100
        effective_max_ep_len = max(max_ep_len_from_data, horizon)

        print(f"Model config: obs_size={obs_size}, action_size={action_size}, action_type={action_type}")
        print(f"Horizon: {horizon}, Max episode length: {effective_max_ep_len}")

        # Train model with minimax returns only
        print(f"\n🚀 Training Minimax Model")
        minimax_model, minimax_losses = self._train_single_model(
            relabeled_trajs=relabeled_trajs,
            use_minimax_returns=True,
            obs_size=obs_size,
            action_size=action_size,
            action_type=action_type,
            horizon=horizon,
            effective_max_ep_len=effective_max_ep_len,
            model_suffix="Minimax"
        )

        # Save minimax model
        self._save_model(minimax_model, minimax_losses, method, "minimax", {
            "obs_size": obs_size,
            "action_size": action_size,
            "action_type": action_type,
            "horizon": horizon,
            "effective_max_ep_len": effective_max_ep_len,
            "use_minimax_returns": True
        })

        print(f"\n✅ DT model training complete!")
        return minimax_model

    def _save_model(self, model: torch.nn.Module, training_losses: List, method: str, 
                   model_type: str, model_params: Dict):
        """Save a single model with appropriate naming"""
        
        # Create model path with suffix
        base_path = dt_model_name(seed=self.seed, game=self.game_name, method=method)
        
        # Add suffix before file extension
        if base_path.endswith('.pth'):
            model_path = base_path.replace('.pth', f'_{model_type}.pth')
        else:
            model_path = f"{base_path}_{model_type}.pth"
        
        # Create checkpoint
        checkpoint = {
            "model_state_dict": model.state_dict(),
            "training_losses": training_losses,
            "model_params": model_params,
            "model_type": model_type
        }
        
        # Save model
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(checkpoint, model_path)
        print(f"✅ {model_type.capitalize()} DT model saved to {model_path}")

    def load_model(self, method: str, model_type: str = "minimax") -> torch.nn.Module:
        """
        Load the minimax DT model
        
        Args:
            method: Training method used
            model_type: Should be "minimax" (only option now)
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

        dt_model = DecisionTransformer(
            state_dim=model_params["obs_size"],
            act_dim=model_params["action_size"],
            hidden_size=32,
            max_length=model_params["horizon"],
            max_ep_len=model_params["effective_max_ep_len"],
            action_tanh=(model_params["action_type"] == 'continuous'),
            rtg_seq=True
        ).to(self.device)
        
        # Load state dict
        dt_model.load_state_dict(checkpoint["model_state_dict"])
        
        print(f"✅ Loaded {model_type} DT model from {model_path}")
        return dt_model

    def train_and_save(self, relabeled_trajs: List, method: str) -> torch.nn.Module:
        """
        Train and save the minimax model
        """
        return self.train(relabeled_trajs, method)