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
    

    def train(self, relabeled_trajs: List, is_implicit = False) -> torch.nn.Module:
        
        print(f"\n=== Training Decision Transformer ===")
        use_minimax_returns = True
        dt_model_path = dt_model_name(seed=self.seed, game= self.game_name, is_implicit= is_implicit)    
        
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
        horizon = self.dt_train_args.get('context_size', 20)
        max_ep_len_from_data = max([len(t.obs) for t in relabeled_trajs]) if relabeled_trajs else 100
        effective_max_ep_len = max(max_ep_len_from_data, horizon)

        print(f"Model config: obs_size={obs_size}, action_size={action_size}, action_type={action_type}")
        print(f"Horizon: {horizon}, Max episode length: {effective_max_ep_len}")

        # Initialize model
        dt_model = DecisionTransformer(
            state_dim=obs_size,
            act_dim=action_size,
            hidden_size=self.dt_train_args.get('hidden_size', 64),
            max_length=horizon,
            max_ep_len=effective_max_ep_len,
            action_tanh=(action_type == 'continuous'),
            action_type=action_type,
            n_layer=self.dt_train_args.get('transformer_n_layer', 2),
            n_head=self.dt_train_args.get('transformer_n_head', 1),
            n_inner=self.dt_train_args.get('transformer_n_inner', 256),
            dropout=self.dt_train_args.get('transformer_dropout', 0.1)
        ).to(self.device)

        # Initialize optimizer
        dt_optimizer = torch.optim.AdamW(
            dt_model.parameters(), 
            lr=self.dt_train_args.get('model_lr', 1e-4),
            weight_decay=self.dt_train_args.get('model_wd', 1e-4)
        )

        # Create dataset and dataloader
        dt_dataset = ARDTDataset(
            trajs=relabeled_trajs,
            horizon=horizon,
            gamma=self.dt_train_args.get('gamma', 0.99),
            act_type=action_type,
            use_minimax_returns=use_minimax_returns,
        )
        
        dt_dataloader = torch.utils.data.DataLoader(
            dt_dataset,
            batch_size=self.dt_train_args.get('batch_size', 64),
            num_workers=self.n_cpu
        )

        # Training loop
        dt_model.train()
        training_losses = []
        
        num_epochs = self.dt_train_args.get('dt_epochs', 10)
        for epoch in range(num_epochs):
            total_dt_loss = 0
            batch_count = 0
            
            pbar = tqdm(dt_dataloader, desc=f" Epoch {epoch+1}/{num_epochs}")
            for batch_idx, batch_data in enumerate(pbar):
                obs, acts, adv_acts, minimax_ret, seq_len = [d.to(self.device) for d in batch_data]
                minimax_ret = (minimax_ret / self.dt_train_args.get('scale', 1.0))

                dt_optimizer.zero_grad()

                batch_size, horizon_len = obs.shape[0], obs.shape[1]
                timesteps = torch.arange(horizon_len, device=self.device).repeat(batch_size, 1)
                attention_mask = (torch.arange(horizon_len, device=self.device)[None, :] < seq_len[:, None]).bool()

                # Forward pass
                state_preds, action_preds_tensor, return_preds = dt_model(
                    states=obs, 
                    actions=acts, 
                    returns_to_go=minimax_ret.unsqueeze(-1),
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
                torch.nn.utils.clip_grad_norm_(dt_model.parameters(), max_norm=1.0)
                
                dt_optimizer.step()
                total_dt_loss += loss.item()
                batch_count += 1
                
                # Update progress bar
                pbar.set_postfix({'Loss': f'{loss.item():.6f}'})
            
            # Calculate epoch statistics
            if batch_count > 0:
                avg_loss = total_dt_loss / batch_count
                training_losses.append(avg_loss)
                print(f"Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
            else:
                print(f" Epoch {epoch+1}: No valid batches processed")

        # Save model
        checkpoint = {
            "model_state_dict": dt_model.state_dict(),
            "optimizer_state_dict": dt_optimizer.state_dict(),
            "training_losses": training_losses,
            "model_params": {
                "obs_size": obs_size,
                "action_size": action_size,
                "action_type": action_type,
                "horizon": horizon,
                "effective_max_ep_len": effective_max_ep_len
            }
        }
        
        torch.save(checkpoint, dt_model_path)
        print(f"✅ DT  model saved to {dt_model_path}")
        print(f"✅ DT training complete\n")

        return dt_model