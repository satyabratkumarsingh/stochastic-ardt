
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
from tqdm import tqdm
import json
import os
from typing import Dict, List, Optional, Tuple
from collections import namedtuple
import matplotlib.pyplot as plt
import yaml
from pathlib import Path
from core_models.dataset.ardt_dataset import ARDTDataset
from utils.trajectory_utils import  get_action_dim
from core_models.behaviour_cloning.behaviour_cloning import MLPBCModel
import torch.nn.functional as F
from utils.saved_names import behaviour_cloning_model_name

class BehaviourCloningTrainer:
   
    
    def __init__(self, seed, game_name, config, n_cpu = 1, device = 'cpu'):
        self.seed = seed
        self.game_name = game_name
        config = yaml.safe_load(Path(config).read_text())
        self.bc_train_args = config
        self.n_cpu = n_cpu
        self.device = device
        
    def compute_loss(self, predicted_actions, target_actions, seq_lengths):
        batch_size, max_seq_len, action_dim = predicted_actions.shape
        
        # Create mask
        mask = torch.arange(max_seq_len, device=self.device)[None, :] < seq_lengths[:, None]
        
        if self.action_type == 'continuous':
            # Use MSE loss
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, action_dim)
            loss_matrix = (predicted_actions - target_actions) ** 2
            masked_loss = loss_matrix * mask_expanded.float()
            loss = masked_loss.sum() / (mask_expanded.float().sum() + 1e-8)
            
        elif self.action_type == 'discrete':
            # Option 1: Use MSE loss for discrete actions (often works better)
            mask_expanded = mask.unsqueeze(-1).expand(-1, -1, action_dim)
            loss_matrix = (predicted_actions - target_actions) ** 2
            masked_loss = loss_matrix * mask_expanded.float()
            loss = masked_loss.sum() / (mask_expanded.float().sum() + 1e-8)
            
            # Option 2: If you want to keep cross-entropy, scale it up
            # loss = loss * 1000  # Scale up the loss
        
        return loss
    
    def train_step(self, batch) -> float:
        obs, actions, adv_actions, returns, seq_lengths = batch
        obs = obs.to(self.device)
        actions = actions.to(self.device)
        returns = returns.to(self.device)
        seq_lengths = seq_lengths.to(self.device)
        
        self.optimizer.zero_grad()

        # Forward pass
        _, predicted_actions, _ = self.model(obs, actions, returns, None, None)
       
        
        # Handle the target actions based on the actual data structure
        if actions.shape[1] == 1:
            # If we only have 1 timestep per sequence, just use that
            target_actions = actions[:, 0, :].unsqueeze(1)  # Shape: [batch_size, 1, action_dim]
        else:
            # Use the last valid action in each sequence
            batch_indices = torch.arange(obs.shape[0], device=self.device)
            target_indices = torch.clamp(seq_lengths - 1, 0, actions.shape[1] - 1)  # Clamp to valid range
            target_actions = actions[batch_indices, target_indices].unsqueeze(1)
        
        # Compute loss
        loss = self.compute_loss(
            predicted_actions,
            target_actions,
            torch.ones(predicted_actions.shape[0], device=self.device)  # All sequences have length 1
        )

        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def train(self, relabeled_trajs, is_implicit=False):
        print(f"\n=== Training behaviour cloning Transformer ===")
        bc_model_path = behaviour_cloning_model_name(seed=self.seed, game=self.game_name, is_implicit=is_implicit)    

        first_traj = relabeled_trajs[0]
        obs_size = len(first_traj.obs[0])
        action_size = get_action_dim(first_traj.actions)

        # Fixed action type detection
        sample_actions = []
        for traj in relabeled_trajs[:5]:  # Check multiple trajectories
            sample_actions.extend(traj.actions[:3])  # First few actions from each
        
        # Check if all actions are valid discrete vectors (one-hot or null)
        is_discrete = True
        for action in sample_actions:
            if not (isinstance(action, (list, np.ndarray)) and 
                   len(action) > 1 and
                   all(x in [0.0, 1.0] for x in action) and 
                   sum(action) in [0.0, 1.0]):  # Allow null actions (sum=0) or one-hot (sum=1)
                is_discrete = False
                break
        
        action_type = 'discrete' if is_discrete else 'continuous'
        self.action_type = action_type
        
        print(f"Detected action type: {action_type}")
        print(f"Sample actions: {sample_actions[:3]}")
        
        # Create model with max_length=1 for single-step BC
        self.model = MLPBCModel(
            state_dim=obs_size,
            act_dim=action_size,
            hidden_size=self.bc_train_args.get('hidden_size', 256),
            n_layer=self.bc_train_args.get('n_layer', 3),
            dropout=self.bc_train_args.get('dropout', 0.1),
            max_length=1  # Single-step BC
        ).to(self.device)
        
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), 
            lr=self.bc_train_args.get('model_lr', 1e-2),  # Increased from 1e-3
            weight_decay=self.bc_train_args.get('model_wd', 1e-4)
        )

        self.train_losses = []
        self.step = 0
        self.epoch = 0

        # Create dataset
        horizon = 1  # Single-step BC
        dataset = ARDTDataset(
            trajs=relabeled_trajs,
            horizon=horizon,
            gamma=self.bc_train_args.get('gamma', 0.99),
            act_type=action_type,
            epoch_len=self.bc_train_args.get('epoch_len', 10000),
            new_rewards=False, 
            use_minimax_returns=False 
        )
        
        bc_dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=self.bc_train_args.get('batch_size', 32),
            num_workers=self.n_cpu
        )
        
        num_epochs = self.bc_train_args.get('dt_epochs', 10)
        self.train_bc(bc_dataloader=bc_dataloader, bc_model_path=bc_model_path, num_epochs=num_epochs)
    
    def train_bc(self, bc_dataloader, bc_model_path, num_epochs: int, max_steps: Optional[int] = None):
        """
        Main training loop.
        
        Args:
            bc_dataloader: DataLoader for training data
            bc_model_path: Path to save the model
            num_epochs: number of epochs to train
            max_steps: maximum number of steps (optional)
        """
        print(f"Starting behavior cloning training for {num_epochs} epochs on {self.device}")
        print(f"Training objective: Learn state->action mapping")
        print(f"Model has {sum(p.numel() for p in self.model.parameters()):,} parameters")
        print(f"Action type: {self.action_type}")
        
        self.model.train()
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            epoch_losses = []
            
            # Training loop with progress bar
            pbar = tqdm(bc_dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
            
            for batch_idx, batch in enumerate(pbar):
                # Early stopping check if max_steps is specified
                if max_steps is not None and self.step >= max_steps:
                    print(f"Reached maximum steps ({max_steps}), stopping training.")
                    break
                
                # Training step
                loss = self.train_step(batch)
                epoch_losses.append(loss)
                self.train_losses.append(loss)
                
                # Update progress bar with better precision
                pbar.set_postfix({
                    'loss': f'{loss:.6f}',  # More decimal places
                    'avg_loss': f'{np.mean(epoch_losses[-100:]):.6f}',
                    'step': self.step
                })
                
                self.step += 1
            
            # Break if we hit max_steps during the epoch
            if max_steps is not None and self.step >= max_steps:
                break
            
            # End of epoch summary
            epoch_avg_loss = np.mean(epoch_losses)
            print(f"Epoch {epoch+1} completed. Average loss: {epoch_avg_loss:.4f}")
            
            # Save checkpoint every few epochs
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(bc_model_path)
        
        # Save final checkpoint
        self.save_checkpoint(bc_model_path)
        print("Training completed!")
        print(f"Final average loss: {np.mean(self.train_losses[-100:]):.4f}")


    def save_checkpoint(self, filename: str = None):
        """Save model checkpoint."""
        if filename is None:
            filename = f'model_epoch_{self.epoch}_step_{self.step}.pt'
        
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': self.epoch,
            'step': self.step,
            'train_losses': self.train_losses,
            
        }
        
        torch.save(checkpoint, filename)
        print(f"Checkpoint saved to {filename}")