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


def train_decision_transformer(dt_train_args,
                                relabeled_trajs: List,
                                use_minimax_returns: bool = True,
                                model_name: str = "ardt", n_cpu = 1, device = 'cpu') -> torch.nn.Module:
    """
    Train a Decision Transformer model
    """
    print(f"\n=== Training {model_name.upper()} Decision Transformer ===")
    print(f"Using minimax returns: {use_minimax_returns}")
    
    # Determine save path
    save_dir = "offline_data"
    os.makedirs(save_dir, exist_ok=True)
  

    if use_minimax_returns:
        save_path = os.path.join(save_dir, f"{model_name}_minimax_dt.pth")
    else:
        save_path = os.path.join(save_dir, f"{model_name}_baseline_dt.pth")
    
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
    horizon = dt_train_args.get('context_size', 20)
    max_ep_len_from_data = max([len(t.obs) for t in relabeled_trajs]) if relabeled_trajs else 100
    effective_max_ep_len = max(max_ep_len_from_data, horizon)

    print(f"Model config: obs_size={obs_size}, action_size={action_size}, action_type={action_type}")
    print(f"Horizon: {horizon}, Max episode length: {effective_max_ep_len}")

    # Initialize model
    dt_model = DecisionTransformer(
        state_dim=obs_size,
        act_dim=action_size,
        hidden_size=dt_train_args.get('hidden_size', 64),
        max_length=horizon,
        max_ep_len=effective_max_ep_len,
        action_tanh=(action_type == 'continuous'),
        action_type=action_type,
        n_layer=dt_train_args.get('transformer_n_layer', 2),
        n_head=dt_train_args.get('transformer_n_head', 1),
        n_inner=dt_train_args.get('transformer_n_inner', 256),
        dropout=dt_train_args.get('transformer_dropout', 0.1)
    ).to(device)

    # Initialize optimizer
    dt_optimizer = torch.optim.AdamW(
        dt_model.parameters(), 
        lr=dt_train_args.get('model_lr', 1e-4),
        weight_decay=dt_train_args.get('model_wd', 1e-4)
    )

    # Create dataset and dataloader
    dt_dataset = ARDTDataset(
        trajs=relabeled_trajs,
        horizon=horizon,
        gamma=dt_train_args.get('gamma', 0.99),
        act_type=action_type,
        use_minimax_returns=use_minimax_returns,
    )
    
    dt_dataloader = torch.utils.data.DataLoader(
        dt_dataset,
        batch_size=dt_train_args.get('batch_size', 64),
        num_workers=n_cpu
    )

    # Training loop
    dt_model.train()
    training_losses = []
    
    num_epochs = dt_train_args.get('dt_epochs', 10)
    for epoch in range(num_epochs):
        total_dt_loss = 0
        batch_count = 0
        
        pbar = tqdm(dt_dataloader, desc=f"{model_name.upper()} Epoch {epoch+1}/{num_epochs}")
        for batch_idx, batch_data in enumerate(pbar):
            obs, acts, adv_acts, minimax_ret, seq_len = [d.to(device) for d in batch_data]
            minimax_ret = (minimax_ret / dt_train_args.get('scale', 1.0))

            dt_optimizer.zero_grad()

            batch_size, horizon_len = obs.shape[0], obs.shape[1]
            timesteps = torch.arange(horizon_len, device=device).repeat(batch_size, 1)
            attention_mask = (torch.arange(horizon_len, device=device)[None, :] < seq_len[:, None]).bool()

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
                    loss = torch.tensor(0.0, device=device)
                else:
                    loss = torch.nn.functional.cross_entropy(
                        action_preds_flat, actions_target_flat.long(), reduction='mean'
                    )
            else:
                actions_target_flat = acts[valid_mask]
                if action_preds_flat.numel() == 0:
                    loss = torch.tensor(0.0, device=device)
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
            print(f"{model_name.upper()} Epoch {epoch+1} Avg Loss: {avg_loss:.6f}")
        else:
            print(f"{model_name.upper()} Epoch {epoch+1}: No valid batches processed")

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
    
    torch.save(checkpoint, save_path)
    print(f"✅ {model_name.upper()} model saved to {save_path}")
    print(f"✅ {model_name.upper()} training complete\n")

    return dt_model

def train_both_models(dt_train_args, relabeled_trajs: List[Trajectory], run_implicit: bool = False
) -> Tuple[torch.nn.Module, torch.nn.Module]:
    """
    Train both baseline and ARDT models only if saved models are not found or fail to load.
    """
    baseline_path = "offline_data/baseline_baseline_dt.pth"
    ardt_path = "offline_data/ardt_minimax_dt.pth"

    # --- Try loading saved models first ---
    print("🔍 Checking for saved models...")
    if os.path.exists(baseline_path) and os.path.exists(ardt_path):
        print("📂 Found pretrained models. Attempting to load...")
        baseline_model, ardt_model = load_trained_models(
            dt_train_args=dt_train_args,
            baseline_path=baseline_path,
            ardt_path=ardt_path
        )
        # --- Return early if loading successful ---
        if baseline_model is not None and ardt_model is not None:
            print("✅ Successfully loaded both models from disk.")
            return baseline_model, ardt_model
        else:
            print("⚠️ One or both models failed to load. Proceeding to train them again...")
    else:
        print("❌ Saved models not found. Proceeding to train from scratch...")

    # --- Train models if not found or failed to load ---
    print("🔄 Loading and relabeling trajectories...")
    print(f"✅ Loaded {len(relabeled_trajs)} trajectories")

    print("\n" + "=" * 60)
    print("🚀 TRAINING BOTH DECISION TRANSFORMER MODELS")
    print("=" * 60)

    baseline_model = train_decision_transformer(
        dt_train_args,
        relabeled_trajs,
        use_minimax_returns=False,
        model_name="baseline"
    )

    ardt_model = train_decision_transformer(
        dt_train_args,
        relabeled_trajs,
        use_minimax_returns=True,
        model_name="ardt"
    )

    print("=" * 60)
    print("✅ BOTH MODELS TRAINED SUCCESSFULLY")
    print("=" * 60)

    return baseline_model, ardt_model

def load_trained_models(dt_train_args, baseline_path: Optional[str] = None,
                           ardt_path: Optional[str] = None, device: str = 'cpu') -> Tuple[Optional[torch.nn.Module], Optional[torch.nn.Module]]:
        """
        Load previously trained models
        """
        if baseline_path is None:
            baseline_path = "offline_data/baseline_baseline_dt.pth"
        if ardt_path is None:
            ardt_path = "offline_data/ardt_minimax_dt.pth"
        
        baseline_model = None
        ardt_model = None
        
        # Load baseline model
        if os.path.exists(baseline_path):
            try:
                checkpoint = torch.load(baseline_path, map_location=device)
                model_params = checkpoint['model_params']
                
                baseline_model = DecisionTransformer(
                    state_dim=model_params['obs_size'],
                    act_dim=model_params['action_size'],
                    hidden_size=dt_train_args.get('hidden_size', 64),
                    max_length=model_params['horizon'],
                    max_ep_len=model_params['effective_max_ep_len'],
                    action_tanh=(model_params['action_type'] == 'continuous'),
                    action_type=model_params['action_type'],
                    n_layer=dt_train_args.get('transformer_n_layer', 2),
                    n_head=dt_train_args.get('transformer_n_head', 1),
                    n_inner=dt_train_args.get('transformer_n_inner', 256),
                    dropout=dt_train_args.get('transformer_dropout', 0.1)
                ).to(device)
                
                baseline_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded baseline model from {baseline_path}")
            except Exception as e:
                print(f"❌ Failed to load baseline model: {e}")
        
        # Load ARDT model
        if os.path.exists(ardt_path):
            try:
                checkpoint = torch.load(ardt_path, map_location=device)
                model_params = checkpoint['model_params']
                
                ardt_model = DecisionTransformer(
                    state_dim=model_params['obs_size'],
                    act_dim=model_params['action_size'],
                    hidden_size=dt_train_args.get('hidden_size', 64),
                    max_length=model_params['horizon'],
                    max_ep_len=model_params['effective_max_ep_len'],
                    action_tanh=(model_params['action_type'] == 'continuous'),
                    action_type=model_params['action_type'],
                    n_layer=dt_train_args.get('transformer_n_layer', 2),
                    n_head=dt_train_args.get('transformer_n_head', 1),
                    n_inner=dt_train_args.get('transformer_n_inner', 256),
                    dropout=dt_train_args.get('transformer_dropout', 0.1)
                ).to(device)
                
                ardt_model.load_state_dict(checkpoint['model_state_dict'])
                ardt_model = ardt_model
                print(f"✅ Loaded ARDT model from {ardt_path}")
            except Exception as e:
                print(f"❌ Failed to load ARDT model: {e}")
        
        return baseline_model, ardt_model
