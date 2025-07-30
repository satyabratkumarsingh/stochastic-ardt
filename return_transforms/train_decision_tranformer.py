from data_class.trajectory import Trajectory
import torch 
import pickle
from pathlib import Path
import gym
import numpy as np
import yaml
from tqdm import tqdm
import torch.nn.functional as F
from decision_transformer.decision_transformer.models.decision_transformer import DecisionTransformer
from return_transforms.datasets.ardt_dataset import ARDTDataset
from offline_setup.base_offline_env import BaseOfflineEnv
from return_transforms.eval_function import EvalFnGenerator

my_evaluation_targets = np.array([
    -2.0,    # Pessimistic: expect to lose
    -1.0,    # Slightly pessimistic  
    0.0,     # Neutral: break even
    1.0,     # Cautiously optimistic
    2.0,     # Moderately optimistic
    2.5,     # Near optimal
    2.909,   # OPTIMAL: Your exact prompt value
    3.2,     # Slightly above optimal
    3.5      # Ambitious target
])


def train_decision_transformer(
    env: BaseOfflineEnv,
    relabeled_trajs: list[Trajectory],
    config_path: str,
    device: str,
    n_cpu: int
) -> torch.nn.Module:
    """
    Trains a Decision Transformer model on the relabeled trajectories.
    """
    print("\n=== Decision Transformer Training Started ===")

    config = yaml.safe_load(Path(config_path).read_text())
    dt_train_args = config['train_args']
    
    obs_size = relabeled_trajs[0].obs[0].shape[0]
    action_size = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else env.action_space.n
    adv_action_size = env.adv_action_space.n if isinstance(env.adv_action_space, gym.spaces.Discrete) else env.adv_action_space.shape[0]

    horizon = dt_train_args.get('context_size', 20)
    max_ep_len_from_data = max([len(t.obs) for t in relabeled_trajs]) if relabeled_trajs else env.max_episode_steps
    effective_max_ep_len = max(max_ep_len_from_data, horizon) # Take the max of data-derived and context size
  
    action_type = 'discrete' if isinstance(env.action_space, gym.spaces.Discrete) else 'continuous'

    dt_model = DecisionTransformer(
        state_dim=obs_size,
        act_dim=action_size,
        hidden_size=dt_train_args.get('hidden_size', 64),
        max_length=horizon, # Use horizon as max_length for DT
        max_ep_len=effective_max_ep_len, # Max episode length in dataset for timestep encoding
        action_tanh=(action_type == 'continuous'),
        action_type=action_type,
        n_layer=dt_train_args.get('transformer_n_layer', 2),
        n_head=dt_train_args.get('transformer_n_head', 1),
        n_inner=dt_train_args.get('transformer_n_inner', 256),
        dropout=dt_train_args.get('transformer_dropout', 0.1)
    ).to(device)

    dt_optimizer = torch.optim.AdamW(
        dt_model.parameters(), lr=dt_train_args.get('model_lr', 1e-4),
        weight_decay=dt_train_args.get('model_wd', 1e-4)
    )

    # Instantiate your ARDTDataset for DT training
    dt_dataset = ARDTDataset(
        trajs=relabeled_trajs,
        horizon=horizon, # Use horizon for DT training
        gamma=dt_train_args.get('gamma', 0.99),
        act_type=action_type,
        use_minimax_returns=True,
    )
    dt_dataloader = torch.utils.data.DataLoader(
        dt_dataset,
        batch_size=dt_train_args.get('batch_size', 64),
        num_workers=n_cpu
    )

    dt_model.train()
    for epoch in range(dt_train_args.get('dt_epochs', 10)):
        total_dt_loss = 0
        for batch_idx, batch_data in enumerate(tqdm(dt_dataloader, desc=f"DT Epoch {epoch+1}/{dt_train_args.get('dt_epochs', 10)}")):

            obs, acts, adv_acts, minimax_ret, seq_len = [d.to(device) for d in batch_data]
            minimax_ret = (minimax_ret / dt_train_args.get('scale', 1.0)) # Scale returns here

            dt_optimizer.zero_grad()
            
            batch_size, horizon_len, _ = obs.shape
            timesteps = torch.arange(horizon_len, device=device).repeat(batch_size, 1) # (B, H)
            attention_mask = (torch.arange(horizon_len, device=device)[None, :] < seq_len[:, None]).bool() # (B, H)

            state_preds, action_preds_tensor, return_preds = dt_model(
                states=obs, actions=acts, returns_to_go=minimax_ret.unsqueeze(-1), # unsqueeze ret to (B,H,1)
                timesteps=timesteps, attention_mask=attention_mask
            )
            
            # Only consider valid predictions based on attention_mask
            # The action_preds will have shape (batch_size, horizon, act_dim)
            # The target actions (acts) will have shape (batch_size, horizon) for discrete or (batch_size, horizon, act_dim) for continuous
            
            valid_mask = attention_mask # (B, H)
            
            # Flatten predictions and targets to apply loss only on valid elements
            action_preds_flat = action_preds_tensor[valid_mask] # Shape (N_valid, act_dim)
            
            if action_type == 'discrete':
                # Target actions are `int64` for discrete case, need to flatten and use long()
                actions_target_flat = acts[valid_mask].float() # Shape (N_valid,)
                
                if action_preds_flat.numel() == 0:
                    loss = torch.tensor(0.0, device=device)
                else:
                    loss = F.cross_entropy(action_preds_flat, actions_target_flat, reduction='mean')
            else: # Continuous
                actions_target_flat = acts[valid_mask] # Shape (N_valid, act_dim)
                if action_preds_flat.numel() == 0:
                    loss = torch.tensor(0.0, device=device)
                else:
                    loss = F.mse_loss(action_preds_flat, actions_target_flat, reduction='mean')
            
            loss.backward()
            dt_optimizer.step()
            total_dt_loss += loss.item()
        print(f"DT Epoch {epoch+1} Avg Loss: {total_dt_loss/(batch_idx+1):.4f}")

    print("\nDecision Transformer training complete.")
    return dt_model

def evaluate_decision_transformer(
    dt_model: torch.nn.Module,
    env: gym.Env,
    relabeled_trajs: list[Trajectory],
    prompt_value: float,
    config_path: str,
    device: str,
    algo_name: str = 'ardt_dt', # A sensible default
    returns_filename: str = 'dt_eval_results',
    dataset_name: str = 'relabel_data',
    test_adv_name: str = 'no_adv',
    added_dataset_name: str = 'none',
    added_dataset_prop: float = 0.0
):
    """
    Evaluates a trained Decision Transformer model.

    Args:
        dt_model (torch.nn.Module): The trained Decision Transformer model.
        env (gym.Env): The environment for evaluation.
        relabeled_trajs (list[Trajectory]): The relabeled trajectories (used for state_mean/std calc).
        prompt_value (float): The target return to condition the DT on during evaluation.
        config_path (str): Path to the YAML configuration file (for eval-specific args).
        device (str): The device for evaluation.
        # EvalFnGenerator specific args:
        algo_name, returns_filename, dataset_name, test_adv_name, added_dataset_name, added_dataset_prop
    """
    print("\n=== Decision Transformer Evaluation Started ===")

    config = yaml.safe_load(Path(config_path).read_text())
    eval_args = config['train_args'] # Assuming eval args are nested under train_args for simplicity

    obs_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0] if isinstance(env.action_space, gym.spaces.Box) else 1
    action_size = 2
    adv_action_size = env.adv_action_space.n if isinstance(env.adv_action_space, gym.spaces.Discrete) else env.adv_action_space.shape[0]
    max_ep_len = max([len(t.obs) for t in relabeled_trajs]) if relabeled_trajs else env.max_episode_steps # Ensure max_ep_len is valid
    action_type = 'discrete' if isinstance(env.action_space, gym.spaces.Discrete) else 'continuous'

    # Calculate state_mean and state_std from all relabeled trajectories
    all_states = np.concatenate([np.array(t.obs) for t in relabeled_trajs], axis=0)
    state_mean = np.mean(all_states, axis=0)
    state_std = np.std(all_states, axis=0) + 1e-6

    eval_generator = EvalFnGenerator(
        seed=eval_args.get('seed', 42),
        env_name=env.env_name,
        task=env,
        num_eval_episodes=eval_args.get('num_eval_episodes', 10),
        state_dim=obs_size,
        act_dim=action_size,
        adv_act_dim=adv_action_size,
        action_type=action_type,
        max_traj_len=max_ep_len,
        scale=eval_args.get('scale', 1.0),
        state_mean=state_mean,
        state_std=state_std,
        batch_size=eval_args.get('batch_size', 64),
        normalize_states=eval_args.get('normalize_states', False),
        device=device,
        algo_name=algo_name,
        returns_filename=returns_filename,
        dataset_name=dataset_name,
        test_adv_name=test_adv_name,
        added_dataset_name=added_dataset_name,
        added_dataset_prop=added_dataset_prop,
        target_returns_to_evaluate=my_evaluation_targets
    )
    
    final_evaluation_summary = eval_generator.run_full_evaluation(model=dt_model,model_type='dt')
    
    print(f"Final Evaluation Results for prompt {prompt_value:.2f}: {final_evaluation_summary}")
    print("\nDecision Transformer evaluation complete.")
    return final_evaluation_summary