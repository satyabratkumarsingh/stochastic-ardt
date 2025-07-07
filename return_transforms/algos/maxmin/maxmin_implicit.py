import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from data_loading.load_mujoco import Trajectory
from return_transforms.models.ardt.maxmin_model import RtgFFN, RtgLSTM
from return_transforms.datasets.ardt_dataset import ARDTDataset
from return_transforms.models.ardt.value_net import ValueNet

IQL_LOSS_WEIGHT = 0.2
Q_ONLY_LOSS_WEIGHT = 0.2

def _iql_loss(
    v_values: torch.Tensor,
    q_values: torch.Tensor,
    rewards: torch.Tensor,
    acts_mask: torch.Tensor,
    gamma: float,
    device: str
) -> torch.Tensor:
    """
    Compute IQL loss to align Q-values with TD targets, ensuring conservative Q-learning.
    
    Args:
        v_values: State values, shape [batch_size, seq_len, 1]
        q_values: Q-values, shape [batch_size, seq_len, action_size]
        rewards: Rewards, shape [batch_size, seq_len-1, 1]
        acts_mask: Mask, shape [batch_size, seq_len]
        gamma: Discount factor
        device: Device for computations
    
    Returns:
        loss: Scalar tensor, IQL loss
    """
    batch_size, seq_len, _ = v_values.shape
    
    # Squeeze singleton dimensions
    v_values = v_values.squeeze(-1)  # [batch_size, seq_len]
    rewards = rewards.squeeze(-1)  # [batch_size, seq_len-1]
    q_values = q_values.squeeze(-1) if q_values.shape[-1] == 1 else q_values  # [batch_size, seq_len] or [batch_size, seq_len, action_size]
    
    # Compute TD target
    q_target = rewards + gamma * v_values[:, 1:].detach()  # [batch_size, seq_len-1]
    q_target = torch.nan_to_num(q_target, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Compute IQL loss
    if len(q_values.shape) == 3:
        q_values = q_values[:, :-1]  # [batch_size, seq_len-1, action_size]
        q_pred = q_values.max(dim=-1)[0]  # [batch_size, seq_len-1]
    else:
        q_pred = q_values[:, :-1]  # [batch_size, seq_len-1]
    
    loss = ((q_pred - q_target) ** 2 * ~acts_mask[:, :-1]).mean()
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    
    return loss


def _q_only_loss(
    q_values: torch.Tensor,
    rewards: torch.Tensor,
    acts_mask: torch.Tensor,
    gamma: float,
    device: str
) -> torch.Tensor:
    """
    Compute Q-only loss to align Q-values with TD targets based on rewards and next Q-values.
    
    Args:
        q_values: Q-values from qsa_pr_model or qsa_adv_model, shape [batch_size, seq_len, action_size]
        rewards: Single-step rewards, shape [batch_size, seq_len-1, 1]
        acts_mask: Mask for valid timesteps, shape [batch_size, seq_len]
        gamma: Discount factor
        device: Device for computations
    
    Returns:
        loss: Scalar tensor, Q-only loss
    """
    batch_size, seq_len, action_size = q_values.shape
    # Compute TD target: r_t + gamma * Q(s_{t+1}, a_{t+1})
    # For action_size=1, q_values is [batch_size, seq_len, 1]
    if action_size == 1:
        q_next = q_values[:, 1:].detach()  # [batch_size, seq_len-1, 1]
    else:
        # For multi-action, take max Q-value over actions
        q_next = q_values[:, 1:].detach().max(dim=-1, keepdim=True)[0]  # [batch_size, seq_len-1, 1]
    
    q_target = rewards + gamma * q_next  # [batch_size, seq_len-1, 1]
    q_target = torch.nan_to_num(q_target, nan=0.0, posinf=1e6, neginf=-1e6)
    
    # Compute Q-values for current state
    q_pred = q_values[:, :-1]  # [batch_size, seq_len-1, action_size]
    if action_size > 1:
        q_pred = q_pred.max(dim=-1, keepdim=True)[0]  # [batch_size, seq_len-1, 1]
    
    # Compute MSE loss for valid timesteps
    loss = ((q_pred - q_target) ** 2 * ~acts_mask[:, :-1].unsqueeze(-1)).mean()
    loss = torch.nan_to_num(loss, nan=0.0, posinf=0.0, neginf=0.0)
    
    return loss



def _expectile_fn(
    td_error: torch.Tensor,
    acts_mask: torch.Tensor,
    alpha: float = 0.7,
    discount_weighted: bool = False
) -> torch.Tensor:
    """
    Expectile loss function to focus on different quantiles of the TD-error distribution.
    """
    relu_td = F.relu(td_error)
    norm = torch.norm(relu_td, p=2, dim=-1, keepdim=True)
    norm = torch.where(norm == 0, torch.ones_like(norm), norm)  # Avoid division by zero
    normalized = relu_td / norm
    batch_loss = torch.abs(alpha - normalized) * (td_error ** 2)
    if discount_weighted:
        weights = 0.5 ** np.array(range(len(batch_loss)))[::-1]
        return (batch_loss[~acts_mask] * torch.from_numpy(weights).to(td_error.device)).mean()
    else:
        return (batch_loss.squeeze(-1) * ~acts_mask).mean()

def maxmin(
        trajs: list[Trajectory],
        action_space: gym.spaces,
        adv_action_space: gym.spaces,
        train_args: dict,
        device: str,
        n_cpu: int,
        is_simple_model: bool = False,
        is_toy: bool = False,
        is_discretize: bool = False,
    ) -> tuple[np.ndarray, float]:
    """
    Train a max-min adversarial RL model with value functions and new losses.
    """
    
   
    if isinstance(action_space, gym.spaces.Discrete):
        obs_size = np.prod(trajs[0].obs[0].shape)
        action_size = action_space.n
        adv_action_size = adv_action_space.n
        action_type = 'discrete'
    else:
        obs_size = np.prod(np.array(trajs[0].obs[0]).shape)
        action_size = action_space.shape[0]
        adv_action_size = adv_action_space.shape[0]
        action_type = 'continuous'

    # Build dataset and dataloader for training
    max_len = max([len(traj.obs) for traj in trajs]) + 1
    dataset = ARDTDataset(
        trajs, 
        max_len, 
        gamma=train_args['gamma'], 
        act_type=action_type
    )
    print(dataset)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_args['batch_size'], num_workers=n_cpu
    )

    # Set up the models (MLP or LSTM-based) involved in the ARDT algorithm
    print(f'Creating models... (simple={is_simple_model})')
    if is_simple_model:
        qsa_pr_model = RtgFFN(obs_size, action_size, include_adv=False).to(device)
        qsa_adv_model = RtgFFN(obs_size, action_size, adv_action_size, include_adv=True).to(device)
        v_model = ValueNet(obs_size, is_lstm=False).to(device)
    else:
        qsa_pr_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=False).to(device)
        qsa_adv_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)
        v_model = ValueNet(obs_size, train_args, is_lstm=True).to(device)

    qsa_pr_optimizer = torch.optim.AdamW(
        qsa_pr_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )
    qsa_adv_optimizer = torch.optim.AdamW(
        qsa_adv_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )
    value_optimizer = torch.optim.AdamW(
        v_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )


    # Start training and running the ARDT algorithm
    mse_epochs = train_args['mse_epochs']
    maxmin_epochs = train_args['maxmin_epochs'] 
    total_epochs = mse_epochs + maxmin_epochs
    assert maxmin_epochs % 2 == 0

    print('Training...')
    qsa_pr_model.train()
    qsa_adv_model.train()

    
    for epoch in range(total_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        total_loss = 0
        total_pr_loss = 0
        total_adv_loss = 0
        total_v_loss = 0
        total_q_loss = 0
        total_batches = 0
        total_iql_loss = 0

        for obs, acts, adv_acts, ret, seq_len in pbar:
            total_batches += 1
            qsa_pr_optimizer.zero_grad()
            qsa_adv_optimizer.zero_grad()
            value_optimizer.zero_grad()
        
            # Adjust for toy environment
            if is_toy:
                obs, acts, adv_acts, ret = (
                    obs[:, :-1], acts[:, :-1], adv_acts[:, :-1], ret[:, :-1]
                )
            if seq_len.max() >= obs.shape[1]:
                seq_len -= 1

            gamma=train_args['gamma']

          
            # Set up variables
            batch_size = obs.shape[0]
            obs_len = obs.shape[1]
            
            obs = obs.view(batch_size, obs_len, -1).to(device)
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            acts_mask = (acts.sum(dim=-1) == 0)
            ret = (ret / train_args['scale']).to(device)
            seq_len = seq_len.to(device)

            # Adjustment for initial prompt learning
            obs[:, 0] = obs[:, 1]
            ret[:, 0] = ret[:, 1]
            acts_mask[:, 0] = False


            # # Model predictions
            ret_pr_pred = qsa_pr_model(obs, acts) # Q(s, a) for protagonist
            ret_adv_pred = qsa_adv_model(obs, acts, adv_acts) # Q(s, a, adv)
            
            v_pred = v_model(obs)  # [batch_size, obs_len, 1]
         
            ret = torch.nan_to_num(ret, nan=0.0, posinf=1e6, neginf=-1e6)
            ret = (ret / train_args['scale']).to(device)
            if ret.shape != (batch_size, obs_len):
                raise ValueError(f"Expected ret shape [{batch_size}, {obs_len}], got {ret.shape}")
            rewards = ret[:, :-1] - gamma * ret[:, 1:]  # [batch_size, seq_len-1]
            rewards = rewards.unsqueeze(-1)  # [batch_size, seq_len-1, 1]
            rewards = torch.nan_to_num(rewards, nan=0.0, posinf=1e6, neginf=-1e6)


            # Losses
            if epoch < mse_epochs:
                ret_pr_pred = torch.clamp(qsa_pr_model(obs, acts), -1e6, 1e6)  # [128, 3, 1]
                ret_adv_pred = torch.clamp(qsa_adv_model(obs, acts, adv_acts), -1e6, 1e6)  # [128, 3, 1]
                v_pred = torch.clamp(v_model(obs), -1e6, 1e6)  # [128, 3, 1]

                ret_pr_loss = (((ret_pr_pred - ret.unsqueeze(-1)) ** 2) * ~acts_mask.unsqueeze(-1)).mean()
                ret_adv_loss = (((ret_adv_pred - ret.unsqueeze(-1)) ** 2) * ~acts_mask.unsqueeze(-1)).mean()

                value_loss_pr = _expectile_fn(ret_pr_pred.detach().squeeze(-1) - v_pred.squeeze(-1), acts_mask, alpha=0.7)
                value_loss_adv = _expectile_fn(ret_adv_pred.detach().squeeze(-1) - v_pred.squeeze(-1), acts_mask, alpha=0.5)
                v_loss = 0.7 * value_loss_pr + 0.3 * value_loss_adv

                iql_loss_pr = _iql_loss(v_pred, ret_pr_pred, rewards, acts_mask, gamma, device)
                iql_loss_adv = _iql_loss(v_pred, ret_adv_pred, rewards, acts_mask, gamma, device)

                q_only_loss_pr = _q_only_loss(ret_pr_pred, rewards, acts_mask, gamma, device)
                q_only_loss_adv = _q_only_loss(ret_adv_pred, rewards, acts_mask, gamma, device)


                total_loss_epoch = (ret_pr_loss + ret_adv_loss + v_loss + IQL_LOSS_WEIGHT *
                                     (iql_loss_pr + iql_loss_adv) + Q_ONLY_LOSS_WEIGHT * (q_only_loss_pr + q_only_loss_adv))
                total_loss_epoch = torch.nan_to_num(total_loss_epoch, nan=0.0, posinf=0.0, neginf=0.0)
                total_loss_epoch.backward()
                qsa_pr_optimizer.step()
                qsa_adv_optimizer.step()
                value_optimizer.step()
                total_loss += total_loss_epoch.item() if not torch.isnan(total_loss_epoch) else 0
                total_pr_loss += ret_pr_loss.item() if not torch.isnan(ret_pr_loss) else 0
                total_adv_loss += ret_adv_loss.item() if not torch.isnan(ret_adv_loss) else 0
                total_v_loss += v_loss.item() if not torch.isnan(v_loss) else 0
                total_q_loss += (q_only_loss_pr.item() + q_only_loss_adv.item()) if not (torch.isnan(q_only_loss_pr) or torch.isnan(q_only_loss_adv)) else 0
                total_iql_loss += iql_loss_pr.item() if not torch.isnan(iql_loss_pr) else 0

            elif epoch % 2 == 0:
                # Max step: protagonist maximizes
                ret_pr_loss = _expectile_fn(ret_pr_pred - ret_adv_pred.detach(), acts_mask, train_args['alpha'])
                
                # L_V: V(s) ≈ Q(s, a) for protagonist
                v_loss = _expectile_fn(ret_pr_pred.detach().squeeze(-1) - v_pred.squeeze(-1), acts_mask, alpha=train_args.get('tau', 0.7))
                
                # L_Q: Q(s, a) = r + γ Q(s', a') using _q_only_loss
                q_only_loss_pr = _q_only_loss(ret_pr_pred, rewards, acts_mask, gamma, device)
                
                total_loss_epoch = (
                    ret_pr_loss +
                    v_loss +
                    Q_ONLY_LOSS_WEIGHT * q_only_loss_pr
                )
                total_loss_epoch = torch.nan_to_num(total_loss_epoch, nan=0.0, posinf=0.0, neginf=0.0)
                total_loss_epoch.backward()
          
                qsa_pr_optimizer.step()
                value_optimizer.step()
                total_loss += total_loss_epoch.item() if not torch.isnan(total_loss_epoch) else 0
                total_pr_loss += ret_pr_loss.item() if not torch.isnan(ret_pr_loss) else 0
                total_v_loss += v_loss.item() if not torch.isnan(v_loss) else 0
                total_q_loss += q_only_loss_pr.item() if not torch.isnan(q_only_loss_pr) else 0
                total_iql_loss += iql_loss_pr.item() if not torch.isnan(iql_loss_pr) else 0

            else:
                # Min step: adversary minimizes
                 # Min step: adversary minimizes
                ret_tree_loss = _expectile_fn(
                    ret_pr_pred[:, 1:].detach() + rewards - ret_adv_pred[:, :-1],
                    acts_mask[:, :-1],
                    train_args['alpha']
                )
                ret_leaf_loss = (
                    (ret_adv_pred[range(batch_size), seq_len].flatten() - ret[range(batch_size), seq_len]) ** 2
                ).mean()
                ret_adv_loss = ret_tree_loss * (1 - train_args['leaf_weight']) + ret_leaf_loss * train_args['leaf_weight']
                
                # L_V: V(s) ≈ Q(s, a) for adversary
                v_loss = _expectile_fn(ret_adv_pred.detach().squeeze(-1) - v_pred.squeeze(-1), acts_mask, alpha=train_args.get('tau', 0.5))
                
                # L_Q: Q(s, a) = r + γ Q(s', a') using _q_only_loss
                q_only_loss_adv = _q_only_loss(ret_adv_pred, rewards, acts_mask, gamma, device)
                
                # L_IQL: Q(s, a) = r + γ V(s') using _iql_loss
                iql_loss_adv = _iql_loss(v_pred, ret_adv_pred, rewards, acts_mask, gamma, device)
                
                total_loss_epoch = (
                    ret_adv_loss +
                    v_loss +
                    Q_ONLY_LOSS_WEIGHT * q_only_loss_adv +
                    IQL_LOSS_WEIGHT * iql_loss_adv
                )
                total_loss_epoch = torch.nan_to_num(total_loss_epoch, nan=0.0, posinf=0.0, neginf=0.0)
                total_loss_epoch.backward()
                qsa_adv_optimizer.step()
                value_optimizer.step()
                total_loss += total_loss_epoch.item() if not torch.isnan(total_loss_epoch) else 0
                total_adv_loss += ret_adv_loss.item() if not torch.isnan(ret_adv_loss) else 0
                total_v_loss += v_loss.item() if not torch.isnan(v_loss) else 0
                total_q_loss += q_only_loss_adv.item() if not torch.isnan(q_only_loss_adv) else 0
                total_iql_loss += iql_loss_adv.item() if not torch.isnan(iql_loss_adv) else 0

            pbar.set_description(f"Epoch {epoch} | Total Loss: {total_loss / total_batches:.6f} | "
                     f"Pr Loss: {total_pr_loss / total_batches:.6f} (batch: {ret_pr_loss.item():.6f}) | "
                     f"Adv Loss: {total_adv_loss / total_batches:.6f} (batch: {ret_adv_loss.item():.6f})\n"
                     f"V Loss: {total_v_loss / total_batches:.6f} (batch: {v_loss.item():.6f}) | "
                     f"Q Loss: {total_q_loss / total_batches:.6f} (pr: {q_only_loss_pr.item():.6f}, adv: {q_only_loss_adv.item():.6f}) | "
                     f"IQL Loss: {total_iql_loss / total_batches:.6f} (pr: {iql_loss_pr.item():.6f}, adv: {iql_loss_adv.item():.6f})")
            


    # Get learned returns using ValueNet
    with torch.no_grad():
        learned_returns = []
        prompt_value = -np.inf

        for traj in tqdm(trajs):
            obs = torch.from_numpy(np.array(traj.obs)).float().to(device).view(1, -1, obs_size)
            acts = torch.from_numpy(np.array(traj.actions)).to(device).view(1, -1)
            if action_type == "discrete" and not is_discretize:
                acts = torch.from_numpy(np.array(traj.actions)).float().to(device).view(1, -1, action_size)
            else:
                acts = acts.view(1, -1, action_size)
            # Use ValueNet for returns
            returns = v_model(obs).cpu().flatten().numpy()
            if prompt_value < returns[-len(traj.actions)]:
                prompt_value = returns[-len(traj.actions)]
            learned_returns.append(np.round(returns * train_args['scale'], decimals=3))

    return learned_returns, np.round(prompt_value * train_args['scale'], decimals=3)