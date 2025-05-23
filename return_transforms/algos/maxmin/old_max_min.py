import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from data_loading.load_mujoco import Trajectory
from return_transforms.models.ardt.maxmin_model import RtgFFN, RtgLSTM
from return_transforms.datasets.ardt_dataset import ARDTDataset


class ValueNet(nn.Module):
    def __init__(self, obs_size):
        super(ValueNet, self).__init__()
        self.obs_size = obs_size
        self.fc1 = nn.Linear(obs_size, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, obs):
        batch_size, obs_len, dim = obs.shape
        if dim != self.obs_size:
            raise ValueError(f"Expected obs last dim {self.obs_size}, got {dim}")
        x = obs.reshape(-1, self.obs_size)  # Use reshape instead of view
        x = self.fc1(x)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.fc3(x)
        x = x.reshape(batch_size, obs_len, 1)  # Use reshape here too
        return x
    
    

def _expectile_fn(
        td_error: torch.Tensor, 
        acts_mask: torch.Tensor, 
        alpha: float = 0.01, 
        discount_weighted: bool = False
    ) -> torch.Tensor:
    """
    Expectile loss function to focus on different quantiles of the TD-error distribution.
    """
    batch_loss = torch.abs(alpha - F.normalize(F.relu(td_error), dim=-1))
    batch_loss *= (td_error ** 2)
    if discount_weighted:
        weights = 0.5 ** np.array(range(len(batch_loss)))[::-1]
        return (
            batch_loss[~acts_mask] * torch.from_numpy(weights).to(td_error.device)
        ).mean()
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
    
    # Initialize state and action spaces
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

    # Build dataset and dataloader
    max_len = max([len(traj.obs) for traj in trajs]) + 1
    dataset = ARDTDataset(
        trajs, 
        action_size, 
        adv_action_size, 
        max_len, 
        gamma=train_args['gamma'], 
        act_type=action_type
    )
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=train_args['batch_size'], num_workers=n_cpu
    )

    # Set up models
    print(f'Creating models... (simple={is_simple_model})')
    if is_simple_model:
        qsa_pr_model = RtgFFN(obs_size, action_size, include_adv=False).to(device)
        qsa_adv_model = RtgFFN(obs_size, action_size, adv_action_size, include_adv=True).to(device)
    else:
        qsa_pr_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=False).to(device)
        qsa_adv_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)
    
    # Add ValueNet
    value_net = ValueNet(obs_size).to(device)

    # Optimizers
    qsa_pr_optimizer = torch.optim.AdamW(
        qsa_pr_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )
    qsa_adv_optimizer = torch.optim.AdamW(
        qsa_adv_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )
    value_optimizer = torch.optim.AdamW(
        value_net.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )

    # Training parameters
    mse_epochs = train_args['mse_epochs']
    maxmin_epochs = train_args['maxmin_epochs'] 
    total_epochs = mse_epochs + maxmin_epochs
    assert maxmin_epochs % 2 == 0
    gamma = train_args['gamma']
    tau = 0.7  # τ < 0.5 for worst-case robustness

    print('Training...')
    qsa_pr_model.train()
    qsa_adv_model.train()
    value_net.train()
    
    for epoch in range(total_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        total_loss = 0
        total_pr_loss = 0
        total_adv_loss = 0
        total_v_loss = 0
        total_q_loss = 0
        total_batches = 0

        for obs, acts, adv_acts, ret, seq_len in pbar:
            total_batches += 1
            qsa_pr_optimizer.zero_grad()
            qsa_adv_optimizer.zero_grad()
            value_optimizer.zero_grad()
            
            # Adjust for toy environment
            if is_toy:
                print(f"obs shape before toy adjustment: {obs.shape}")
                obs, acts, adv_acts, ret = (
                    obs[:, :-1], acts[:, :-1], adv_acts[:, :-1], ret[:, :-1]
                )
                print(f"obs shape after toy adjustment: {obs.shape}")
            if seq_len.max() >= obs.shape[1]:
                seq_len -= 1

            # Set up variables
            batch_size = obs.shape[0]
            obs_len = obs.shape[1]
            obs = obs.view(batch_size, obs_len, -1).to(device).contiguous()
            if obs.shape[-1] != obs_size:
                raise ValueError(f"obs last dim {obs.shape[-1]} does not match obs_size {obs_size}")
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            acts_mask = (acts.sum(dim=-1) == 0)
            ret = (ret / train_args['scale']).to(device)
            seq_len = seq_len.to(device)
            rewards = (ret[:, :-1] - ret[:, 1:]).view(batch_size, -1)

            # Adjustment for initial prompt learning
            obs[:, 0] = obs[:, 1]
            ret[:, 0] = ret[:, 1]
            acts_mask[:, 0] = False

            # Model predictions
            ret_pr_pred = qsa_pr_model(obs, acts).view(batch_size, obs_len)
            ret_adv_pred = qsa_adv_model(obs, acts, adv_acts).view(batch_size, obs_len)
            v_pred = value_net(obs)
            print(f"v_pred shape: {v_pred.shape}")

            # Losses
            if epoch < mse_epochs:
                ret_pr_loss = (((ret_pr_pred - ret) ** 2) * ~acts_mask).mean()
                ret_adv_loss = (((ret_adv_pred - ret) ** 2) * ~acts_mask).mean()
                v_loss = _expectile_fn(ret_pr_pred.detach() - v_pred.squeeze(-1), acts_mask, alpha=0.5)
                total_loss_epoch = ret_pr_loss + ret_adv_loss + v_loss
                total_loss_epoch.backward()
                torch.nn.utils.clip_grad_norm_(qsa_pr_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(qsa_adv_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
                qsa_pr_optimizer.step()
                qsa_adv_optimizer.step()
                value_optimizer.step()
                total_loss += total_loss_epoch.item()
                total_pr_loss += ret_pr_loss.item()
                total_adv_loss += ret_adv_loss.item()
                total_v_loss += v_loss.item()

            elif epoch % 2 == 0:
                ret_pr_loss = _expectile_fn(ret_pr_pred - ret_adv_pred.detach(), acts_mask, alpha=0.5)
                v_loss = _expectile_fn(ret_pr_pred.detach() - v_pred.squeeze(-1), acts_mask, alpha=0.5)
                print(f"obs[:, 1:] shape: {obs[:, 1:].shape}, is_contiguous: {obs[:, 1:].is_contiguous()}")
                next_v_pred = value_net(obs[:, 1:].contiguous()).squeeze(-1)
                q_target = rewards + gamma * next_v_pred
                q_loss = ((ret_pr_pred[:, :-1] - q_target) ** 2 * ~acts_mask[:, :-1]).mean()
                total_loss_epoch = ret_pr_loss + v_loss + q_loss
                total_loss_epoch.backward()
                torch.nn.utils.clip_grad_norm_(qsa_pr_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
                qsa_pr_optimizer.step()
                value_optimizer.step()
                total_loss += total_loss_epoch.item()
                total_pr_loss += ret_pr_loss.item()
                total_v_loss += v_loss.item()
                total_q_loss += q_loss.item()

            else:
                ret_tree_loss = _expectile_fn(
                    ret_pr_pred[:, 1:].detach() + rewards - ret_adv_pred[:, :-1], 
                    acts_mask[:, :-1], 
                    alpha=0.5
                )
                ret_leaf_loss = (
                    (ret_adv_pred[range(batch_size), seq_len].flatten() - ret[range(batch_size), seq_len]) ** 2
                ).mean()
                ret_adv_loss = ret_tree_loss * (1 - train_args['leaf_weight']) + ret_leaf_loss * train_args['leaf_weight']
                v_loss = _expectile_fn(ret_adv_pred.detach() - v_pred.squeeze(-1), acts_mask, alpha=0.5)
                next_ret_pr_pred = qsa_pr_model(obs[:, 1:].contiguous(), acts[:, 1:]).detach()
                max_next_q = next_ret_pr_pred.max(dim=-1)[0]
                q_target = rewards + gamma * max_next_q
                q_loss = ((ret_adv_pred[:, :-1] - q_target) ** 2 * ~acts_mask[:, :-1]).mean()
                adv_reg_loss = torch.mean(ret_adv_pred ** 2) * 1e-4
                total_loss_epoch = ret_adv_loss + v_loss + q_loss + adv_reg_loss
                total_loss_epoch.backward()
                torch.nn.utils.clip_grad_norm_(qsa_adv_model.parameters(), max_norm=1.0)
                torch.nn.utils.clip_grad_norm_(value_net.parameters(), max_norm=1.0)
                qsa_adv_optimizer.step()
                value_optimizer.step()
                total_loss += total_loss_epoch.item()
                total_adv_loss += ret_adv_loss.item()
                total_v_loss += v_loss.item()
                total_q_loss += q_loss.item()

            pbar.set_description(
                f"Epoch {epoch} | "
                f"Total Loss: {total_loss / total_batches:.4f} | "
                f"Pr Loss: {total_pr_loss / total_batches:.4f} | "
                f"Adv Loss: {total_adv_loss / total_batches:.4f} | "
                f"V Loss: {total_v_loss / total_batches:.4f} | "
                f"Q Loss: {total_q_loss / total_batches:.4f}"
            )

    # Get learned returns using ValueNet
    with torch.no_grad():
        learned_returns = []
        prompt_value = -np.inf

        for traj in tqdm(trajs):
            obs = torch.from_numpy(np.array(traj.obs)).float().to(device).view(1, -1, obs_size)
            acts = torch.from_numpy(np.array(traj.actions)).to(device).view(1, -1)
            if action_type == "discrete" and not is_discretize:
                acts = torch.nn.functional.one_hot(acts, num_classes=action_size)
            else:
                acts = acts.view(1, -1, action_size)
            # Use ValueNet for returns
            returns = value_net(obs).cpu().flatten().numpy()
            if prompt_value < returns[-len(traj.actions)]:
                prompt_value = returns[-len(traj.actions)]
            learned_returns.append(np.round(returns * train_args['scale'], decimals=3))

    return learned_returns, np.round(prompt_value * train_args['scale'], decimals=3)