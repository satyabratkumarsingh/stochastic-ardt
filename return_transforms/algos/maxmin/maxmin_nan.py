import gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from data_loading.load_mujoco import Trajectory
from return_transforms.models.ardt.maxmin_model import RtgFFN, RtgLSTM
from return_transforms.datasets.ardt_dataset import ARDTDataset
import torch
import torch.nn as nn
import torch.nn.functional as F



import torch
import torch.nn as nn
from return_transforms.models.basic.mlp import MLP


class ValueNetwork(nn.Module):
    """
    A neural network designed to predict the state value function V(s) given only observations.
    
    Key Features:
    - If `is_lstm` is True, uses an LSTM to capture temporal dependencies in the observation sequence.
    - If `is_lstm` is False, uses an MLP for feed-forward prediction.
    - Outputs a single value per state, representing V(s).
    """
    def __init__(
        self,
        state_dim: int,
        model_args: dict = None,
        hidden_dim: int = 64,
        is_lstm: bool = False
    ):
        super().__init__()
        self.state_dim = state_dim
        self.is_lstm = is_lstm

        # Validate inputs
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if model_args is None and is_lstm:
            raise ValueError("model_args dictionary is required for LSTM mode")

        # Set hidden dimension from model_args if provided
        hidden_dim = model_args.get('hidden_size', hidden_dim) if model_args else hidden_dim
        self.hidden_dim = hidden_dim

        # Observation processing MLP
        self.obs_model = MLP(state_dim, hidden_dim, **({'hidden_size': 64, 'num_layers': 2}))

        if is_lstm:
            self.lstm_model = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.value_head = nn.Linear(hidden_dim, 1)
        else:
            self.value_head = MLP(hidden_dim, 1, **({'hidden_size': 64, 'num_layers': 2}))

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        # Validate input shape
        if len(obs.shape) < 2:
            raise ValueError(f"obs must have at least 2 dimensions (batch_size, seq_len, ...), got shape {obs.shape}")
        
        batch_size, seq_len = obs.shape[:2]
        device = self.obs_model.layers[0].weight.device
        obs = obs.to(device)

        # Process observations through MLP
        obs_reps = self.obs_model(obs.view(batch_size * seq_len, -1)).view(batch_size, seq_len, -1)

        if self.is_lstm:
            # Initialize LSTM hidden state
            hidden = (
                torch.zeros(1, batch_size, self.hidden_dim).to(device),
                torch.zeros(1, batch_size, self.hidden_dim).to(device)
            )
            x, _ = self.lstm_model(obs_reps, hidden)
            value_pred = self.value_head(x)  # [batch_size, seq_len, 1]
        else:
            value_pred = self.value_head(obs_reps)  # [batch_size, seq_len, 1]
        
        return value_pred
    
def _expectile_fn(
    td_error: torch.Tensor, 
    acts_mask: torch.Tensor, 
    alpha: float = 0.01, 
    discount_weighted: bool = False
) -> torch.Tensor:
    batch_loss = torch.abs(alpha - F.normalize(F.relu(td_error), dim=-1))
    batch_loss *= (td_error ** 2)
    if discount_weighted:
        weights = 0.5 ** torch.arange(len(batch_loss), device=td_error.device).flip(0)
        return (batch_loss[~acts_mask] * weights).mean()
    else:
        return (batch_loss.squeeze(-1) * ~acts_mask).mean()

def _value_function_loss(
    v_values: torch.Tensor,
    q_values: torch.Tensor,
    acts_mask: torch.Tensor,
    obs: torch.Tensor,
    device: str
) -> torch.Tensor:
    batch_size, seq_len = v_values.shape
    q_values_masked = q_values.masked_fill(acts_mask.unsqueeze(-1), -float('inf'))
    q_max = q_values_masked.max(dim=-1)[0]  # [batch_size, seq_len]
    q_next_mean = q_values_masked[:, 1:].mean(dim=-1)  # [batch_size, seq_len-1]
    loss = ((v_values[:, :-1] - q_max[:, :-1]) ** 2).mean()
    loss += ((v_values[:, 1:] - q_next_mean) ** 2).mean()
    return loss

def _iql_loss(
    v_values: torch.Tensor,
    q_values: torch.Tensor,
    rewards: torch.Tensor,
    acts_mask: torch.Tensor,
    gamma: float,
    device: str
) -> torch.Tensor:
    batch_size, seq_len = v_values.shape
    q_values_masked = q_values.masked_fill(acts_mask.unsqueeze(-1), -float('inf'))
    q_max = q_values_masked.max(dim=-1)[0]  # [batch_size, seq_len]
    q_next_masked = q_values[:, 1:].masked_fill(acts_mask[:, 1:].unsqueeze(-1), -float('inf'))
    q_next_max = q_next_masked.max(dim=-1)[0]  # [batch_size, seq_len-1]
    target = rewards.squeeze(-1) + gamma * q_next_max  # [batch_size, seq_len-1]
    l_v = ((v_values[:, :-1] - q_max[:, :-1]) ** 2).mean()
    l_q = ((v_values[:, 1:] - target) ** 2).mean()
    return l_v + l_q

def _q_only_loss(
    q_values: torch.Tensor,
    rewards: torch.Tensor,
    acts_mask: torch.Tensor,
    gamma: float,
    device: str
) -> torch.Tensor:
    batch_size, seq_len, action_size = q_values.shape
    q_values_masked = q_values.masked_fill(acts_mask.unsqueeze(-1), -float('inf'))
    q_next_max = q_values_masked[:, 1:].max(dim=-1)[0]  # [batch_size, seq_len-1]
    target = rewards.squeeze(-1) + gamma * q_next_max  # [batch_size, seq_len-1]
    td_error = target.unsqueeze(-1) - q_values[:, :-1]  # [batch_size, seq_len-1, action_size]
    loss = (td_error ** 2).mean()
    return loss

def maxmin(
    trajs: list[Trajectory],
    action_space: gym.spaces.Space,
    adv_action_space: gym.spaces.Space,
    train_args: dict,
    device: str,
    n_cpu: int,
    is_simple_model: bool = False,
    is_toy: bool = False,
    is_discretize: bool = False,
) -> tuple[np.ndarray, float]:
    # Validate train_args
    required_keys = ['gamma', 'model_lr', 'model_wd', 'batch_size', 'mse_epochs', 'maxmin_epochs', 'scale', 'alpha', 'leaf_weight']
    for key in required_keys:
        if key not in train_args:
            raise ValueError(f"Missing required train_args key: {key}")

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
        v_model = ValueNetwork(obs_size, is_lstm=False).to(device)  # V(s) model, MLP-based
    else:
        qsa_pr_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=False).to(device)
        qsa_adv_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)
        v_model = ValueNetwork(obs_size, train_args, is_lstm=True).to(device)  # V(s) model, LSTM-based

    qsa_pr_optimizer = torch.optim.AdamW(
        qsa_pr_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )
    qsa_adv_optimizer = torch.optim.AdamW(
        qsa_adv_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )
    v_optimizer = torch.optim.AdamW(
        v_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )

    # Training
    mse_epochs = train_args['mse_epochs']
    maxmin_epochs = train_args['maxmin_epochs'] 
    total_epochs = mse_epochs + maxmin_epochs
    assert maxmin_epochs % 2 == 0, "maxmin_epochs must be even"

    print('Training...')
    qsa_pr_model.train()
    qsa_adv_model.train()
    v_model.train()
    
    value_loss_weight = train_args.get('value_loss_weight', 0.1)
    iql_loss_weight = train_args.get('iql_loss_weight', 0.1)
    q_only_loss_weight = train_args.get('q_only_loss_weight', 0.1)
    
    for epoch in range(total_epochs):
        pbar = tqdm(dataloader, total=len(dataloader))
        total_loss = 0
        total_pr_loss = 0
        total_adv_loss = 0
        total_v_loss = 0
        total_batches = 0

        for obs, acts, adv_acts, ret, seq_len in pbar:
            total_batches += 1
            qsa_pr_optimizer.zero_grad()
            qsa_adv_optimizer.zero_grad()
            v_optimizer.zero_grad()
            
            if is_toy:
                obs, acts, adv_acts, ret = (
                    obs[:, :-1], acts[:, :-1], adv_acts[:, :-1], ret[:, :-1]
                )
                seq_len = seq_len - 1

            if seq_len.max() >= obs.shape[1]:
                seq_len = seq_len - 1

            batch_size = obs.shape[0]
            obs_len = obs.shape[1]
            
            obs = obs.view(batch_size, obs_len, -1).to(device)
            acts = acts.to(device)
            adv_acts = adv_acts.to(device)
            acts_mask = (acts.sum(dim=-1) == 0)
            ret = (ret / train_args['scale']).to(device)
            seq_len = seq_len.to(device)

            obs[:, 0] = obs[:, 1]
            ret[:, 0] = ret[:, 1]
            acts_mask[:, 0] = False

            # Compute rewards
            rewards = (ret[:, :-1] - ret[:, 1:]).view(batch_size, -1, 1)

            # Compute V(s)
            v_pred = v_model(obs).view(batch_size, obs_len)  # [batch_size, seq_len]

            # Calculate losses
            if epoch < mse_epochs:
                # MSE stage
                ret_pr_pred = qsa_pr_model(obs, acts).view(batch_size, obs_len, -1)
                ret_pr_loss = (((ret_pr_pred - ret.unsqueeze(-1)) ** 2) * ~acts_mask.unsqueeze(-1)).mean()
                ret_adv_pred = qsa_adv_model(obs, acts, adv_acts).view(batch_size, obs_len, -1)
                ret_adv_loss = (((ret_adv_pred - ret.unsqueeze(-1)) ** 2) * ~acts_mask.unsqueeze(-1)).mean()
                
                # New losses
                value_loss_pr = _value_function_loss(v_pred, ret_pr_pred, acts_mask, obs, device)
                iql_loss_pr = _iql_loss(v_pred, ret_pr_pred, rewards, acts_mask, train_args['gamma'], device)
                q_only_loss_pr = _q_only_loss(ret_pr_pred, rewards, acts_mask, train_args['gamma'], device)
                
                value_loss_adv = _value_function_loss(v_pred, ret_adv_pred, acts_mask, obs, device)
                iql_loss_adv = _iql_loss(v_pred, ret_adv_pred, rewards, acts_mask, train_args['gamma'], device)
                q_only_loss_adv = _q_only_loss(ret_adv_pred, rewards, acts_mask, train_args['gamma'], device)
                
                total_pr_loss_term = (
                    ret_pr_loss +
                    value_loss_weight * value_loss_pr +
                    iql_loss_weight * iql_loss_pr +
                    q_only_loss_weight * q_only_loss_pr
                )
                total_adv_loss_term = (
                    ret_adv_loss +
                    value_loss_weight * value_loss_adv +
                    iql_loss_weight * iql_loss_adv +
                    q_only_loss_weight * q_only_loss_adv
                )
                total_v_loss_term = value_loss_weight * (value_loss_pr + value_loss_adv) + iql_loss_weight * (iql_loss_pr + iql_loss_adv)
                
                # Combine losses and backpropagate once
                total_loss_term = total_pr_loss_term + total_adv_loss_term + total_v_loss_term
                total_loss_term.backward()
                qsa_pr_optimizer.step()
                qsa_adv_optimizer.step()
                v_optimizer.step()
                
                total_loss += total_loss_term.item()
                total_pr_loss += total_pr_loss_term.item()
                total_adv_loss += total_adv_loss_term.item()
                total_v_loss += total_v_loss_term.item()
            elif epoch % 2 == 0:
                # Max step
                ret_pr_pred = qsa_pr_model(obs, acts)
                ret_adv_pred = qsa_adv_model(obs, acts, adv_acts)
                ret_pr_loss = _expectile_fn(ret_pr_pred - ret_adv_pred.detach(), acts_mask, train_args['alpha'])
                
                value_loss_pr = _value_function_loss(v_pred, ret_pr_pred, acts_mask, obs, device)
                iql_loss_pr = _iql_loss(v_pred, ret_pr_pred, rewards, acts_mask, train_args['gamma'], device)
                q_only_loss_pr = _q_only_loss(ret_pr_pred, rewards, acts_mask, train_args['gamma'], device)
                
                total_pr_loss_term = (
                    ret_pr_loss +
                    value_loss_weight * value_loss_pr +
                    iql_loss_weight * iql_loss_pr +
                    q_only_loss_weight * q_only_loss_pr
                )
                total_v_loss_term = value_loss_weight * value_loss_pr + iql_loss_weight * iql_loss_pr
                
                # Combine losses and backpropagate once
                total_loss_term = total_pr_loss_term + total_v_loss_term
                total_loss_term.backward()
                qsa_pr_optimizer.step()
                v_optimizer.step()
                
                total_loss += total_loss_term.item()
                total_pr_loss += total_pr_loss_term.item()
                total_v_loss += total_v_loss_term.item()
            else:
                # Min step
                ret_pr_pred = qsa_pr_model(obs, acts)
                ret_adv_pred = qsa_adv_model(obs, acts, adv_acts)
                ret_tree_loss = _expectile_fn(
                    ret_pr_pred[:, 1:].detach() + rewards - ret_adv_pred[:, :-1], 
                    acts_mask[:, :-1], 
                    train_args['alpha']
                )
                ret_leaf_loss = (
                    (ret_adv_pred[range(batch_size), seq_len].flatten() - ret[range(batch_size), seq_len]) ** 2
                ).mean()
                ret_adv_loss = ret_tree_loss * (1 - train_args['leaf_weight']) + ret_leaf_loss * train_args['leaf_weight']
                
                value_loss_adv = _value_function_loss(v_pred, ret_adv_pred, acts_mask, obs, device)
                iql_loss_adv = _iql_loss(v_pred, ret_adv_pred, rewards, acts_mask, train_args['gamma'], device)
                q_only_loss_adv = _q_only_loss(ret_adv_pred, rewards, acts_mask, train_args['gamma'], device)
                
                total_adv_loss_term = (
                    ret_adv_loss +
                    value_loss_weight * value_loss_adv +
                    iql_loss_weight * iql_loss_adv +
                    q_only_loss_weight * q_only_loss_adv
                )
                total_v_loss_term = value_loss_weight * value_loss_adv + iql_loss_weight * iql_loss_adv
                
                # Combine losses and backpropagate once
                total_loss_term = total_adv_loss_term + total_v_loss_term
                total_loss_term.backward()
                qsa_adv_optimizer.step()
                v_optimizer.step()
                
                total_loss += total_loss_term.item()
                total_adv_loss += total_adv_loss_term.item()
                total_v_loss += total_v_loss_term.item()

            pbar.set_description(
                f"Epoch {epoch} | "
                f"Total Loss: {total_loss / total_batches:.4f} | "
                f"Pr Loss: {total_pr_loss / total_batches:.4f} | "
                f"Adv Loss: {total_adv_loss / total_batches:.4f} | "
                f"V Loss: {total_v_loss / total_batches:.4f}"
            )

    # Get learned returns and prompt values
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
            returns = qsa_pr_model(
                obs.view(obs.shape[0], -1, obs_size), acts.float()
            ).cpu().flatten().numpy()

            if prompt_value < returns[-len(traj.actions)]:
                prompt_value = returns[-len(traj.actions)]

            learned_returns.append(np.round(returns * train_args['scale'], decimals=3))

    return learned_returns, np.round(prompt_value * train_args['scale'], decimals=3)