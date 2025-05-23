
import gym
import numpy as np
import torch
import torch.nn.functional as F
from tqdm.autonotebook import tqdm

from data_loading.load_mujoco import Trajectory
from return_transforms.models.ardt.maxmin_model import RtgFFN, RtgLSTM
from return_transforms.datasets.ardt_dataset import ARDTDataset

def _expectile_fn(
        td_error: torch.Tensor, 
        acts_mask: torch.Tensor, 
        alpha: float = 0.01, 
        discount_weighted: bool = False
    ) -> torch.Tensor:
    """
    Expectile loss function to focus on different quantiles of the TD-error distribution.

    Args:
        td_error (torch.Tensor): Temporal difference error.
        acts_mask (torch.Tensor): Mask for invalid actions.
        alpha (float, optional): Expectile quantile parameter (default is 0.01).
        discount_weighted (bool, optional): If True, apply discount weighting.

    Returns:
        torch.Tensor: Computed expectile loss.
    """
    # Normalize and apply ReLU to the TD-error
    batch_loss = torch.abs(alpha - F.normalize(F.relu(td_error), dim=-1))
    
    # Square the TD-error
    batch_loss *= (td_error ** 2)

    # Apply discount weighting if needed
    if discount_weighted:
        weights = 0.5 ** np.array(range(len(batch_loss)))[::-1]
        return (
            batch_loss[~acts_mask] * torch.from_numpy(weights).to(td_error.device)
        ).mean()
    else:
        # Calculate expectile loss for valid actions
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
    Train a max-min adversarial reinforcement learning model to handle worst-case returns.

    Args:
        trajs (list[Trajectory]): List of trajectories.
        action_space (gym.spaces.Space): The action space of the environment.
        adv_action_space (gym.spaces.Space): Adversarial action space.
        train_args (dict): Training arguments including epochs, learning rates, and batch size.
        device (str): Device to run computations on ('cpu' or 'cuda').
        n_cpu (int): Number of CPUs to use for data loading.
        is_simple_model (bool, optional): Use a simpler model for testing (default is False).
        is_toy (bool, optional): Whether the environment is a toy model (default is False).
        is_discretize (bool, optional): Whether to discretize actions for certain environments (default is False).

    Returns:
        tuple: Learned return labels and highest returns-to-go (prompt value).
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

    # Build dataset and dataloader for training
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

    # Set up the models (MLP or LSTM-based) involved in the ARDT algorithm
    print(f'Creating models... (simple={is_simple_model})')
    if is_simple_model:
        qsa_pr_model = RtgFFN(obs_size, action_size, include_adv=False).to(device)
        qsa_adv_model = RtgFFN(obs_size, action_size, adv_action_size, include_adv=True).to(device)
    else:
        qsa_pr_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=False).to(device)
        qsa_adv_model = RtgLSTM(obs_size, action_size, adv_action_size, train_args, include_adv=True).to(device)

    qsa_pr_optimizer = torch.optim.AdamW(
        qsa_pr_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
    )
    qsa_adv_optimizer = torch.optim.AdamW(
        qsa_adv_model.parameters(), lr=train_args['model_lr'], weight_decay=train_args['model_wd']
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
        total_batches = 0

        for obs, acts, adv_acts, ret, seq_len in pbar:
            total_batches += 1
            qsa_pr_optimizer.zero_grad()
            qsa_adv_optimizer.zero_grad()
            
            # Adjust for toy environment
            if is_toy:
                obs, acts, adv_acts, ret = (
                    obs[:, :-1], acts[:, :-1], adv_acts[:, :-1], ret[:, :-1]
                )
            if seq_len.max() >= obs.shape[1]:
                seq_len -= 1

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

            # Calculate the losses at the different tages
            if epoch < mse_epochs:
                # MSE-based learning stage to learn general loss landscape
                ret_pr_pred = qsa_pr_model(obs, acts).view(batch_size, obs_len)
                ret_pr_loss = (((ret_pr_pred - ret) ** 2) * ~acts_mask).mean()
                ret_adv_pred = qsa_adv_model(obs, acts, adv_acts).view(batch_size, obs_len)
                ret_adv_loss = (((ret_adv_pred - ret) ** 2) * ~acts_mask).mean()
                # Backpropagate
                ret_pr_loss.backward()
                qsa_pr_optimizer.step()
                ret_adv_loss.backward()
                qsa_adv_optimizer.step()
                # Update losses
                total_loss += ret_pr_loss.item() + ret_adv_loss.item()
                total_pr_loss += ret_pr_loss.item()
                total_adv_loss += ret_adv_loss.item()
            elif epoch % 2 == 0:
                # Max step: protagonist attempts to maximise at each node
                ret_pr_pred = qsa_pr_model(obs, acts)
                ret_adv_pred = qsa_adv_model(obs, acts, adv_acts)
                ret_pr_loss = _expectile_fn(ret_pr_pred - ret_adv_pred.detach(), acts_mask, train_args['alpha'])            
                # Backpropagate
                ret_pr_loss.backward()
                qsa_pr_optimizer.step()
                # Update losses
                total_loss += ret_pr_loss.item()
                total_pr_loss += ret_pr_loss.item()
            else:
                # Min step: adversary attempts to minimise at each node             
                rewards = (ret[:, :-1] - ret[:, 1:]).view(batch_size, -1, 1)
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
                # Backpropagate
                ret_adv_loss.backward()
                qsa_adv_optimizer.step()
                # Update losses
                total_loss += ret_adv_loss.item()
                total_adv_loss += ret_adv_loss.item()

            pbar.set_description(
                f"Epoch {epoch} | "
                f"Total Loss: {total_loss / total_batches:.4f} | "
                f"Pr Loss: {total_pr_loss / total_batches:.4f} | "
                f"Adv Loss: {total_adv_loss / total_batches:.4f}"
            )

    # Get the learned return labels and prompt values (i.e. highest returns-to-go)
    with torch.no_grad():
        learned_returns = []
        prompt_value = -np.inf

        for traj in tqdm(trajs):
            # Predict returns
            obs = torch.from_numpy(np.array(traj.obs)).float().to(device).view(1, -1, obs_size)
            acts = torch.from_numpy(np.array(traj.actions)).to(device).view(1, -1)
            if action_type == "discrete" and not is_discretize:
                acts = torch.nn.functional.one_hot(acts, num_classes=action_size)
            else:
                acts = acts.view(1, -1, action_size)
            returns = qsa_pr_model(
                obs.view(obs.shape[0], -1, obs_size), acts.float()
            ).cpu().flatten().numpy()

            # Compare against previously held prompt value, keep the highest
            if prompt_value < returns[-len(traj.actions)]:
                prompt_value = returns[-len(traj.actions)]

            # Update the learned returns
            learned_returns.append(np.round(returns * train_args['scale'], decimals=3))

    # Return the learned returns and the scaled prompt value
    return learned_returns, np.round(prompt_value * train_args['scale'], decimals=3)