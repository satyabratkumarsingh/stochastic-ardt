
import time

import numpy as np
import torch
from tqdm import tqdm

from data_loading.load_mujoco import Trajectory

def slice_and_reshape_actions(traj_actions, start_idx, context_size, action_dim, pad_value=0.0):
    actions = np.array(traj_actions)
    if actions.ndim == 1:
        actions = actions.reshape(-1, 1)
    
    # Adjust action dimension to match action_dim
    current_action_dim = actions.shape[1] if actions.ndim > 1 else 1
    if current_action_dim < action_dim:
        padding = np.full((actions.shape[0], action_dim - current_action_dim), pad_value)
        actions = np.hstack([actions, padding])
    elif current_action_dim > action_dim:
        actions = actions[:, :action_dim]

    slice_actions = actions[start_idx:start_idx + context_size]
    
    # Pad at the beginning if the sliced sequence is shorter than context_size
    if len(slice_actions) < context_size:
        padding = np.full((context_size - len(slice_actions), action_dim), pad_value)
        slice_actions = np.vstack([padding, slice_actions])
    elif len(slice_actions) > context_size:
        slice_actions = slice_actions[:context_size]
    
    result = slice_actions.reshape(1, context_size, action_dim)
    if result.shape != (1, context_size, action_dim):
        raise ValueError(f"Unexpected shape {result.shape}, expected (1, {context_size}, {action_dim})")
    
    return result



class TrainConfigs:
    """
    Training configurations for the model.

    Args:
        action_dim (int): The dimension of the action space.
        adv_action_dim (int): The dimension of the adversarial action space.
        action_type (str): The type of action space (continuous or discrete).
        state_dim (int): The dimension of the state space.
        state_mean (float): The mean of the state space.
        state_std (float): The standard deviation of the state space.
        returns_scale (int): The scale of the returns.
        top_pct_traj (float): The percentage of top trajectories to use.
        episode_length (int): The length of the episode.
        batch_size (int): The size of the batch.
        normalize_states (bool): Whether to normalize the states.
    """
    def __init__(
        self,
        action_dim: int,
        adv_action_dim: int,
        action_type: str,
        state_dim: int,
        state_mean: float,
        state_std: float,
        returns_scale: int,
        top_pct_traj: float,
        episode_length: int,
        batch_size: int = 1,
        normalize_states: bool = True,
    ):
        self.action_dim = action_dim
        self.adv_action_dim = adv_action_dim
        self.action_type = action_type
        self.state_dim = state_dim
        self.state_mean = state_mean
        self.state_std = state_std
        self.returns_scale = returns_scale
        self.top_pct_traj = top_pct_traj
        self.episode_length = episode_length
        self.batch_size = batch_size
        self.normalize_states = normalize_states


class Trainer:
    """
    Trainer class for training a model.

    Args:
        model (torch.nn.Module): The model to train.
        model_type (str): The type of model being trained.
        optimizer (torch.optim.Optimizer): The optimizer to use for training.
        scheduler (torch.optim.lr_scheduler): The learning rate scheduler to use.
        gradients_clipper (callable): A function to clip gradients.
        context_size (int): The size of the context window.
        with_adv_action (bool): Whether to use adversarial actions.
        env_name (str): The name of the environment.
        trajectories (list[Trajectory]): The trajectories to train on.
        trajectories_sorted_idx (np.array): The indices of the sorted trajectories.
        trajectories_sorted_probs (np.array): The probabilities of the sorted trajectories.
        train_configs (TrainConfigs): The training configurations.
        eval_fns (list[callable]): The evaluation functions to use.
    """
    def __init__(
            self,
            model: torch.nn.Module, 
            model_type: str,
            optimizer: torch.optim.Optimizer,
            scheduler: torch.optim.lr_scheduler,
            gradients_clipper: callable,
            context_size: int,
            with_adv_action: bool,
            env_name: str,
            num_trajectories: int,
            trajectories: list[Trajectory],
            trajectories_sorted_idx: np.array,
            trajectories_sorted_probs: np.array,
            train_configs: TrainConfigs,
            eval_fns: list[callable]
        ):
       
        # Track training start time and initialize diagnostics
        self.start_time = time.time()
        self.diagnostics = dict()
        
        # Model-related parameters
        self.model = model
        self.model_type = model_type
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.gradients_clipper = gradients_clipper
        self.context_size = context_size
        self.with_adv_action = with_adv_action
        
        # Data-related parameters
        self.env_name = env_name
        self.num_trajectories = num_trajectories
        self.trajectories = trajectories
        self.trajectories_sorted_idx = trajectories_sorted_idx
        self.trajectories_sorted_probs = trajectories_sorted_probs
        
        # Training configurations
        self.train_configs = train_configs
        
        # Define loss function based on the action type (continuous or discrete)
        self.loss_fn = self._get_loss_fn(train_configs.action_type)
        
        # Evaluation functions to assess model performance
        self.eval_fns = [] if eval_fns is None else eval_fns

    def _get_loss_fn(self, action_type: str):
        """
        Get the appropriate loss function based on the action type.
        """
        if action_type == 'continuous':
            return lambda a_hat, a: torch.mean((a_hat - a)**2)
        else:
            ce_loss = torch.nn.CrossEntropyLoss()
            return lambda a_hat, a: ce_loss(a_hat, torch.argmax(a, dim=-1))

    def slice_and_reshape_dones(self, dones, start_idx, context_size, pad_value=2):
        dones = np.array(dones)
        slice_dones = dones[start_idx:start_idx + context_size]
        if len(slice_dones) < context_size:
            padding = np.full((context_size - len(slice_dones),), pad_value)
            slice_dones = np.concatenate([slice_dones, padding])
        elif len(slice_dones) > context_size:
            slice_dones = slice_dones[:context_size]
        return slice_dones.reshape(1, context_size)

    def _get_normalized_value(self, output_list, actions, si, dim, pad_value=-10):
        output_list = output_list if output_list is not None else []
        result = slice_and_reshape_actions(
            actions,
            start_idx=si,
            context_size=self.context_size,
            action_dim=dim,
            pad_value=pad_value  # Pass pad_value explicitly
        )
        output_list.append(result)
        return output_list


    def _get_batch(self, device: str = "cpu"):
        """
        Get a batch of data from the trajectories.
        """
        batch_idx = np.random.choice(
            np.arange(self.num_trajectories),
            size=self.train_configs.batch_size,
            p=self.trajectories_sorted_probs,
            replace=True,
        )
        s, a, adv_a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], [], []
        for i in range(self.train_configs.batch_size):
            traj = self.trajectories[int(self.trajectories_sorted_idx[batch_idx[i]])]
            if self.env_name not in ["halfcheetah", "hopper", "walker2d"]:
                si = 0
            else:
                si = np.random.randint(0, len(traj['observations']))

            # Handle states
            if self.env_name == "connect_four":
                cur_state = np.array([obs for obs in traj['observations']])
            else:
                cur_state = np.array(traj['observations'])

            # Ensure cur_state has shape [traj_length, state_dim]
            if cur_state.ndim == 1:
                cur_state = cur_state.reshape(-1, 1)
            if cur_state.shape[-1] != self.train_configs.state_dim:
                if cur_state.ndim == 2 and cur_state.shape[-1] < self.train_configs.state_dim:
                    padding = np.zeros((cur_state.shape[0], self.train_configs.state_dim - cur_state.shape[-1]))
                    cur_state = np.hstack([cur_state, padding])
                else:
                    raise ValueError(f"Unexpected state shape {cur_state.shape}, expected [traj_length, {self.train_configs.state_dim}]")

            # Slice and pad states
            slice_states = cur_state[si:si + self.context_size]
            if len(slice_states) < self.context_size:
                padding = np.zeros((self.context_size - len(slice_states), self.train_configs.state_dim))
                slice_states = np.vstack([padding, slice_states])
            elif len(slice_states) > self.context_size:
                slice_states = slice_states[:self.context_size]
            s.append(slice_states.reshape(1, self.context_size, self.train_configs.state_dim))

            # Use slice_and_reshape_actions for actions, adversarial actions, rewards, and rtg
            a = self._get_normalized_value(a, traj['actions'], si, dim=self.train_configs.action_dim, pad_value=-10)
            adv_a = self._get_normalized_value(adv_a, traj['adv_actions'], si, dim=self.train_configs.adv_action_dim, pad_value=0)
            r = self._get_normalized_value(r, traj['rewards'], si, dim=1, pad_value=0)
            d.append(self.slice_and_reshape_dones(traj.get('terminals', traj['dones']), si, self.context_size, pad_value=2))
            rtg = self._get_normalized_value(rtg, traj['rtg'], si, dim=1, pad_value=0)
            rtg[-1] = rtg[-1] / self.train_configs.returns_scale

            # Handle timesteps
            timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
            timesteps[-1][timesteps[-1] >= self.train_configs.episode_length] = self.train_configs.episode_length - 1
            tlen = s[-1].shape[1]
            timesteps[-1] = np.concatenate([np.zeros((1, self.context_size - tlen)), timesteps[-1]], axis=1)
            mask.append(np.concatenate([np.zeros((1, self.context_size - tlen)), np.ones((1, tlen))], axis=1))

            # Normalize states if required
            if self.train_configs.normalize_states:
                s[-1] = (s[-1] - self.train_configs.state_mean) / (self.train_configs.state_std + 1e-8)

        s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
        a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
        adv_a = torch.from_numpy(np.concatenate(adv_a, axis=0)).to(dtype=torch.float32, device=device)
        r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
        d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
        rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
        timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
        mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)

        if self.with_adv_action:
            return s, a, adv_a, r, d, rtg, timesteps, mask
        else:
            return s, a, r, d, rtg, timesteps, mask

    # def _get_batch(
    #         self,  
    #         device: str = "cpu"):
    #     """
    #     Get a batch of data from the trajectories.
    #     """
    #     batch_idx = np.random.choice(
    #     np.arange(self.num_trajectories),
    #     size=self.train_configs.batch_size,
    #     p=self.trajectories_sorted_probs,
    #     replace=True,
    # )
    #     s, a, adv_a, r, d, rtg, timesteps, mask = [], [], [], [], [], [], [], []
    #     for i in range(self.train_configs.batch_size):
    #     #for i in range(1):
    #         traj = self.trajectories[int(self.trajectories_sorted_idx[batch_idx[i]])]
    #         if self.env_name not in ["halfcheetah", "hopper", "walker2d"]:
    #             si = 0
    #         else:
    #             si = np.random.randint(0, traj['rewards'].shape[0]) 

    #         if self.env_name == "connect_four":
    #             cur_state = np.array([obs for obs in traj['observations']])
    #         else:
    #             cur_state = traj['observations']
            
    #         s.append(cur_state[si:si + self.context_size].reshape(1, -1, self.train_configs.state_dim))

            

    #         # print(f"State shape before padding: {s[-1].shape}")
    #         # print(f"Action shape before padding: {traj['actions'].shape}")
    #         # print(f"Reward shape before padding: {traj['rewards'].shape}")
    #         # print(f"Dones shape before padding: {traj['dones'].shape}")
    #         # print(f"RTG shape before padding: {traj['rtg']}")
           
    #         # *** WORKING VERSION ***
    #         # a.append(traj['actions'][si:si + self.context_size].reshape(1, -1, self.train_configs.action_dim))
    #         # adv_a.append(traj['adv_actions'][si:si + self.context_size].reshape(1, -1, self.train_configs.adv_action_dim))
    #         # r.append(traj['rewards'][si:si + self.context_size].reshape(1, -1, 1))
    #         # d.append(traj.get('terminals', traj['dones'])[si:si + self.context_size].reshape(1, -1))
    #         # rtg.append(traj['rtg'][si:si + self.context_size].reshape(1, -1, 1))
    #         # timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
    #         # timesteps[-1][timesteps[-1] >= self.train_configs.episode_length] = self.train_configs.episode_length - 1
    #         # #apply padding and normalisation
    #         # tlen = s[-1].shape[1]
    #         # s[-1] = np.concatenate([np.zeros((1, self.context_size - tlen, self.train_configs.state_dim)), s[-1]], axis=1)
    #         # if self.train_configs.normalize_states:
    #         #     s[-1] = (s[-1] - self.state_mean) / self.state_std
    #         # a[-1] = np.concatenate([np.ones((1, self.context_size - tlen, self.train_configs.action_dim)) * -10., a[-1]], axis=1)
    #         # adv_a[-1] = np.concatenate([np.zeros((1, self.context_size - tlen, self.train_configs.adv_action_dim)), adv_a[-1]], axis=1)
    #         # r[-1] = np.concatenate([np.zeros((1, self.context_size -  tlen, 1)), r[-1]], axis=1)
    #         # d[-1] = np.concatenate([np.ones((1, self.context_size - tlen)) * 2, d[-1]], axis=1)
    #         # rtg[-1] = np.concatenate([np.zeros((1, self.context_size - tlen, 1)),   rtg[-1]], axis=1) / self.train_configs.returns_scale
    #         # timesteps[-1] = np.concatenate([np.zeros((1, self.context_size - tlen)), timesteps[-1]], axis=1)
    #         # mask.append(np.concatenate([np.zeros((1, self.context_size - tlen)), np.ones((1, tlen))], axis=1))

    #         # *** NON WORKING VERSION ***
    #         a = self._get_normalized_value(a, traj['actions'], si, dim=self.train_configs.action_dim, pad_value=-10)
    #         adv_a = self._get_normalized_value(adv_a, traj['adv_actions'], si, dim=self.train_configs.adv_action_dim, pad_value=0)
    #         r = self._get_normalized_value(r, traj['rewards'], si, dim=1, pad_value=0)
    #         d.append(self.slice_and_reshape_dones(traj.get('terminals', traj['dones']), si, self.context_size, pad_value=2))
    #         rtg = self._get_normalized_value(rtg, traj['rtg'], si, dim=1, pad_value=0)
    #         rtg[-1] = rtg[-1] / self.train_configs.returns_scale

    #         timesteps.append(np.arange(si, si + s[-1].shape[1]).reshape(1, -1))
    #         timesteps[-1][timesteps[-1] >= self.train_configs.episode_length] = self.train_configs.episode_length - 1

    #         tlen = s[-1].shape[1]
    #         s[-1] = np.concatenate([np.zeros((1, self.context_size - tlen, self.train_configs.state_dim)), s[-1]], axis=1)
    #         if self.train_configs.normalize_states:
    #             s[-1] = (s[-1] - self.state_mean) / self.state_std
    #         timesteps[-1] = np.concatenate([np.zeros((1, self.context_size - tlen)), timesteps[-1]], axis=1)
    #         mask.append(np.concatenate([np.zeros((1, self.context_size - tlen)), np.ones((1, tlen))], axis=1))

    #         # print('====================================')
    #         # print(f"State shape AFTER padding: {s[-1].shape}")
    #         # print(f"Action shape AFTER padding: {traj['actions'].shape}")
    #         # print(f"Reward shape AFTER padding: {traj['rewards'].shape}")
    #         # print(f"Dones shape AFTER padding: {traj['dones'].shape}")
    #         # print(f"RTG shape AFTER padding: {traj['rtg']}")

      

    #     s = torch.from_numpy(np.concatenate(s, axis=0)).to(dtype=torch.float32, device=device)
    #     a = torch.from_numpy(np.concatenate(a, axis=0)).to(dtype=torch.float32, device=device)
    #     adv_a = torch.from_numpy(np.concatenate(adv_a, axis=0)).to(dtype=torch.float32, device=device)
    #     r = torch.from_numpy(np.concatenate(r, axis=0)).to(dtype=torch.float32, device=device)
    #     d = torch.from_numpy(np.concatenate(d, axis=0)).to(dtype=torch.long, device=device)
    #     rtg = torch.from_numpy(np.concatenate(rtg, axis=0)).to(dtype=torch.float32, device=device)
    #     timesteps = torch.from_numpy(np.concatenate(timesteps, axis=0)).to(dtype=torch.long, device=device)
    #     mask = torch.from_numpy(np.concatenate(mask, axis=0)).to(device=device)
        
    #     if self.with_adv_action:
    #         return s, a, adv_a, r, d, rtg, timesteps, mask
    #     else:
    #         return s, a, r, d, rtg, timesteps, mask

    def train_iteration(self, num_steps, iter_num=0, device="cpu", print_logs=False):
        """
        Train the model for a given number of steps.
        """
        train_losses = []
        logs = dict()
        train_start = time.time()
        
        # Set model to training mode
        self.model.train()

        # Training loop
        for _ in tqdm(range(num_steps)):
            # Perform a training step
            train_loss = self.train_step()
            train_losses.append(train_loss)
            
            # Step the learning rate scheduler if it is defined
            if self.scheduler is not None:
                self.scheduler.step()

        # Record training statistics
        logs['time/training'] = time.time() - train_start
        logs['training/train_loss_mean'] = np.mean(train_losses)
        logs['training/train_loss_std'] = np.std(train_losses)

        # Set model to evaluation mode and evaluate the model using the eval functions
        eval_start = time.time()
        self.model.eval()
        for eval_fn in self.eval_fns:
            outputs = eval_fn(model=self.model, model_type=self.model_type)
            for k, v in outputs.items():
                logs[f'evaluation/{k}'] = v

        logs['time/evaluation'] = time.time() - eval_start
        logs['time/total'] = time.time() - train_start

        # Log and print diagonistics metrics
        for k in self.diagnostics:
            logs[k] = self.diagnostics[k]

        if print_logs:
            print('=' * 80)
            print(f'Iteration {iter_num}')
            for k, v in logs.items():
                print(f'{k}: {v}')

        return logs