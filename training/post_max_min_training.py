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
from utils.trajectory_utils import get_relabeled_trajectories
from typing import Dict, List, Optional, Tuple
from training.dt_trainer import DecisionTransformerTrainer
from training.bc_trainer import BehaviourCloningTrainer

def train_dt_models(seed, game_name, method, dt_train_args, n_cpu = 1, device = 'cpu'):

    dt_trainer = DecisionTransformerTrainer(seed, game_name, dt_train_args, n_cpu , device)
    trajectories, prompt_value = get_relabeled_trajectories(seed, game_name, method)
    all_rtg = [traj.minimax_returns_to_go for traj in trajectories]
    print(f"RTG range: {min(all_rtg)} to {max(all_rtg)}")
    both_models =  dt_trainer.train_and_save(trajectories, method = method)
    print("==========Both decision transformer models trained successfully ============")
    return both_models






