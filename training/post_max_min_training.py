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

def train_dt_baseline(seed, game_name, dt_train_args, n_cpu = 1, device = 'cpu', is_implicit = False):

    dt_trainer = DecisionTransformerTrainer(seed, game_name, dt_train_args, n_cpu , device)
    trajectories, prompt_value = get_relabeled_trajectories(seed, game_name, is_implicit)
    #dt_trainer.train(trajectories, is_implicit = is_implicit)

    bc_trainer = BehaviourCloningTrainer(seed, game_name, dt_train_args, n_cpu , device)
    bc_trainer.train(trajectories, is_implicit = is_implicit)






