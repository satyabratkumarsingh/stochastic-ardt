

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
from utils.saved_names import dt_model_name, behaviour_cloning_model_name
from core_models.behaviour_cloning.behaviour_cloning import MLPBCModel

class ModelLoader:

    def __init__(self, seed, game_name, config, n_cpu = 1, device = 'cpu'):
        self.seed = seed
        self.game_name = game_name
        config = yaml.safe_load(Path(config).read_text())
        self.dt_train_args = config
        self.n_cpu = n_cpu
        self.device = device
    

    def load_models(self, method):
        bc_model_path = behaviour_cloning_model_name(seed=self.seed, game=self.game_name, method=method) 
        dt_model_path = dt_model_name(seed=self.seed, game=self.game_name, method=method)   
        # Load new model trained from DT
        if os.path.exists(dt_model_path):
            try:
                checkpoint = torch.load(dt_model_path, map_location=self.device)

                # Step 2: Use the saved parameters to initialize a new model
                model_params = checkpoint['model_params']
                obs_size = model_params['obs_size']
                action_size = model_params['action_size']
                action_type = model_params['action_type']
                horizon = model_params['horizon']
                effective_max_ep_len = model_params['effective_max_ep_len']

                # Initialize a new model with the CORRECT parameters from the checkpoint
                dt_model = DecisionTransformer(
                    state_dim=obs_size,
                    act_dim=action_size,
                    hidden_size=32,          # Match the checkpoint's hidden size
                    max_length=horizon,
                    max_ep_len=effective_max_ep_len,
                    action_tanh=(action_type == 'continuous'),
                    action_type=action_type,
                    n_layer=2,               # Match the number of layers from the checkpoint
                    n_head=1,                 # Match the number of heads
                    n_inner=256,              # Match the inner size from the checkpoint
                    dropout=0.1
                )
                dt_model.load_state_dict(checkpoint['model_state_dict'])
                print(f"✅ Loaded decision transformer model from {dt_model_path}")
            except Exception as e:
                print(f"❌ Failed to load decision transformer model: {e}")
        else:
            raise
        # Load baseline model
        if os.path.exists(bc_model_path):
            try:
                checkpoint = torch.load(bc_model_path, map_location=self.device)
                model_params = checkpoint['model_params']
                
                bc_model= MLPBCModel(
                    state_dim=model_params['obs_size'],
                    act_dim=model_params['action_size'],
                    hidden_size=self.dt_train_args.get('hidden_size', 256),
                    n_layer=self.dt_train_args.get('n_layer', 3),
                    dropout=self.dt_train_args.get('dropout', 0.1),
                    max_length=1 
                ).to(self.device)
                
                bc_model.load_state_dict(checkpoint['model_state_dict'])

                print(f"✅ Loaded behaviour cloning from {bc_model_path}")
            except Exception as e:
                print(f"❌ Failed to load behaviour cloning model: {e}")
        
        return bc_model, dt_model