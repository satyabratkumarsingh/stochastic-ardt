import pickle
from pathlib import Path
import json
import numpy as np
from data_class.trajectory import Trajectory
from utils.saved_names import pkl_name_min_max_relabeled, prompt_min_max
import torch 


def get_action_dim(action_data):
    if isinstance(action_data, list):
        if not action_data:
            return 1
        if isinstance(action_data[0], (int, float, np.number)):
            return 1
        elif isinstance(action_data[0], (list, np.ndarray, torch.Tensor)):
            return len(action_data[0])
    elif isinstance(action_data, np.ndarray):
        if action_data.ndim == 1:
            return 1
        elif action_data.ndim > 1:
            return action_data.shape[-1]
    elif torch.is_tensor(action_data):
        if action_data.ndim == 1:
            return 1
        elif action_data.ndim > 1:
            return action_data.shape[-1]
    raise TypeError(f"Unsupported action data type: {type(action_data)}")

def get_relabeled_trajectories(seed, game, is_implicit = False):
     
    try:
        trajectories_file_path = Path(pkl_name_min_max_relabeled(seed, game, is_implicit))
        prompt_file_path = Path(prompt_min_max(seed, game, is_implicit))
        # Load trajectories
        with open(trajectories_file_path, 'rb') as f:
            loaded_relabeled_trajs: list[Trajectory] = pickle.load(f)

        # Load prompt value
        with open(prompt_file_path, 'r') as f:
            loaded_prompt_value = json.load(f)
        
        print(f"Loaded {len(loaded_relabeled_trajs)} relabeled trajectories from {trajectories_file_path.name}")
        print(f"Loaded prompt value: {loaded_prompt_value:.3f} from {prompt_file_path.name}")
        return loaded_relabeled_trajs, loaded_prompt_value

    except FileNotFoundError:
        print(f"Error: One or more expected pickle files not found.")
        print(f"Attempted to load:")
        print(f"  Trajectories: {trajectories_file_path}")
        print(f"  Prompt: {prompt_file_path}")
        print(f"Please ensure the 'ret_file_base' and 'run_implicit' flag match the saving process.")
    except Exception as e:
        print(f"An unexpected error occurred during loading or training: {e}")
