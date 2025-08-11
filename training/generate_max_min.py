import pickle
from pathlib import Path

import gym
import numpy as np
import yaml
from data_class.trajectory import Trajectory
from dataclasses import asdict
from utils.saved_names import pkl_name_min_max_relabeled, json_name_min_max_relabeled, prompt_min_max
import json


def _convert_ndarray_to_list(obj):
    """Recursively convert numpy arrays and scalars to lists/native types in nested structures."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    # FIX: Add checks for NumPy scalar types
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, list):
        return [_convert_ndarray_to_list(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: _convert_ndarray_to_list(v) for k, v in obj.items()}
    else:
        return obj

def generate_maxmin(
        game_name: str,                               
        method: str,
        seed: int,
        env: gym.Env,
        trajs: list[Trajectory],
        config: dict,
        device: str,
        n_cpu: int,
        is_simple_model: bool = False,
        is_toy: bool = False
    ):

    config = yaml.safe_load(Path(config).read_text())
    # Normalize observations if specified in the config
    if config['normalize']:
        trajs = _normalize_obs(trajs)

    if method == 'implicit_q':
        print("****** Running implicit Q learning =======")
        from core_models.implicit_q.maxmin_implicit import maxmin
    elif method == 'max_min':
        print("******* Running normal ARDT with MAX MIN =======")
        from core_models.maxmin.maxmin import maxmin
    elif method == 'cql':
        print("******* Running normal ARDT CQL  =======")
        from core_models.cql.cql import maxmin
    else:
        raise ValueError(f"Unknown method: {method}")

    print('Generating ARDT returns (maxmin phase)...')
    relabeled_trajs, prompt_value = maxmin(
        trajs,
        env.action_space,
        env.adv_action_space,
        config['train_args'],
        device,
        n_cpu,
        is_simple_model=is_simple_model,
        is_toy=is_toy
    )

    pkl_file_relabeled = pkl_name_min_max_relabeled(seed=seed, game= game_name, method=method)
    json_file = json_name_min_max_relabeled(seed=seed, game= game_name, method=method)
    prompt_file = prompt_min_max(seed=seed, game= game_name, method=method)
    print(f' ==============Done. Saving relabeled trajectories and prompts with prefix {pkl_file_relabeled}.')
    Path(pkl_file_relabeled).parent.mkdir(parents=True, exist_ok=True) 

    with open(pkl_file_relabeled, 'wb') as f:
        pickle.dump(relabeled_trajs, f)
    print(f"Saved relabeled trajectories to {pkl_file_relabeled}")

    # Convert dataclasses -> dict -> lists for JSON
    relabeled_trajs_dict = [asdict(t) for t in relabeled_trajs]
    relabeled_trajs_clean = [_convert_ndarray_to_list(t) for t in relabeled_trajs_dict]

    with open(json_file, "w") as f: # Use final prefix for JSON
        json.dump(relabeled_trajs_clean, f, indent=4) # Added indent for readability
    print(f"Saved JSON representation to {json_file}")


    # --- JSON version for prompt_value ---
    # The fix ensures this conversion handles NumPy scalar types correctly
    prompt_value_clean = _convert_ndarray_to_list(prompt_value)
    with open(prompt_file, "w") as f: # Use final prefix for JSON prompt
        json.dump(prompt_value_clean, f)
    print(f"Saved prompt JSON to {prompt_file}")


def _normalize_obs(trajs: list[Trajectory]) -> list[Trajectory]:
    # Collect all observations from all trajectories
    obs_list = []
    for traj in trajs:
        obs_list.extend(traj.obs)

    # Compute mean and standard deviation of observations
    obs = np.array(obs_list)
    obs_mean = np.mean(obs, axis=0)
    obs_std = np.std(obs, axis=0) + 1e-8

    # Normalize each observation in each trajectory
    for traj in trajs:
        for i in range(len(traj.obs)):
            traj.obs[i] = (traj.obs[i] - obs_mean) / obs_std

    # Return normalized trajectories
    return trajs