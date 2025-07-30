import pickle
from pathlib import Path

import gym
import numpy as np
import yaml
from data_class.trajectory import Trajectory
from dataclasses import asdict
import json


def _convert_ndarray_to_list(obj):
    """Recursively convert numpy arrays to lists in nested structures."""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [_convert_ndarray_to_list(i) for i in obj]
    elif isinstance(obj, dict):
        return {k: _convert_ndarray_to_list(v) for k, v in obj.items()}
    else:
        return obj

def generate_maxmin(
        env: gym.Env,
        trajs: list[Trajectory],
        config: dict,
        ret_file: str, # This will be the base prefix
        device: str,
        n_cpu: int,
        is_simple_model: bool = False,
        is_toy: bool = False,
        run_implicit: bool = False
    ):

    config = yaml.safe_load(Path(config).read_text())
    # Normalize observations if specified in the config
    if config['normalize']:
        trajs = _normalize_obs(trajs)

    # --- FIX START: Determine the final output file prefix upfront ---
    final_ret_file_prefix = ret_file
    if run_implicit:
        final_ret_file_prefix = f"{ret_file}_implicit"
        print("****** Running implicit Q learning =======")
        from return_transforms.algos.maxmin.maxmin_implicit import maxmin
    else:
        print("******* Running normal ARDT with MAX MIN =======")
        final_ret_file_prefix = f"{ret_file}_maxmin"
        from return_transforms.algos.maxmin.maxmin import maxmin


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

    print(f' ======== Done. Saving relabeled trajectories and prompts with prefix {final_ret_file_prefix}.')
    Path(final_ret_file_prefix).parent.mkdir(parents=True, exist_ok=True) # Use the final prefix for dir creation

    # Save relabeled trajectories (with minimax_returns_to_go)
    with open(f"{final_ret_file_prefix}_trajectories.pkl", 'wb') as f: # More explicit filename
        pickle.dump(relabeled_trajs, f)
    print(f"Saved relabeled trajectories to {final_ret_file_prefix}_trajectories.pkl")

    # Save prompt values
    with open(f"{final_ret_file_prefix}_prompt.pkl", 'wb') as f:
        pickle.dump(prompt_value, f)
    print(f"Saved prompt value to {final_ret_file_prefix}_prompt.pkl")

    # Convert dataclasses -> dict -> lists for JSON
    relabeled_trajs_dict = [asdict(t) for t in relabeled_trajs]
    relabeled_trajs_clean = [_convert_ndarray_to_list(t) for t in relabeled_trajs_dict]

    with open(f"{final_ret_file_prefix}.json", "w") as f: # Use final prefix for JSON
        json.dump(relabeled_trajs_clean, f, indent=4) # Added indent for readability
    print(f"Saved JSON representation to {final_ret_file_prefix}.json")


    # --- JSON version for prompt_value ---
    prompt_value_clean = _convert_ndarray_to_list(prompt_value)
    with open(f"{final_ret_file_prefix}_prompt.json", "w") as f: # Use final prefix for JSON prompt
        json.dump(prompt_value_clean, f)
    print(f"Saved prompt JSON to {final_ret_file_prefix}_prompt.json")


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