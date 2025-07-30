import pickle
from pathlib import Path
from data_class.trajectory import Trajectory

def get_relabeled_trajectories(ret_file, run_implicit = False):
     
    if run_implicit:
        effective_file_prefix = f"{ret_file}_implicit"
    else:
        effective_file_prefix = f"{ret_file}_maxmin"
   

    try:
        trajectories_file_path = Path(f"{effective_file_prefix}.pkl")
        prompt_file_path = Path(f"{effective_file_prefix}_prompt.pkl")
        # Load trajectories
        with open(trajectories_file_path, 'rb') as f:
            loaded_relabeled_trajs: list[Trajectory] = pickle.load(f)

        # Load prompt value
        with open(prompt_file_path, 'rb') as f:
            loaded_prompt_value = pickle.load(f)
        
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

