import os

def pkl_name_min_max_relabeled(seed, game, is_implicit = False):
    if is_implicit:
       filename = f"Runs/{game}_implicit_q_{seed}_relabeled_trajectories.pkl"
    else:
        filename =  f"Runs/{game}_max_min_{seed}_relabeled_trajectories.pkl"
    return filename
    
def json_name_min_max_relabeled(seed, game, is_implicit = False):
    if is_implicit:
        filename =  f"Runs/{game}_implicit_q_{seed}_relabeled_trajectories.json"
    else:
        filename =  f"Runs/{game}_max_min_{seed}_relabeled_trajectories.json"
    return filename

def prompt_min_max(seed, game, is_implicit = False):
    if is_implicit:
        filename =  f"Runs/{game}_implicit_q_{seed}_prompt.json"
    else:
        filename =  f"Runs/{game}_max_min_{seed}_prompt.json"
    return filename
    
def dt_model_name(seed, game, is_implicit = False):
    if is_implicit:
        filename =  f"Runs/{game}_dt_implcit_{seed}_model.pth"
    else:
        filename =  f"Runs/{game}_dt_{seed}_model.pth"
    return filename

def behaviour_cloning_model_name(seed, game, is_implicit = False):
    if is_implicit:
        filename =  f"Runs/{game}_bc_implcit_{seed}_model.pth"
    else:
        filename =  f"Runs/{game}_bc_{seed}_model.pth"
    return filename