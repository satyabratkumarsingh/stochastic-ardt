import os

def pkl_name_min_max_relabeled(seed, game, method):
    if method == 'implicit_q':
       filename = f"Runs/{game}_implicit_q_{seed}_relabeled_trajectories.pkl"
    elif method == 'max_min':
        filename =  f"Runs/{game}_max_min_{seed}_relabeled_trajectories.pkl"
    elif method == 'cql':
        filename =  f"Runs/{game}_cql_{seed}_relabeled_trajectories.pkl"
    else:
        raise ValueError(f"Unknown method: {method}")
    return filename
    
def json_name_min_max_relabeled(seed, game, method):
    if method == 'implicit_q':
        filename =  f"Runs/{game}_implicit_q_{seed}_relabeled_trajectories.json"
    elif method == 'max_min':
        filename =  f"Runs/{game}_max_min_{seed}_relabeled_trajectories.json"
    elif method == 'cql':
        filename =  f"Runs/{game}_cql_{seed}_relabeled_trajectories.json"
    else:
        raise ValueError(f"Unknown method: {method}")
    return filename

def prompt_min_max(seed, game, method):
    if method == 'implicit_q':
        filename =  f"Runs/{game}_implicit_q_{seed}_prompt.json"
    elif method == 'max_min':
        filename =  f"Runs/{game}_max_min_{seed}_prompt.json"
    elif method == 'cql':
        filename =  f"Runs/{game}_cql_{seed}_prompt.json"
    else:
        raise ValueError(f"Unknown method: {method}")
    return filename
    
def dt_model_name(seed, game, method):
    if method == 'implicit_q':
        filename =  f"Runs/{game}_dt_implcit_{seed}_model.pth"
    elif method == 'max_min':
        filename =  f"Runs/{game}_dt_max_min_{seed}_model.pth"
    elif method == 'cql':
        filename =  f"Runs/{game}_dt_cql_{seed}_model.pth"
    else:
        raise ValueError(f"Unknown method: {method}")
    return filename

def behaviour_cloning_model_name(seed, game, method):
    if method == 'implicit_q':
        filename =  f"Runs/{game}_bc_implcit_{seed}_model.pth"
    elif method == 'max_min':
        filename =  f"Runs/{game}_bc_max_min_{seed}_model.pth"
    elif method == 'cql':
        filename =  f"Runs/{game}_bc_cql_{seed}_model.pth"
    else:
        raise ValueError(f"Unknown method: {method}")
    filename =  f"Runs/{game}_bc_{seed}_model.pth"
    return filename