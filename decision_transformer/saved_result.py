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