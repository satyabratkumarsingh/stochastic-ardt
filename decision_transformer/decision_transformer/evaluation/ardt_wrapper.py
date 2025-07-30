import torch

class ARDTModelWrapper(torch.nn.Module):
    """
    Wrapper to make ARDT model compatible with your evaluation interface.
    """
    
    def __init__(self, ardt_model: torch.nn.Module, obs_size: int, action_size: int):
        super().__init__()
        self.ardt_model = ardt_model
        self.obs_size = obs_size
        self.action_size = action_size
    
    def get_action(self, states, actions, rewards, returns_to_go=None, timesteps=None, batch_size=1):
        """
        Interface compatible with your evaluation code.
        This matches the expected signature from your evaluate_episode function.
        """
        # The ARDT model expects (obs, returns_to_go) for action prediction
        # Reshape inputs to match expected format
        batch_size, seq_len = states.shape[0], states.shape[1]
        
        # Use returns_to_go for conditioning (this is the key ARDT innovation!)
        if returns_to_go is not None:
            rtg_input = returns_to_go  # Shape: [batch_size, seq_len, 1]
        else:
            # Fallback - shouldn't happen in proper ARDT evaluation
            rtg_input = torch.zeros(batch_size, seq_len, 1, device=states.device)
        
        # Forward pass through ARDT model
        action_logits = self.ardt_model(states, rtg_input)  # [batch_size, seq_len, action_size]
        
        return action_logits
    
    def forward(self, obs, returns_to_go):
        """Standard forward pass."""
        return self.ardt_model(obs, returns_to_go)