import torch
import torch.nn.functional as F

class MLPBCModel(torch.nn.Module):
    def __init__(
            self, 
            state_dim: int, 
            act_dim: int, 
            hidden_size: int, 
            n_layer: int, 
            dropout: float = 0.1, 
            max_length: int = 1,  # Set to 1 for single-step BC
            **kwargs
        ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size

        # Use only state_dim as input size for single-step BC
        input_size = self.state_dim  # Changed from max_length * self.state_dim
        
        layers = [torch.nn.Linear(input_size, hidden_size)]
        
        for _ in range(n_layer - 1):
            layers.extend([
                torch.nn.ReLU(),
                torch.nn.Dropout(dropout),
                torch.nn.Linear(hidden_size, hidden_size)
            ])
        
        layers.extend([
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(hidden_size, self.act_dim),
            # Remove Tanh for now - let the loss function handle output range
        ])

        self.model = torch.nn.Sequential(*layers)

    def forward(
            self, 
            states: torch.Tensor, 
            actions: torch.Tensor, 
            rewards: torch.Tensor, 
            attention_mask: torch.Tensor | None = None, 
            target_return: torch.Tensor | None = None
        ):
        # Take only the last state for each sequence
        batch_size = states.shape[0]
        last_states = states[:, -1, :]  # Shape: (batch_size, state_dim)
        
        # Pass through model
        actions = self.model(last_states).reshape(batch_size, 1, self.act_dim)
        
        return None, actions, None