import torch
import torch.nn as nn

class ValueNet(nn.Module):
    """
    A neural network to predict the state-value function V(s).
    This network can use either a simple Feedforward Neural Network (FFN)
    or a Long Short-Term Memory (LSTM) network.
    
    Args:
        obs_dim (int): The dimension of the observation space.
        is_lstm (bool): If True, use an LSTM layer; otherwise, use a simple FFN.
        train_args (dict, optional): A dictionary of training arguments, required for LSTM.
                                    Expected to contain 'hidden_dim'.
    """

    def __init__(self, obs_dim: int, is_lstm: bool = False, train_args: dict = None):
        super().__init__()
        self.is_lstm = is_lstm
        self.obs_dim = obs_dim
        
        if self.is_lstm:
            hidden_dim = 256 
            self.lstm = nn.LSTM(
                input_size=self.obs_dim,
                hidden_size=hidden_dim,
                batch_first=True
            )
            self.fc_layers = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1)  # Output is a single value, V(s)
            )
        else:
            # Simple FFN
            self.fc_layers = nn.Sequential(
                nn.Linear(self.obs_dim, 256),
                nn.ReLU(),
                nn.Linear(256, 256),
                nn.ReLU(),
                nn.Linear(256, 1)  # Output is a single value, V(s)
            )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of the ValueNet.

        Args:
            obs (torch.Tensor): The observation tensor.
                                Shape: [batch_size, seq_len, obs_dim]

        Returns:
            torch.Tensor: The predicted state-value.
                          Shape: [batch_size, seq_len, 1]
        """
        if self.is_lstm:
            # The LSTM processes the sequence
            lstm_out, _ = self.lstm(obs)
            # Apply fully connected layers to the LSTM output
            v_values = self.fc_layers(lstm_out)
        else:
            batch_size, seq_len, obs_dim = obs.shape
            
            # 2. Reshape obs for FFN: [batch_size * seq_len, obs_dim]
            # Use .reshape() for safety as discussed previously
            obs_reshaped = obs.reshape(batch_size * seq_len, obs_dim)

            # 3. Pass reshaped tensor through FFN layers
            v_values = self.fc_layers(obs_reshaped)

            # 4. Reshape output back to [batch_size, seq_len, 1]
            v_values = v_values.reshape(batch_size, seq_len, 1)
        
        return v_values