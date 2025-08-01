import torch
import torch.nn as nn
import torch.nn.functional as F
from return_transforms.models.basic.mlp import MLP

class ValueNet(nn.Module):
    def __init__(self, state_dim: int, model_args: dict = None, hidden_dim: int = 64, is_lstm: bool = False):
        super().__init__()
        self.state_dim = state_dim
        self.is_lstm = is_lstm
        if state_dim <= 0:
            raise ValueError("state_dim must be positive")
        if model_args is None and is_lstm:
            raise ValueError("model_args dictionary is required for LSTM mode")
        hidden_dim = model_args.get('hidden_size', hidden_dim) if model_args else hidden_dim
        self.hidden_dim = hidden_dim
        self.obs_model = MLP(state_dim, hidden_dim, **({'hidden_size': 64, 'num_layers': 2}))
        if is_lstm:
            self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.value_head = nn.Linear(hidden_dim, 1)
        else:
            self.value_head = MLP(hidden_dim, 1, **({'hidden_size': 64, 'num_layers': 2}))

    def forward(self, obs):
        batch_size, seq_len, obs_size = obs.shape
        
        if self.is_lstm:
            # Process observations through LSTM
            obs_reps = self.obs_model(obs.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, -1)
            lstm_out, _ = self.lstm(obs_reps)
            value_pred = self.value_head(lstm_out)
        else:
            # Process observations through MLP
            obs_reps = self.obs_model(obs.reshape(batch_size * seq_len, -1)).reshape(batch_size, seq_len, -1)
            value_pred = self.value_head(obs_reps)[:, :, None]  # [batch_size, seq_len, 1]
        
        return value_pred

