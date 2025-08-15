import torch

from core_models.base_models.mlp import MLP

class RtgFFN(torch.nn.Module):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        adv_action_dim: int,
        hidden_dim: int = 512,
        include_adv: bool = False
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = action_dim
        self.adv_action_dim = adv_action_dim
        self.include_adv = include_adv
        self.hidden_dim = hidden_dim

        self.act_embed = torch.nn.Sequential(
            torch.nn.Linear(self.act_dim, hidden_dim), torch.nn.ReLU(),
        )
        if include_adv:
            self.adv_act_embed = torch.nn.Sequential(
                torch.nn.Linear(self.adv_action_dim, hidden_dim), torch.nn.ReLU(),
            )

        self.obs_embed = torch.nn.Sequential(
            torch.nn.Linear(self.state_dim, hidden_dim), torch.nn.ReLU(),
        )
        
        if include_adv:
            self.rtg_net = torch.nn.Sequential(
                torch.nn.Linear(3 * hidden_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, adv_action_dim)
            )
        else:
            # FIX: Change the final layer's output dimension to 1
            self.rtg_net = torch.nn.Sequential(
                torch.nn.Linear(2 * hidden_dim, hidden_dim), torch.nn.ReLU(),
                torch.nn.Linear(hidden_dim, 1)  # Predicts a single Q-value
            )

    def forward(self, observation: torch.Tensor, pr_action, adv_action=None):
        batch_size, seq_len = observation.shape[:2]
        obs_emd = self.obs_embed(observation)
        
        if not self.include_adv:
            # The rest of the logic is correct assuming the output of rtg_net is of size 1.
            act_all = torch.eye(self.act_dim, device=observation.device)
            act_all = act_all.view(1, 1, self.act_dim, self.act_dim).expand(batch_size, seq_len, -1, -1)
            act_all = act_all.reshape(batch_size * seq_len * self.act_dim, self.act_dim)
            
            act_emd = self.act_embed(act_all)
            obs_emd_expanded = obs_emd.unsqueeze(2).expand(-1, -1, self.act_dim, -1).reshape(batch_size * seq_len * self.act_dim, self.hidden_dim)
            
            # The output of this is now of shape (batch*seq_len*act_dim, 1) -> (768, 1)
            q = self.rtg_net(torch.cat([obs_emd_expanded, act_emd], dim=-1))
            
            # Reshape to (batch, seq_len, act_dim) -> (128, 3, 2) which is now valid
            return q.reshape(batch_size, seq_len, self.act_dim)
        else:
            act_emd = self.act_embed(pr_action)
            adv_emd = self.adv_act_embed(adv_action)
            return self.rtg_net(torch.cat([obs_emd, act_emd, adv_emd], dim=-1))
        
class RtgLSTM(torch.nn.Module):
    def __init__(
        self, 
        state_dim: int,
        action_dim: int,
        adv_action_dim: int, 
        model_args: dict, 
        hidden_dim: int = 64, 
        include_adv: bool = False, 
        is_lstm: bool = True
    ):
        super().__init__()
        self.include_adv = include_adv
        self.is_lstm = is_lstm
        self.action_dim = action_dim
        self.adv_action_dim = adv_action_dim

        hidden_dim = model_args['ret_obs_act_model_args']['hidden_size']
        self.hidden_dim = hidden_dim

        input_dim = state_dim + action_dim + adv_action_dim if include_adv else state_dim + action_dim
        self.ret_obs_act_model = MLP(input_dim, hidden_dim, **model_args['ret_obs_act_model_args'])
        
        if is_lstm:
            self.lstm_model = torch.nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            self.ret_model = torch.nn.Linear(hidden_dim, adv_action_dim if include_adv else action_dim)
        else:
            self.ret_model = MLP(hidden_dim, adv_action_dim if include_adv else action_dim, **model_args['ret_model_args'])

    def forward(self, obs: torch.Tensor, action: torch.Tensor, adv_action: torch.Tensor | None = None):
        batch_size, seq_len = obs.shape[:2]
        
        if self.include_adv:
            x = torch.cat([obs, action, adv_action], dim=-1).view(batch_size * seq_len, -1)
            ret_obs_act_reps = self.ret_obs_act_model(x).view(batch_size, seq_len, -1)
        else:
            act_all = torch.eye(self.action_dim, device=obs.device).view(1, 1, self.action_dim, -1)
            act_all = act_all.expand(batch_size, seq_len, -1, -1).reshape(batch_size * seq_len * self.action_dim, self.action_dim)
            obs_expanded = obs.unsqueeze(2).expand(-1, -1, self.action_dim, -1).reshape(batch_size * seq_len * self.action_dim, -1)
            
            x = torch.cat([obs_expanded, act_all], dim=-1)
            ret_obs_act_reps = self.ret_obs_act_model(x)
            
        if self.is_lstm:
            if self.include_adv:
                hidden = (
                    torch.zeros(1, batch_size, self.hidden_dim).to(obs.device), 
                    torch.zeros(1, batch_size, self.hidden_dim).to(obs.device)
                )
                x, _ = self.lstm_model(ret_obs_act_reps, hidden)
                ret_pred = self.ret_model(x)
            else:
                # Correctly reshape and permute for LSTM processing
                x = ret_obs_act_reps.reshape(batch_size, seq_len, self.action_dim, -1)
                x = x.permute(0, 2, 1, 3)
                x = x.reshape(batch_size * self.action_dim, seq_len, -1)
                
                # Correctly initialize hidden state for the new batch size
                hidden = (
                    torch.zeros(1, batch_size * self.action_dim, self.hidden_dim).to(obs.device),
                    torch.zeros(1, batch_size * self.action_dim, self.hidden_dim).to(obs.device)
                )

                x, _ = self.lstm_model(x, hidden)
                
                # Reshape output back to original dimensions
                x = x.reshape(batch_size, self.action_dim, seq_len, -1)
                x = x.permute(0, 2, 1, 3)
                ret_pred = self.ret_model(x)
                ret_pred = ret_pred.squeeze(-1) # Squeeze the last dimension to get the final shape
        else:
            ret_pred = self.ret_model(ret_obs_act_reps)
        
        return ret_pred