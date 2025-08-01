import torch
import transformers
import torch.nn.functional as F
from core_models.decision_transformer.trajectory_gpt2 import GPT2Model

class DecisionTransformer(torch.nn.Module):
    """
    Predicts next step actions based on past state and actions, conditional on
    the returns-to-go.

    Args:
        state_dim (int): Dimensionality of the state space.
        act_dim (int): Dimensionality of the action space.
        hidden_size (int): Size of the hidden layers.
        max_length (int, optional): Maximum sequence length of states. Default is None.
        max_ep_len (int): Maximum episode length. Default is 4096.
        action_tanh (bool): Whether to apply Tanh activation to actions. Default is True.
        rtg_seq (bool): Whether to use returns-to-go in sequence. Default is True.
        **kwargs: Additional keyword arguments for the transformer configuration.
    """

    def __init__(
        self,
        state_dim: int,
        act_dim: int,
        hidden_size: int,
        max_length: int | None = None,
        max_ep_len: int = 4096,
        action_tanh: bool = True,
        rtg_seq: bool = True,
        action_type: str = 'continuous',
        **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.rtg_seq = rtg_seq
        self.action_type = action_type

        # Initialize transformer model configuration
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        # Embedding layers for timestep, return-to-go, state, and action
        self.embed_timestep = torch.nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)

        # Layer normalization for embeddings
        self.embed_ln = torch.nn.LayerNorm(hidden_size)

        # Output prediction layers for state, action, and return
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        if self.rtg_seq:
            self.predict_action = torch.nn.Sequential(
                *([torch.nn.Linear(hidden_size, self.act_dim)] + ([torch.nn.Tanh()] if action_tanh else []))
            )
        else:
            self.predict_action = torch.nn.Sequential(
                *([torch.nn.Linear(hidden_size * 2, hidden_size),
                   torch.nn.ReLU(),
                   torch.nn.Linear(hidden_size, hidden_size),
                   torch.nn.ReLU(),
                   torch.nn.Linear(hidden_size, self.act_dim)] + ([torch.nn.Tanh()] if action_tanh else []))
            )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        rewards: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            states (torch.Tensor): Tensor of input states (batch_size, seq_length, state_dim).
            actions (torch.Tensor): Tensor of input actions (batch_size, seq_length, act_dim).
            rewards (torch.Tensor): Tensor of input rewards (not used in the forward pass).
            returns_to_go (torch.Tensor): Tensor of returns-to-go (batch_size, seq_length, 1).
            timesteps (torch.Tensor): Tensor of timesteps (batch_size, seq_length).
            attention_mask (torch.Tensor, optional): Attention mask for the transformer. Default is None.

        Returns:
            tuple: Predicted next states, actions, and returns.
        """
        embed_per_timestep = 3 if self.rtg_seq else 2

        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.float, device=states.device)
        else:
            attention_mask = attention_mask.float()

        # Embed each modality: state, action, return-to-go, and timestep
        state_embeddings = self.embed_state(states)

        # Handle action embedding based on action_type
        if self.action_type == 'discrete':
            if actions.ndim == 2 and actions.dtype == torch.long:
                actions_for_embedding = F.one_hot(actions, num_classes=self.act_dim).float()
            elif actions.ndim == 3 and actions.shape[-1] == self.act_dim:
                actions_for_embedding = actions.float()
            else:
                raise ValueError(f"Unexpected discrete action shape or dtype: {actions.shape}, {actions.dtype}")
        else:
            actions_for_embedding = actions.float()
        
        action_embeddings = self.embed_action(actions_for_embedding)
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)
        
        state_embeddings += time_embeddings
        action_embeddings += time_embeddings
        if self.rtg_seq:
            returns_embeddings += time_embeddings

        # Stack and reshape embeddings
        if self.rtg_seq:
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, embed_per_timestep * seq_length, self.hidden_size)
        else:
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, embed_per_timestep * seq_length, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # Adjust attention mask to fit stacked inputs
        if self.rtg_seq:
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, embed_per_timestep * seq_length)
        else:
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, embed_per_timestep * seq_length)

        # Transformer forward pass
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, embed_per_timestep, self.hidden_size).permute(0, 2, 1, 3)
        
        if self.rtg_seq:
            return_preds = self.predict_return(x[:, 2])
            state_preds = self.predict_state(x[:, 2])
            action_preds = self.predict_action(x[:, 1])
        else:
            state_preds = self.predict_state(x[:, 1])
            state_return = torch.cat((x[:, 0], returns_embeddings), dim=-1)
            action_preds = self.predict_action(state_return)
            return_preds = self.predict_return(x[:, 0])  # FIX: Added missing return_preds

        return state_preds, action_preds, return_preds

    def get_action(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        batch_size: int,
        **kwargs
    ) -> torch.Tensor:
        # Reshape inputs
        states = states.reshape(batch_size, -1, self.state_dim)
        actions = actions.reshape(batch_size, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(batch_size, -1, 1)
        timesteps = timesteps.reshape(batch_size, -1)

        # Truncate sequences to max_length and pad if needed
        attention_mask = None

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            attention_mask = torch.cat([
                torch.zeros((batch_size, self.max_length - states.shape[1])),
                torch.ones((batch_size, states.shape[1]))
            ], dim=1)
            attention_mask = attention_mask.to(dtype=torch.float, device=states.device)  # FIX: Removed reshape

            # Pad states, actions, returns_to_go, and timesteps to max_length
            states = torch.cat([
                torch.zeros((states.shape[0], self.max_length - states.shape[1], self.state_dim), device=states.device),
                states
            ], dim=1).to(dtype=torch.float32)

            actions = torch.cat([
                torch.zeros((actions.shape[0], self.max_length - actions.shape[1], self.act_dim), device=actions.device),
                actions
            ], dim=1).to(dtype=torch.float32)

            returns_to_go = torch.cat([
                torch.zeros((returns_to_go.shape[0], self.max_length - returns_to_go.shape[1], 1), device=returns_to_go.device),
                returns_to_go
            ], dim=1).to(dtype=torch.float32)

            timesteps = torch.cat([
                torch.zeros((timesteps.shape[0], self.max_length - timesteps.shape[1]), device=timesteps.device),
                timesteps
            ], dim=1).to(dtype=torch.long)

        # Get predictions and return
        _, action_preds, _ = self.forward(
            states=states, 
            actions=actions, 
            returns_to_go=returns_to_go, 
            timesteps=timesteps, 
            rewards=None, 
            attention_mask=attention_mask, 
            **kwargs
        )
        
        return action_preds