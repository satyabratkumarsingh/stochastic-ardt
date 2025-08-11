import torch
import transformers
import torch.nn.functional as F
from core_models.decision_transformer.trajectory_gpt2 import GPT2Model

class DecisionTransformer(torch.nn.Module):
    """
    FIXED Decision Transformer with proper return conditioning.
    
    Key fixes:
    1. Better return embedding that preserves negative values
    2. Stronger return-action conditioning 
    3. Improved weight initialization
    4. Better sequence handling for single-step predictions
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
        action_type: str = 'discrete',
        **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.rtg_seq = rtg_seq
        self.action_type = action_type

        # Initialize transformer
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        # FIXED: Better embeddings with stronger return conditioning
        self.embed_timestep = torch.nn.Embedding(max_ep_len, hidden_size)
        
        # CRITICAL FIX: Return embedding that works with negative values
        self.embed_return = torch.nn.Sequential(
            torch.nn.Linear(1, hidden_size),
            torch.nn.Tanh(),  # Preserves negative returns unlike ReLU
            torch.nn.Dropout(0.1),
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.LayerNorm(hidden_size)
        )
        
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_ln = torch.nn.LayerNorm(hidden_size)

        # CRITICAL FIX: Action prediction with strong return conditioning
        if self.rtg_seq:
            if action_type == 'discrete':
                # Enhanced architecture for better return conditioning
                self.predict_action = torch.nn.Sequential(
                    torch.nn.Linear(hidden_size, hidden_size * 2),  # Increased capacity
                    torch.nn.GELU(),  # Better activation than ReLU
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(hidden_size * 2, hidden_size),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_size, self.act_dim)
                )
            else:
                # For continuous actions
                layers = [
                    torch.nn.Linear(hidden_size, hidden_size * 2),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.1), 
                    torch.nn.Linear(hidden_size * 2, hidden_size),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_size, self.act_dim)
                ]
                if action_tanh:
                    layers.append(torch.nn.Tanh())
                self.predict_action = torch.nn.Sequential(*layers)
        else:
            # Non-sequential with concatenated features
            input_dim = hidden_size * 2  # State + Return features
            if action_type == 'discrete':
                self.predict_action = torch.nn.Sequential(
                    torch.nn.Linear(input_dim, hidden_size * 2),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(hidden_size * 2, hidden_size),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_size, self.act_dim)
                )
            else:
                layers = [
                    torch.nn.Linear(input_dim, hidden_size * 2),
                    torch.nn.GELU(),
                    torch.nn.Dropout(0.1),
                    torch.nn.Linear(hidden_size * 2, hidden_size),
                    torch.nn.GELU(),
                    torch.nn.Linear(hidden_size, self.act_dim)
                ]
                if action_tanh:
                    layers.append(torch.nn.Tanh())
                self.predict_action = torch.nn.Sequential(*layers)
                
        # Other prediction heads
        self.predict_state = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size // 2, self.state_dim)
        )
        
        self.predict_return = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size // 2),
            torch.nn.GELU(),
            torch.nn.Linear(hidden_size // 2, 1)
        )

        # FIXED: Better weight initialization
        self.apply(self._init_weights)

    def _init_weights(self, module):
        """Improved weight initialization for better training"""
        if isinstance(module, torch.nn.Linear):
            # Xavier initialization works better for deep networks
            torch.nn.init.xavier_normal_(module.weight, gain=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, torch.nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)

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
        Forward pass that DOES NOT allow the model to see the *current* true action
        when predicting the action at that timestep. We shift action embeddings by 1.
        """
        embed_per_timestep = 3 if self.rtg_seq else 2
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.bool, device=states.device)

        # Clamp timesteps to valid range
        timesteps = torch.clamp(timesteps, 0, self.embed_timestep.num_embeddings - 1)

        # Embed states
        state_embeddings = self.embed_state(states)

        # Handle discrete actions properly and create embeddings
        if self.action_type == 'discrete':
            # actions can be long indices or already one-hot float vectors
            if actions.dtype == torch.long:
                actions_one_hot = F.one_hot(actions, num_classes=self.act_dim).float()
            else:
                actions_one_hot = actions.float()
        else:
            actions_one_hot = actions.float()

        action_embeddings = self.embed_action(actions_one_hot)  # shape (B, T, H)

        # ---------------------------
        # CRITICAL: Shift action embeddings right by 1 so that position t contains action at t-1
        # First position is zeros (no previous action).
        # ---------------------------
        shifted_action_embeddings = torch.zeros_like(action_embeddings)
        if seq_length > 1:
            shifted_action_embeddings[:, 1:, :] = action_embeddings[:, :-1, :]

        # use shifted embeddings from here on (prevents label leakage)
        action_embeddings = shifted_action_embeddings

        # Return and time embeddings
        returns_embeddings = self.embed_return(returns_to_go)
        time_embeddings = self.embed_timestep(timesteps)

        # Combine embeddings (same as before)
        if self.rtg_seq:
            returns_embeddings = returns_embeddings + time_embeddings * 0.3
            state_embeddings = state_embeddings + time_embeddings * 0.7
            action_embeddings = action_embeddings + time_embeddings * 0.7
        else:
            returns_embeddings = returns_embeddings + time_embeddings
            state_embeddings = state_embeddings + time_embeddings
            action_embeddings = action_embeddings + time_embeddings

        # Stack them (RTG first)
        if self.rtg_seq:
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, embed_per_timestep * seq_length, self.hidden_size)
        else:
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings), dim=1
            ).permute(0, 2, 1, 3).reshape(batch_size, embed_per_timestep * seq_length, self.hidden_size)

        stacked_inputs = self.embed_ln(stacked_inputs)

        # Stacked attention mask
        if self.rtg_seq:
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, embed_per_timestep * seq_length)
        else:
            stacked_attention_mask = torch.stack(
                (attention_mask, attention_mask), dim=1
            ).permute(0, 2, 1).reshape(batch_size, embed_per_timestep * seq_length)

        # Transformer forward
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask.float(),
        )

        x = transformer_outputs['last_hidden_state']
        x = x.reshape(batch_size, seq_length, embed_per_timestep, self.hidden_size).permute(0, 2, 1, 3)

        if self.rtg_seq:
            return_features = x[:, 0]   # RTG position
            state_features = x[:, 1]    # State position
            action_features = x[:, 2]   # Action position (this is now previous-action embedding context)

            # Predict
            return_preds = self.predict_return(action_features)
            state_preds = self.predict_state(action_features)

            # Combine return and state for action prediction (action_features uses prev action context)
            combined_features = state_features + return_features * 0.8
            action_preds = self.predict_action(combined_features)

        else:
            state_features = x[:, 0]
            action_features = x[:, 1]

            state_preds = self.predict_state(action_features)
            return_preds = self.predict_return(state_features)

            # For non-sequential path we need a return_features variable; keep interface consistent
            action_preds = self.predict_action(torch.cat((state_features, return_features), dim=-1))

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
        # Reshape
        states = states.reshape(batch_size, -1, self.state_dim)
        actions = actions.reshape(batch_size, -1, self.act_dim)
        returns_to_go = returns_to_go.reshape(batch_size, -1, 1)
        timesteps = timesteps.reshape(batch_size, -1)

        # Ensure we don't include the current (unknown) action at the last step:
        # shift actions right by 1 (first slot zeros)
        shifted_actions = torch.zeros_like(actions)
        if actions.shape[1] > 1:
            shifted_actions[:, 1:, :] = actions[:, :-1, :]

        # Handle max_length truncation/padding (same logic as before)
        attention_mask = None
        if self.max_length is not None:
            states = states[:, -self.max_length:]
            shifted_actions = shifted_actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            # create attention mask
            attention_mask = torch.cat([
                torch.zeros((batch_size, self.max_length - states.shape[1])),
                torch.ones((batch_size, states.shape[1]))
            ], dim=1).to(dtype=torch.bool, device=states.device)

            pad_length = self.max_length - states.shape[1]
            if pad_length > 0:
                states = torch.cat([
                    torch.zeros((batch_size, pad_length, self.state_dim), device=states.device),
                    states
                ], dim=1)

                shifted_actions = torch.cat([
                    torch.zeros((batch_size, pad_length, self.act_dim), device=actions.device),
                    shifted_actions
                ], dim=1)

                returns_to_go = torch.cat([
                    torch.zeros((batch_size, pad_length, 1), device=returns_to_go.device),
                    returns_to_go
                ], dim=1)

                timesteps = torch.cat([
                    torch.zeros((batch_size, pad_length, 1), device=timesteps.device),
                    timesteps
                ], dim=1).to(dtype=torch.long)

        # Call forward using shifted actions (previous-action context)
        _, action_preds, _ = self.forward(
            states=states,
            actions=shifted_actions,
            returns_to_go=returns_to_go,
            timesteps=timesteps,
            attention_mask=attention_mask,
            **kwargs
        )

        return action_preds