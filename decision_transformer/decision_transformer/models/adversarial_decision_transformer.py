import torch
import transformers
from decision_transformer.decision_transformer.models.trajectory_gpt2 import GPT2Model

class AdversarialDecisionTransformer(torch.nn.Module):
    def __init__(
            self,
            state_dim: int,
            act_dim: int,
            adv_act_dim: int,
            hidden_size: int,
            max_length: int | None = None,
            max_ep_len: int = 4096,
            action_tanh: bool = True,
            rtg_seq: bool = True,
            **kwargs
    ):
        super().__init__()
        self.state_dim = state_dim
        self.act_dim = act_dim
        self.adv_act_dim = adv_act_dim
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.rtg_seq = rtg_seq
        
        # Initialize GPT2 transformer model configuration
        config = transformers.GPT2Config(
            vocab_size=1,
            n_embd=hidden_size,
            **kwargs
        )
        self.transformer = GPT2Model(config)

        # Embedding layers
        self.embed_timestep = torch.nn.Embedding(max_ep_len, hidden_size)
        self.embed_return = torch.nn.Linear(1, hidden_size)
        self.embed_state = torch.nn.Linear(self.state_dim, hidden_size)
        self.embed_action = torch.nn.Linear(self.act_dim, hidden_size)
        self.embed_adv_action = torch.nn.Linear(self.adv_act_dim, hidden_size)
        self.embed_ln = torch.nn.LayerNorm(hidden_size)

        # Output layers for existing predictions
        self.predict_state = torch.nn.Linear(hidden_size, self.state_dim)
        if self.rtg_seq:
            self.predict_action = torch.nn.Sequential(
                *([torch.nn.Linear(hidden_size, self.act_dim)] + ([torch.nn.Tanh()] if action_tanh else []))
            )
        else:
            self.predict_action = torch.nn.Sequential(
                *([
                    torch.nn.Linear(hidden_size * 2, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, hidden_size),
                    torch.nn.ReLU(),
                    torch.nn.Linear(hidden_size, self.act_dim)
                ] + ([torch.nn.Tanh()] if action_tanh else []))
            )
        self.predict_return = torch.nn.Linear(hidden_size, 1)

        # New output heads for V(s) and Q(s, a)
        self.value_head = torch.nn.Linear(hidden_size, 1)  # Predicts V(s)
        self.q_head = torch.nn.Linear(hidden_size * 2, 1)  # Predicts Q(s, a)

    def forward(
        self,
        states: torch.Tensor,
        actions: torch.Tensor,
        adv_actions: torch.Tensor,
        rewards: torch.Tensor,
        returns_to_go: torch.Tensor,
        timesteps: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        return_q_v: bool = False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        embed_per_timestep = 4 if self.rtg_seq else 3
        batch_size, seq_length = states.shape[0], states.shape[1]

        if attention_mask is None:
            attention_mask = torch.ones((batch_size, seq_length), dtype=torch.long, device=states.device)

        # Embed inputs
        state_embeddings = self.embed_state(states) + self.embed_timestep(timesteps)
        action_embeddings = self.embed_action(actions) + self.embed_timestep(timesteps) if actions is not None else None
        adv_action_embeddings = self.embed_adv_action(adv_actions) + self.embed_timestep(timesteps) if adv_actions is not None else None
        returns_embeddings = self.embed_return(returns_to_go) + self.embed_timestep(timesteps) if self.rtg_seq else None

        # Stack inputs for transformer
        if self.rtg_seq:
            stacked_inputs = torch.stack(
                (returns_embeddings, state_embeddings, action_embeddings, adv_action_embeddings), dim=1
            )
        else:
            stacked_inputs = torch.stack(
                (state_embeddings, action_embeddings, adv_action_embeddings), dim=1
            )
        stacked_inputs = stacked_inputs.permute(0, 2, 1, 3).reshape(batch_size, -1, self.hidden_size)
        stacked_inputs = self.embed_ln(stacked_inputs)

        # Adjust attention mask
        stacked_attention_mask = torch.stack(
            [attention_mask] * embed_per_timestep, dim=1
        ).permute(0, 2, 1).reshape(batch_size, -1)

        # Transformer forward pass
        transformer_outputs = self.transformer(
            inputs_embeds=stacked_inputs,
            attention_mask=stacked_attention_mask,
        )
        x = transformer_outputs['last_hidden_state']

        # Reshape output
        x = x.reshape(batch_size, seq_length, embed_per_timestep, self.hidden_size).permute(0, 2, 1, 3)

        # Existing predictions
        return_preds = self.predict_return(x[:, 2])
        state_preds = self.predict_state(x[:, 2])
        action_preds = self.predict_action(x[:, 1])

        if return_q_v:
            # Compute V(s) from state embeddings
            v_values = self.value_head(x[:, 1])  # Use state position
            # Compute Q(s, a) from state and action embeddings
            q_values = None
            if action_embeddings is not None:
                sa_embed = torch.cat([x[:, 1], action_embeddings], dim=-1)
                q_values = self.q_head(sa_embed)
            return state_preds, action_preds, return_preds, v_values, q_values

        return state_preds, action_preds, return_preds, None, None

    def get_action(self, states, actions, adv_actions, rewards, returns_to_go, timesteps, batch_size, **kwargs):
        # (Existing implementation remains unchanged unless you want to return V(s) or Q(s, a) here)
        states = states.reshape(batch_size, -1, self.state_dim)
        actions = actions.reshape(batch_size, -1, self.act_dim)
        adv_actions = adv_actions.reshape(batch_size, -1, self.adv_act_dim)
        returns_to_go = returns_to_go.reshape(batch_size, -1, 1)
        timesteps = timesteps.reshape(batch_size, -1)

        if self.max_length is not None:
            states = states[:, -self.max_length:]
            actions = actions[:, -self.max_length:]
            adv_actions = adv_actions[:, -self.max_length:]
            returns_to_go = returns_to_go[:, -self.max_length:]
            timesteps = timesteps[:, -self.max_length:]

            attention_mask = torch.cat([
                torch.zeros((batch_size, self.max_length - states.shape[1]), device=states.device),
                torch.ones((batch_size, states.shape[1]), device=states.device)
            ], dim=1).to(dtype=torch.long)

            states = torch.cat([
                torch.zeros((batch_size, self.max_length - states.shape[1], self.state_dim), device=states.device),
                states
            ], dim=1).to(dtype=torch.float32)

            actions = torch.cat([
                torch.zeros((batch_size, self.max_length - actions.shape[1], self.act_dim), device=actions.device),
                actions
            ], dim=1).to(dtype=torch.float32)

            adv_actions = torch.cat([
                torch.zeros((batch_size, self.max_length - adv_actions.shape[1], self.adv_act_dim), device=adv_actions.device),
                adv_actions
            ], dim=1).to(dtype=torch.float32)

            returns_to_go = torch.cat([
                torch.zeros((batch_size, self.max_length - returns_to_go.shape[1], 1), device=returns_to_go.device),
                returns_to_go
            ], dim=1).to(dtype=torch.float32)

            timesteps = torch.cat([
                torch.zeros((batch_size, self.max_length - timesteps.shape[1]), device=timesteps.device),
                timesteps
            ], dim=1).to(dtype=torch.long)

        state_preds, action_preds, return_preds, _, _ = self.forward(
            states, actions, adv_actions, None, returns_to_go, timesteps, attention_mask=attention_mask, **kwargs
        )

        return action_preds