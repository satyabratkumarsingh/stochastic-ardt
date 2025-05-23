import torch

from decision_transformer.decision_transformer.training.trainer import Trainer


class SequenceTrainer(Trainer):
    def train_step(self):
        """
        Train a DT model for one training step.
        """
        states, actions, rewards, dones, returns, timesteps, attention_mask = self._get_batch()
        action_targets = torch.clone(actions)
        #self.embed_action = nn.Linear(in_features=3, out_features=128)

        # Before calling forward
        if rewards.shape[1] < states.shape[1]:
            print("Here =======")
            padding = torch.zeros(1, states.shape[1] - rewards.shape[1], 1, device=rewards.device)
            rewards = torch.cat([rewards, padding], dim=1)  # [1, 4, 1]

        # Compute returns_to_go (cumulative sum of future rewards)
        returns_to_go = torch.flip(torch.cumsum(torch.flip(rewards, [1]), dim=1), [1]) 

        _, action_preds, _ = self.model.forward(
            states, actions, rewards, returns_to_go, timesteps, attention_mask=attention_mask,
        )

        act_dim = action_preds.shape[2]
        action_preds_masked = action_preds.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]
        action_targets_masked = action_targets.reshape(-1, act_dim)[attention_mask.reshape(-1) > 0]

        loss = self.loss_fn(
            action_preds_masked,
            action_targets_masked
        )

        self.optimizer.zero_grad()
        loss.backward()
        self.gradients_clipper(self.model.parameters())
        self.optimizer.step()

        return loss.detach().cpu().item()