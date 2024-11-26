# agent_PPO.py

import torch
import torch.nn.functional as F
from torch.distributions import Categorical
from PPO import PPO  # Assuming model.py contains the PPO model definition
from utils.config import *
from tensordict import TensorDict
import numpy as np


class Agent_PPO:
    def __init__(self, state_size,
                 action_size,
                 lr=LEARNING_RATE,
                 gamma=GAMMA,
                 epsilon=EPS_PPO,
                 tau=TAU,
                 beta=BETA,
                 mem_capacity=MEMORY_CAPACITY,
                 num_epochs=NUM_EPOCHS,
                 batch_size=BATCH_SIZE):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.tau = tau
        self.beta = beta
        self.epsilon = epsilon
        self.num_epochs = num_epochs
        self.batch_size = batch_size

        self.model = PPO(state_size, action_size).to(DEVICE)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)

        self.memory = TensorDict({}, batch_size=[]).to(DEVICE)

        for field in ["state", "action", "reward", "next_state", "done", "log_prob", "value"]:
            if field in ["state", "next_state"]:
                self.memory.set(
                    field,
                    torch.empty(
                        (0, *state_size),
                        dtype=torch.float32,
                        device=DEVICE
                    )
                )
            elif field == "value":
                self.memory.set(
                    field,
                    torch.empty(
                        (0, 1),
                        dtype=torch.float32,
                        device=DEVICE
                    )
                )
            elif field == "action":
                self.memory.set(
                    field,
                    torch.empty(
                        (0,),
                        dtype=torch.long,
                        device=DEVICE
                    )
                )
            else:
                self.memory.set(
                    field,
                    torch.empty(
                        (0,),
                        dtype=torch.float32,
                        device=DEVICE
                    )
                )



    def select_action(self, state):
        # if isinstance(state, np.ndarray) or isinstance(state, LazyFrames):
        #     state = torch.from_numpy(np.array(state)).float().to(DEVICE)
        # elif not torch.is_tensor(state):
        #     state = torch.tensor(state).float().to(DEVICE)
        # else:
        #     state = state.float().to(DEVICE)

        state = state.unsqueeze(0).to(self.model.device)
        logits, value = self.model(state)
        policy = F.softmax(logits, dim=-1)
        m = Categorical(policy)
        action = m.sample()
        log_prob = m.log_prob(action)
        action_int = action.item()
        return action_int, log_prob, value

    def store_in_memory(self, state, action, reward, next_state, done, log_prob, value):
        # Ensure state and next_state are tensors
        # state = torch.tensor(np.array(state), dtype=torch.float32)
        # next_state = torch.tensor(np.array(next_state), dtype=torch.float32)
        # print(f"state: {type(state)}: action = {type(action)}, reward = {type(reward)}, "
        #       f"next_state = {type(next_state)}, done = {type(done)}, log_prob = {type(log_prob)}, value = {type(value)}")
        #
        # print(f"state: {state.shape}, next_state = {next_state.shape}, log_prob = {log_prob.shape}, value = {value.shape}")

        # Prepare transition TensorDict
        transition = TensorDict({
            "state": state.unsqueeze(0),        # Shape: [1, *state_size]
            "action": torch.tensor([action], dtype=torch.long, device=DEVICE),
            "reward": torch.tensor([reward], dtype=torch.float32, device=DEVICE),
            "next_state": next_state.unsqueeze(0),
            "done": torch.tensor([done], dtype=torch.float32, device=DEVICE),
            "log_prob": log_prob,
            "value": value
        }, batch_size=[1]).to(DEVICE)

        # Append transition to memory by concatenating
        for key in transition.keys():
            existing = self.memory.get(key)
            new = transition.get(key)
            #print(f"Concatenating {key}: existing shape {existing.shape}, new shape {new.shape}")  # Debugging
            self.memory.set(
                key,
                torch.cat([existing, new], dim=0)
            )

        #self.memory.add(transition)


    def optimize_model(self):
        # Convert lists to tensors
        if len(self.memory) < self.batch_size:
            # Not enough data to perform optimization
            return

        # Extract all fields from memory
        states = self.memory.get("state")         # Shape: [num_steps, *state_size]
        actions = self.memory.get("action")       # Shape: [num_steps]
        rewards = self.memory.get("reward")       # Shape: [num_steps]
        next_states = self.memory.get("next_state")  # Shape: [num_steps, *state_size]
        dones = self.memory.get("done")           # Shape: [num_steps]
        old_log_probs = self.memory.get("log_prob")  # Shape: [num_steps]
        values = self.memory.get("value")         # Shape: [num_steps]


        # Compute returns and advantages
        returns, advantages = self.compute_gae(values, rewards, dones, next_states)

        # Optimize policy for K epochs
        for _ in range(self.num_epochs):
            # Create random mini-batches
            permutation = torch.randperm(len(states))
            for i in range(0, len(states), self.batch_size):
                indices = permutation[i:i + self.batch_size]
                batch_states = states[indices]
                batch_actions = actions[indices]
                batch_old_log_probs = old_log_probs[indices]
                batch_returns = returns[indices]
                batch_advantages = advantages[indices]

                # Get current policy outputs
                logits, values_new = self.model(batch_states)
                policy = F.softmax(logits, dim=-1)
                m = Categorical(policy)
                new_log_probs = m.log_prob(batch_actions.squeeze())

                # Compute ratio
                ratio = (new_log_probs - batch_old_log_probs.squeeze()).exp()

                # Compute surrogate losses
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.epsilon, 1.0 + self.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()

                # Compute critic loss
                critic_loss = F.smooth_l1_loss(values_new.squeeze(), batch_returns)

                # Compute entropy loss for exploration
                entropy_loss = m.entropy().mean()

                # Total loss
                total_loss = actor_loss + 0.5 * critic_loss - self.beta * entropy_loss

                # Backpropagation
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.optimizer.step()

        # Clear memory after optimization to maintain on-policy data
        self.memory = TensorDict({}, batch_size=[]).to(DEVICE)
        # Reinitialize empty tensors for each field
        for field in ["state", "action", "reward", "next_state", "done", "log_prob", "value"]:
            self.memory.set(field, torch.empty((0, *self.state_size), dtype=torch.float32, device=DEVICE) if field in ["state", "next_state"] else torch.empty((0,), dtype=torch.float32 if field != "action" else torch.long, device=DEVICE))

    def compute_gae(self, values, rewards, dones, next_states):
        """
        Compute Generalized Advantage Estimation (GAE).

        Args:
            values (torch.Tensor): Estimated values V(s).
            rewards (torch.Tensor): Rewards received.
            dones (torch.Tensor): Done flags.
            next_states (torch.Tensor): Next states.

        Returns:
            returns (torch.Tensor): Computed returns.
            advantages (torch.Tensor): Computed advantages.
        """
        with torch.no_grad():
            _, next_value = self.model(next_states)
            next_value = next_value.squeeze()

        returns = torch.zeros_like(rewards, device=DEVICE)
        advantages = torch.zeros_like(rewards, device=DEVICE)
        gae = 0
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + self.gamma * next_value * (1 - dones[step]) - values[step]
            gae = delta + self.gamma * self.tau * (1 - dones[step]) * gae
            advantages[step] = gae
            returns[step] = gae + values[step]
            next_value = values[step]

        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        return returns, advantages
