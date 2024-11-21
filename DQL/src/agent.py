# agent.py
import random
import torch.nn.functional as F
import torch.optim as optim
from DQN import DQN
from ReplayMemory import ReplayMemory
from DQL.utils.config import *
from collections import namedtuple

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'epsilon'))

class Agent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.policy_net = DQN(action_size).to(DEVICE)
        self.target_net = DQN(action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.steps_done = 0
        self.update_target_net()
        self.loss = 0

    def select_action(self, state):
        eps = EPS_END + (EPS_START - EPS_END) * \
              max(0, (EPS_DECAY - self.steps_done) / EPS_DECAY)
        self.steps_done += 1
        if random.random() > eps:
            with torch.no_grad():
                state = state.to(DEVICE)
                action = self.policy_net(state).argmax(1).cpu().unsqueeze(1)
        else:
            action = torch.tensor([[random.randrange(self.action_size)]], dtype=torch.long)
        return action, eps

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        # Convert batch arrays to tensors and move to DEVICE
        state_batch = torch.cat(batch.state).to(DEVICE)
        action_batch = torch.cat(batch.action).to(DEVICE)
        reward_batch = torch.cat(batch.reward).unsqueeze(1).to(DEVICE)
        next_state_batch = torch.cat(batch.next_state).to(DEVICE)
        done_batch = torch.cat(batch.done).to(DEVICE)

        # Compute Q(s_t, a)
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1})
        with torch.no_grad():
            next_state_values = self.target_net(next_state_batch).max(1)[0].unsqueeze(1)
            next_state_values[done_batch] = 0.0  # Zero out terminal states

        # Compute the expected Q values
        expected_state_action_values = reward_batch + (GAMMA * next_state_values)

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # Save loss for logging (optional)
        self.loss = loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())