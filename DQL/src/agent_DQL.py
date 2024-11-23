# agent_DQL.py
import random

import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from DQN import DQN
from ReplayMemory import ReplayMemory
from utils.config import *
from collections import namedtuple

class Agent_DQL:
    def __init__(self, action_size):
        self.action_size = action_size
        self.policy_net = DQN(action_size).to(DEVICE)
        self.target_net = DQN(action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.memory = ReplayMemory(MEMORY_CAPACITY)
        self.steps_done = 0
        self.update_target_net()
        self.loss = nn.HuberLoss()

    def select_action(self, state):
        eps = EPS_END + (EPS_START - EPS_END) * \
              max(0, (EPS_DECAY - self.steps_done) / EPS_DECAY)

        self.steps_done += 1

        observation = torch.tensor(np.array(state), dtype=torch.float32) \
            .to(self.policy_net.device)

        if random.random() > eps:
            with torch.no_grad():
                action = self.policy_net(observation).argmax(1).item()  # Return scalar
        else:
            action = random.randrange(self.action_size)

        return action, eps

    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def optimize_model(self):
        if len(self.memory) < BATCH_SIZE:
            return

        transitions = self.memory.sample(BATCH_SIZE)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [transitions[key] for key in keys]
        print(states.shape)
        print(next_states.shape)

        states = states.squeeze(-1)
        next_states = next_states.squeeze(-1)
        actions = actions.unsqueeze(-1)
        # print(states.shape)
        # print(next_states.shape)
        # Compute Q(s_t, a)
        state_action_values = self.policy_net(states).gather(1, actions)

        # Compute V(s_{t+1})
        with torch.no_grad():
            next_state_values = self.target_net(next_states).max(1)[0].unsqueeze(1)
            next_state_values[dones] = 0.0  # Zero out terminal states

        # Compute the expected Q values
        expected_state_action_values = rewards + (GAMMA * next_state_values) * (1 - dones.float())

        # Optimize model
        loss = self.loss(state_action_values, expected_state_action_values)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # # Save loss for logging (optional)
        # self.loss = loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())