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
    def __init__(self, action_size,
                 input_dims,
                 num_actions,
                 lr=LEARNING_RATE,
                 gamma=GAMMA,
                 epsilon=EPS_START,
                 eps_decay=EPS_DECAY,
                 eps_min=EPS_END,
                 replay_buffer_capacity=MEMORY_CAPACITY,
                 batch_size=BATCH_SIZE,
                 sync_network_rate=SYNC_NETWORK_RATE):
        self.action_size = action_size
        self.policy_net = DQN(action_size).to(DEVICE)
        self.target_net = DQN(action_size).to(DEVICE)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.memory = ReplayMemory(replay_buffer_capacity)
        self.steps_done = 0
        self.update_target_net()
        self.loss = nn.HuberLoss()

        self.epsilon = epsilon
        self.eps_decay = eps_decay
        self.eps_min = eps_min

        self.gamma = gamma
        self.batch_size = batch_size

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon * self.eps_decay, self.eps_min)

    def select_action(self, state):
        self.decay_epsilon()

        self.steps_done += 1

        observation = torch.tensor(np.array(state), dtype=torch.float32) \
            .unsqueeze(0) \
            .to(self.policy_net.device)

        if random.random() > self.epsilon:
            with torch.no_grad():
                action = self.policy_net(observation).argmax(1).item()  # Return scalar
        else:
            action = random.randrange(self.action_size)

        return action, self.epsilon

    def sync_networks(self):
        if self.learn_step_counter % self.sync_network_rate == 0 and self.learn_step_counter > 0:
            self.target_network.load_state_dict(self.online_network.state_dict())

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        keys = ("state", "action", "reward", "next_state", "done")

        states, actions, rewards, next_states, dones = [transitions[key] for key in keys]
        # print(states.shape)
        # print(next_states.shape)

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
        expected_state_action_values = rewards + (self.gamma * next_state_values) * (1 - dones.float())

        # Optimize model
        loss = self.loss(state_action_values, expected_state_action_values)
        loss.backward()
        #torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1)
        self.optimizer.step()

        # # Save loss for logging (optional)
        # self.loss = loss.item()

    def update_target_net(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())