# PPO.py

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from utils.config import DEVICE


class PPO(nn.Module):
    def __init__(self, state_size, action_size):
        super(PPO, self).__init__()

        # Convolutional layers for processing image inputs
        self.conv = nn.Sequential(
            nn.Conv2d(state_size[0], 32, kernel_size=8, stride=4),  # Output: (32, 20, 20)
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),              # Output: (64, 9, 9)
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),              # Output: (64, 7, 7)
            nn.ReLU()
        )

        # Compute the size of the flattened convolutional output
        conv_out_size = self._get_conv_out(state_size)

        # Fully connected layer
        self.fc = nn.Sequential(
            nn.Linear(conv_out_size, 512),
            nn.ReLU()
        )

        # Policy head: outputs action logits
        self.policy_head = nn.Linear(512, action_size)

        # Value head: outputs state value
        self.value_head = nn.Linear(512, 1)

        self.device = DEVICE
        self.to(self.device)

    def _get_conv_out(self, shape):
        o = self.conv(torch.zeros(1, *shape))
        return int(np.prod(o.size()))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.fc(x)
        logits = self.policy_head(x)
        value = self.value_head(x)
        return logits, value