# ReplayMemory.py
from collections import deque, namedtuple
import random
import torch
import numpy as np
from tensordict import TensorDict
from torchrl.data import TensorDictReplayBuffer, LazyMemmapStorage

class ReplayMemory:
    def __init__(self, capacity):
        storage = LazyMemmapStorage(capacity)
        self.memory = TensorDictReplayBuffer(storage=storage)

    def push(self, state, action, reward, next_state, done, epsilon):
        # Save in memory for training
        self.memory.add(TensorDict({
            "state": torch.tensor(np.array(state), dtype=torch.float32),
            "action": torch.tensor(action),
            "reward": reward,
            "next_state": torch.tensor(np.array(next_state), dtype=torch.float32),
            "done": done
        }, batch_size=[]))

    def sample(self, batch_size):
        return self.memory.sample(batch_size)

    def __len__(self):
        return len(self.memory)

    # Getter functions (optional, based on your needs)
    def get_num_transitions(self):
        return len(self.memory)

    def get_transition_memory(self):
        return list(self.memory)