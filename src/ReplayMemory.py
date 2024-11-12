# ReplayMemory.py
from collections import deque, namedtuple
import random

Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done', 'epsilon'))

class ReplayMemory:
    def __init__(self, capacity):

        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done, epsilon):
        # Save in memory for training
        self.memory.append(Transition(state, action, reward, next_state, done, epsilon))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    # Getter functions (optional, based on your needs)
    def get_num_transitions(self):
        return len(self.memory)

    def get_transition_memory(self):
        return list(self.memory)