import random
from collections import deque, namedtuple  # Ensure deque is imported from collections
import os
import csv
Transition = namedtuple('Transition', ('state', 'action', 'reward', 'next_state', 'done'))
class ReplayMemory:
    def __init__(self, capacity, filename='transitions.csv', write_batch_size=1000):
        self.capacity = capacity
        self.memory = deque(maxlen=capacity)
        self.filename = filename
        self.write_batch_size = write_batch_size
        self.write_buffer = []  # Buffer for batching writes

        # Initialize the CSV file
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['state', 'action', 'reward', 'next_state', 'done', 'epsilon'])

    def push(self, state, action, reward, next_state, done, epsilon):
        """Save a transition to memory and write to disk."""
        # Save in memory for training
        self.memory.append(Transition(state, action, reward, next_state, done))

        # Add transition to write buffer
        self.write_buffer.append((
            state.cpu().numpy().tolist(),
            action.cpu().numpy().tolist(),
            reward.item(),
            next_state.cpu().numpy().tolist(),
            done.item(),
            epsilon
        ))

        # Check if buffer is full
        if len(self.write_buffer) >= self.write_batch_size:
            self.flush_to_disk()

    def flush_to_disk(self):
        """Write buffered transitions to disk."""
        with open(self.filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerows(self.write_buffer)
        # Clear the buffer
        self.write_buffer.clear()

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

    def close(self):
        """Ensure all data is written to disk."""
        if self.write_buffer:
            self.flush_to_disk()