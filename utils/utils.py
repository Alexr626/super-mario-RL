import time
from datetime import datetime
import torch
import csv
import os

def get_current_date_time_string():
    return datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def save_model(model, filename):
    """Save the model parameters to a file."""
    torch.save(model.state_dict(), filename)

def load_model(model, filename):
    """Load model parameters from a file."""
    model.load_state_dict(torch.load(filename))

def create_save_files_directories(timestamp, job_id, type):
    # Create frames directory if needed

    os.makedirs(f"frames", exist_ok=True)
    frames_dir = f"frames/{job_id}_{type}_{timestamp}"
    os.makedirs(frames_dir, exist_ok=True)

    # Create episode_logs directory
    episode_logs_dir = f"episode_logs"
    os.makedirs(episode_logs_dir, exist_ok=True)

    # Create episode_log.csv
    log_filename = f'episode_logs/{job_id}_{type}_{timestamp}_episode_log.csv'
    if not os.path.exists(log_filename):
        with open(log_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Updated header with level information
            writer.writerow(['episode', 'total_reward', 'duration', 'start_level', 'end_level'])

    # Set transitions.csv path
    transitions_filename = f'episode_logs/{job_id}_{type}_{timestamp}/transitions.h5'

    os.makedirs(f"checkpoints", exist_ok=True)
    checkpoints_dir = f"checkpoints/{job_id}_{type}_{timestamp}"
    os.makedirs(checkpoints_dir, exist_ok=True)

    return frames_dir, checkpoints_dir, log_filename, transitions_filename

def episode_writer_thread(episode_queue, log_filename):
    with open(log_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        while True:
            episode_data = episode_queue.get()
            if episode_data is None:
                break  # Exit signal
            writer.writerow(episode_data)
            episode_queue.task_done()

def transition_writer_thread(transition_queue, transitions_filename):
    with open(transitions_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        buffer = []
        while True:
            transition_data = transition_queue.get()
            if transition_data is None:
                break  # Exit signal
            buffer.append(transition_data)
            if len(buffer) >= 1000:  # Write in batches of 1000
                writer.writerows(buffer)
                buffer = []
            transition_queue.task_done()
        # Write any remaining transitions
        if buffer:
            writer.writerows(buffer)


class Timer():
    def __init__(self):
        self.times = []

    def start(self):
        self.t = time.time()

    def print(self, msg=''):
        print(f"Time taken: {msg}", time.time() - self.t)

    def get(self):
        return time.time() - self.t

    def store(self):
        self.times.append(time.time() - self.t)

    def average(self):
        return sum(self.times) / len(self.times)