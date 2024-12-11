import time
import importlib
from datetime import datetime
import torch
import csv
import os
import glob

def get_current_date_time_string():
    return datetime.now().strftime("%Y-%m-%d-%H_%M_%S")

def save_checkpoint(model, optimizer, filename):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }
    torch.save(checkpoint, filename)

def load_model(model, optimizer, filename):
    """Load model parameters from a file."""
    model.load_state_dict(torch.load(filename))
    optimizer.load_state_dict(torch)

def create_save_files_directories(timestamp, job_id, type, config_version):
    # Create frames directory if needed

    frames_dir = f"results/frames/{type}_{timestamp}_conf{config_version}"
    os.makedirs(frames_dir, exist_ok=True)

    # Create episode_logs directory
    episode_logs_dir = f"results/episode_logs/{type}_{timestamp}_conf{config_version}"
    os.makedirs(episode_logs_dir, exist_ok=True)

    # Create episode_log.csv
    log_filename = f'results/episode_logs/{type}_{timestamp}_conf{config_version}/{job_id}_{type}_episode_log.csv'
    if not os.path.exists(log_filename):
        with open(log_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Updated header with level information
            writer.writerow(['episode', 'total_reward', 'episode duration', 'total duration', 'start_level', 'end_level'])

    # Set transitions.csv path
    transitions_filename = f'episode_logs/{type}_{timestamp}_conf{config_version}/{job_id}_{type}/transitions.h5'

    checkpoints_dir = f"results/checkpoints/{type}_{timestamp}_conf{config_version}"
    os.makedirs(checkpoints_dir, exist_ok=True)

    return frames_dir, checkpoints_dir, log_filename, transitions_filename

# def count_time(start_time, num_frames, env_info, ):


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

def load_config(config_version, type):
    if not config_version or not type:
        raise ValueError("Both 'config_version' and 'implementation_type' must be provided.")

    # Construct the module name based on the pattern
    module_name = f"utils.config_{config_version}_{type}"
    try:
        # Import the configuration module dynamically
        config_module = importlib.import_module(module_name)
    except ImportError as e:
        raise ImportError(f"Could not import config module '{module_name}'.") from e

    return config_module

def find_latest_checkpoint(checkpoint_dir):
    checkpoint_files = [f for f in os.listdir(checkpoint_dir) if f.startswith(f'mario') and f.endswith('.pth')]
    if not checkpoint_files:
        print("None")
        return None
    # Extract episode numbers and sort
    checkpoint_files.sort(key=lambda x: int(x.split('_')[-1].split('.pth')[0]), reverse=True)
    print(checkpoint_files)
    print(checkpoint_files[0])
    return os.path.join(checkpoint_dir, checkpoint_files[0])

def get_latest_episode_number(checkpoint_file):
    base = os.path.basename(checkpoint_file)
    episode_str = base.split('_')[-1].split('.pth')[0]
    return int(episode_str)

def find_episode_log_file(log_dir):
    for file in os.listdir(log_dir):
        if file.endswith('_episode_log.csv'):
            return os.path.join(log_dir, file)
    return None

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