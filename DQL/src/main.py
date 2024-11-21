# main.py
from env import make_env
from agent import Agent
from DQL.utils.utils import save_model
from DQL.utils.config import *
import numpy as np
import os
from datetime import datetime
from PIL import Image
import csv
import time  # Added for timing
import queue
import torch.profiler
import argparse
import h5py

def train(transitions):

    parser = argparse.ArgumentParser(description="Super Mario RL Training")
    parser.add_argument('--job_id', type=int, required=True, help='Unique Job ID')
    args = parser.parse_args()
    job_id = args.job_id

    # Test
    # job_id = 5

    # Generate a timestamp for the current training run
    timestamp = datetime.now().strftime('%m%d_%H%M%S')

    # Create environment
    env = make_env()
    save_frames = env.render_mode == "rgb_array"

    # Create directories and get filenames
    if save_frames:
        frames_dir, checkpoints_dir, log_filename, transitions_filename = (
            create_save_files_directories(timestamp, job_id, save_frames))
    else:
        checkpoints_dir, log_filename, transitions_filename = (
            create_save_files_directories(timestamp, job_id, save_frames))
    agent = Agent(env.action_space.n)

    if transitions:
        # Initialize HDF5 file for transitions with compression
        with h5py.File(transitions_filename, 'a') as hdf5_file:  # Append mode
            if 'transitions' not in hdf5_file:
                dt = np.dtype([
                    ('state', 'f4', (4, 84, 84)),        # Adjust shape as needed
                    ('action', 'i4'),                    # Scalar integer
                    ('reward', 'f4'),                    # Scalar float
                    ('next_state', 'f4', (4, 84, 84)),   # Adjust shape as needed
                    ('done', 'bool'),                    # Scalar boolean
                    ('epsilon', 'f4')                    # Scalar float
                ])
                hdf5_file.create_dataset(
                    'transitions',
                    shape=(0,),
                    maxshape=(None,),
                    dtype=dt,
                    compression="gzip",
                    compression_opts=9
                )

        transitions_buffer = []

    for episode in range(NUM_EPISODES):
        episode_start_time = time.time()
        frame_counter = 0
        reset_result = env.reset()

        state, info = reset_result

        # Extract start_level from info
        start_level = info.get('level', 'unknown')  # Replace 'level' with the correct key if different

        # Convert the initial state to a tensor and move to DEVICE
        frames_array = np.array(state._frames)
        state = torch.from_numpy(frames_array).unsqueeze(0).float().squeeze(-1).to(DEVICE)

        total_reward = 0
        done = False
        current_level = start_level
        while not done:
            frame_counter += 1

            action, eps = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated

            # Extract current_level from info
            current_level = info.get('level', current_level)  # Update current_level if 'level' is present

            # Convert the next state to a tensor and move to DEVICE
            frames_array = np.array(next_state._frames)
            next_state_tensor = torch.from_numpy(frames_array).unsqueeze(0).float().squeeze(-1).to(DEVICE)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            done_tensor = torch.tensor([done], dtype=torch.bool)

            # Push transition with epsilon
            agent.memory.push(state.cpu(), action.cpu(), reward_tensor.cpu(), next_state_tensor.cpu(), done_tensor.cpu(), eps)

            if transitions:
                transition_data = (
                    state.cpu().numpy(),          # Shape: (84, 84, 4)
                    action.cpu().item(),          # Scalar integer
                    reward,                       # Scalar float
                    next_state_tensor.cpu().numpy(),  # Shape: (84, 84, 4)
                    done,                  # Scalar boolean
                    eps                           # Scalar float
                )
                transitions_buffer.append(transition_data)

            agent.optimize_model()
            state = next_state_tensor
            total_reward += reward

            # Save frame at intervals
            if frame_counter % FRAME_SAVE_INTERVAL == 0 and save_frames:
                frame = env.render()  # No arguments needed
                if frame is not None:
                    try:
                        image = Image.fromarray(frame)
                        image_filename = os.path.join(frames_dir, f"episode_{episode}_step_{frame_counter}_level_{current_level}.png")
                        image.save(image_filename)
                    except Exception as e:
                        print(f"Error saving frame: {e}")

            # Update target network at intervals
            if agent.steps_done % TARGET_UPDATE == 0:
                agent.update_target_net()

        # At the end of the episode, set end_level
        end_level = current_level

        # Calculate episode duration
        episode_duration = time.time() - episode_start_time

        # Log episode statistics
        episode_data = [episode, total_reward, episode_duration, start_level, end_level]
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(episode_data)

        if transitions:
            with h5py.File(transitions_filename, 'a') as hdf5_file:
                dataset = hdf5_file['transitions']
                # Resize dataset to accommodate new transitions
                dataset.resize((dataset.shape[0] + len(transitions_buffer)), axis=0)
                # Convert transitions_buffer to structured NumPy array
                structured_data = np.array(transitions_buffer, dtype=dt)
                # Append structured data to dataset
                dataset[-len(transitions_buffer):] = structured_data
            transitions_buffer.clear()

        # Save model checkpoint at intervals
        if (episode + 1) % SAVE_INTERVAL == 0:
            save_model(agent.policy_net, f"{checkpoints_dir}/mario_dqn_{episode}.pth")

        # Print episode summary
        print(f"Episode {episode}: Total Reward = {total_reward}, Duration = {episode_duration:.2f}s, Start Level = {start_level}, End Level = {end_level}")

        # hdf5_file.close()
        env.close()

def create_save_files_directories(timestamp, job_id, save_frames):
    # Create frames directory if needed

    os.makedirs("frames", exist_ok=True)
    if save_frames:
        frames_dir = f"frames/{job_id}_{timestamp}"
        os.makedirs(frames_dir, exist_ok=True)

    # Create episode_logs directory
    episode_logs_dir = f"episode_logs"
    os.makedirs(episode_logs_dir, exist_ok=True)

    # Create episode_log.csv
    log_filename = f'episode_logs/{job_id}_{timestamp}_episode_log.csv'
    if not os.path.exists(log_filename):
        with open(log_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Updated header with level information
            writer.writerow(['episode', 'total_reward', 'duration', 'start_level', 'end_level'])

    # Set transitions.csv path
    transitions_filename = f'episode_logs/{job_id}_{timestamp}/transitions.h5'

    os.makedirs("checkpoints", exist_ok=True)
    checkpoints_dir = f"checkpoints/{job_id}_{timestamp}"
    os.makedirs(checkpoints_dir, exist_ok=True)

    if save_frames:
        return frames_dir, checkpoints_dir, log_filename, transitions_filename
    else:
        return checkpoints_dir, log_filename, transitions_filename

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

if __name__ == "__main__":
    # profiler = cProfile.Profile()
    # profiler.enable()
    # train()
    # profiler.disable()
    # s = io.StringIO()
    # sortby = 'cumulative'
    # ps = pstats.Stats(profiler, stream=s).sort_stats(sortby)
    # ps.print_stats(10)  # Print top 10 functions
    # print(s.getvalue())
    train(transitions=False)