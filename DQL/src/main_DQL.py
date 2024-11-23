# main_DQL.py
from env import make_env
from agent_DQL import Agent_DQL
from utils.utils import *
from utils.config import *
import numpy as np
import os
from datetime import datetime
from PIL import Image
import csv
import time  # Added for timing
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

    type = "DQL"

    # Create environment
    env = make_env()

    # Create directories and get filenames
    frames_dir, checkpoints_dir, log_filename, transitions_filename = (
        create_save_files_directories(timestamp, job_id, type))

    agent = Agent_DQL(env.action_space.n)

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
        reset_result = env.reset()
        state, info = reset_result
        state = np.array(state).squeeze(-1)
        print(state.shape)


        # Convert the initial state to a tensor
        # frames_array = np.array(state._frames)
        # state = torch.from_numpy(frames_array).unsqueeze(0).float().squeeze(-1)

        episode_start_time = time.time()
        frame_counter = 0
        total_reward = 0
        done = False
        start_level = info.get('level', 'unknown')  # Replace 'level' with the correct key if different
        current_level = start_level

        while not done:
            frame_counter += 1

            action, eps = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = np.array(next_state).squeeze(-1)
            done = terminated or truncated

            # Extract current_level from info
            current_level = info.get('level', current_level)  # Update current_level if 'level' is present

            next_state_tensor = next_state
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            done_tensor = torch.tensor([done], dtype=torch.bool)

            # Push transition with epsilon
            agent.memory.push(state, action, reward_tensor, next_state, done_tensor, eps)

            if transitions:
                transition_data = (
                    state.cpu().numpy(),          # Shape: (84, 84, 4)
                    action.cpu().item(),          # Scalar integer
                    reward,                       # Scalar float
                    next_state.cpu().numpy(),  # Shape: (84, 84, 4)
                    done,                  # Scalar boolean
                    eps                           # Scalar float
                )
                transitions_buffer.append(transition_data)

            agent.optimize_model()
            state = next_state_tensor
            total_reward += reward

            # Save frame at intervals
            if frame_counter % SAVE_INTERVAL == 0:
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

if __name__ == "__main__":
    train(transitions=False)