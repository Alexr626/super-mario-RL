# main.py
import torch
from env import make_env
from agent import Agent
from utils.utils import save_model
from utils.config import *
import numpy as np
import os
from datetime import datetime
from PIL import Image
import csv
import time  # Added for timing
import threading
import queue
import cProfile
import pstats

def train():

    # Generate a timestamp for the current training run
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Create environment
    env = make_env()
    save_frames = env.render_mode == "rgb_array"

    # Create directories and get filenames
    log_filename, transitions_filename = create_save_files_directories(timestamp, save_frames)
    agent = Agent(env.action_space.n)

    episode_queue = queue.Queue()
    transition_queue = queue.Queue()

    # Start writer threads
    episode_writer = threading.Thread(target=episode_writer_thread, args=(episode_queue, log_filename))
    transition_writer = threading.Thread(target=transition_writer_thread, args=(transition_queue, transitions_filename))
    episode_writer.start()
    transition_writer.start()

    try:
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
                agent.memory.push(state.cpu(), action.cpu(), reward_tensor, next_state_tensor.cpu(), done_tensor, eps)

                transition_data = (
                    state.cpu().numpy().tolist(),
                    action.cpu().numpy().tolist(),
                    reward,
                    next_state_tensor.cpu().numpy().tolist(),
                    done,
                    eps
                )
                transition_queue.put(transition_data)

                agent.optimize_model()
                state = next_state_tensor
                total_reward += reward

                # Save frame at intervals
                if frame_counter % FRAME_SAVE_INTERVAL == 0 and save_frames:
                    frame = env.render()  # No arguments needed
                    if frame is not None:
                        try:
                            image = Image.fromarray(frame)
                            timestamp_image = datetime.now().strftime('%Y%m%d_%H%M%S')
                            image_filename = f'frames/{timestamp}/episode_{episode}_step_{frame_counter}_level_{current_level}_{timestamp_image}.png'
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
            episode_queue.put(episode_data)

            # Save model checkpoint at intervals
            if episode % SAVE_INTERVAL == 0:
                save_model(agent.policy_net, f"checkpoints/mario_dqn_{episode}.pth")

            # Print episode summary
            print(f"Episode {episode}: Total Reward = {total_reward}, Duration = {episode_duration:.2f}s, Start Level = {start_level}, End Level = {end_level}")

    finally:
        # Signal the writer threads to terminate
        episode_queue.put(None)
        transition_queue.put(None)

        # Wait for writer threads to finish
        episode_writer.join()
        transition_writer.join()

        # Close environment and ReplayMemory
        env.close()
        agent.memory.close()

    # Close environment and ReplayMemory
    env.close()
    agent.memory.close()

def create_save_files_directories(timestamp, save_frames):
    """
    Create necessary directories for saving frames and logs.

    Args:
        timestamp (str): Current timestamp string.
        save_frames (bool): Whether to save frames.

    Returns:
        tuple: Paths to episode_log.csv and transitions.csv
    """
    # Create frames directory if needed
    if save_frames:
        frames_dir = f"frames/{timestamp}"
        os.makedirs(frames_dir, exist_ok=True)

    # Create episode_logs directory
    episode_logs_dir = f"episode_logs/{timestamp}"
    os.makedirs(episode_logs_dir, exist_ok=True)

    # Create episode_log.csv
    log_filename = f'episode_logs/{timestamp}/episode_log.csv'
    if not os.path.exists(log_filename):
        with open(log_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # Updated header with level information
            writer.writerow(['episode', 'total_reward', 'duration', 'start_level', 'end_level'])

    # Set transitions.csv path
    transitions_filename = f'episode_logs/{timestamp}/transitions.csv'
    if not os.path.exists(transitions_filename):
        with open(transitions_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['state', 'action', 'reward', 'next_state', 'done', 'epsilon'])

    return log_filename, transitions_filename

def episode_writer_thread(episode_queue, log_filename):
    """
    Thread function to handle asynchronous writing of episode statistics.

    Args:
        episode_queue (queue.Queue): Queue containing episode data.
        log_filename (str): Path to the episode_log.csv file.
    """
    with open(log_filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile)
        while True:
            episode_data = episode_queue.get()
            if episode_data is None:
                break  # Exit signal
            writer.writerow(episode_data)
            episode_queue.task_done()

def transition_writer_thread(transition_queue, transitions_filename):
    """
    Thread function to handle asynchronous writing of transitions.

    Args:
        transition_queue (queue.Queue): Queue containing transition data.
        transitions_filename (str): Path to the transitions.csv file.
    """
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
    # stats = pstats.Stats(profiler).sort_stats('cumtime')
    # stats.print_stats(10)

    train()