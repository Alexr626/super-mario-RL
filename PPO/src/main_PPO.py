# main_PPO.py

import argparse
import torch
import os
import shutil
from agent_PPO import Agent_PPO
from env import make_env 
import numpy as np
from PIL import Image
from utils.config import *
from utils.utils import *

def train(job_id):
    type = "PPO"
    torch.manual_seed(123)
    timestamp = datetime.now().strftime('%m%d_%H%M%S')

    frames_dir, checkpoints_dir, log_filename, transitions_filename = (
        create_save_files_directories(timestamp, job_id, type))

    # Create environment
    env = make_env()
    state_size = env.observation_space.shape
    action_size = env.action_space.n

    # Initialize agent
    agent = Agent_PPO(state_size, action_size)

    # Initialize variables
    start_time = time.time()
    total_steps = 0
    env.reset()

    for episode in range(NUM_EPISODES):
        state, _ = env.reset()
        state = torch.tensor(np.array(state), dtype=torch.float32).squeeze(-1)

        episode_start_time = time.time()
        frame_counter = 0
        total_reward = 0
        done = False

        while not done:
            frame_counter += 1
            action, log_prob, value = agent.select_action(state)
            next_state, reward, done, truncated, info = env.step(action)
            next_state = torch.tensor(np.array(next_state), dtype=torch.float32).squeeze(-1)
            total_reward += reward

            if frame_counter == 1:
                start_world = str(info['world']) + "_" + str(info['stage'])  # Replace 'stage' with the correct key if different

            current_world = str(info['world']) + "_" + str(info['stage'])

            agent.store_in_memory( state, action, reward, next_state, done, log_prob, value)
            agent.optimize_model()
            state = next_state
            total_reward += reward

            if frame_counter % SAVE_INTERVAL == 0:
                frame = env.render()  # No arguments needed
                if frame is not None:
                    try:
                        image = Image.fromarray(frame)
                        image_filename = os.path.join(frames_dir, f"episode_{episode}_step_{frame_counter}_level_{current_world}.png")
                        image.save(image_filename)
                    except Exception as e:
                        print(f"Error saving frame: {e}")

        end_world = current_world

        # Calculate durations
        episode_duration = time.time() - episode_start_time
        total_duration = time.time() - start_time

        episode_data = [episode, total_reward, episode_duration, total_duration, start_world, end_world]
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(episode_data)

        if (episode + 1) % SAVE_INTERVAL == 0:
            save_model(agent.model, os.path.join(checkpoints_dir, f"/mario_ppo_{episode}.pth"))

        print(f"Episode {episode}: Total Reward = {total_reward}, Episode Duration = {episode_duration:.2f}s, "
              f"Total Duration = {total_duration:.2f}s, Start stage = {start_world}, End stage = {end_world}")

        # Optimize the model

    env.close()

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Super Mario RL Training")
    parser.add_argument('--job_id', type=int, required=True, help='Unique Job ID')
    args = parser.parse_args()
    job_id = args.job_id

    # Test
    job_id = 5
    train(job_id)