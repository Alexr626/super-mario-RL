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

def train():

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    env = make_env()
    save_frames = env.render_mode == "rgb_array"
    log_filename = create_save_files_directories(timestamp, save_frames)
    agent = Agent(env.action_space.n, timestamp)

    for episode in range(NUM_EPISODES):
        episode_start_time = datetime.now().time()
        frame_counter = 0
        reset_result = env.reset()

        if isinstance(reset_result, tuple):
            # Gym >= 0.26
            state, info = reset_result
        else:
            # Gym <= 0.25
            state = reset_result
            info = {}

        frames_array = np.array(state._frames)
        state = torch.tensor(frames_array).unsqueeze(0).float().squeeze(-1)
        #print(state)
        total_reward = 0
        done = False

        while not done:
            frame_counter += 1

            action, eps = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            done = terminated or truncated

            next_frames_array = np.array(next_state._frames)
            next_state_tensor = torch.tensor(next_frames_array).unsqueeze(0).float().squeeze(-1)
            # print(next_state_tensor)
            # print(next_state_tensor.shape)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            done_tensor = torch.tensor([done], dtype=torch.bool)

            agent.memory.push(state, action, reward_tensor, next_state_tensor, done_tensor, eps)
            agent.optimize_model()
            state = next_state_tensor
            total_reward += reward

            if agent.steps_done % TARGET_UPDATE == 0:
                agent.update_target_net()

            if frame_counter % FRAME_SAVE_INTERVAL == 0 and save_frames:
                frame = env.render()
                image = Image.fromarray(frame)
                image.save(f'frames/{timestamp}/episode_{episode}_step_{frame_counter}.png')

        episode_duration = datetime.now().time() - episode_start_time
        with open(log_filename, 'a', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([episode, total_reward, episode_duration])

        if episode % SAVE_INTERVAL == 0:
            save_model(agent.policy_net, f"mario_dqn_{episode}.pth")
        print(f"Episode {episode}: Total Reward = {total_reward}, Duration = {episode_duration:.2f}s")

    env.close()
    agent.memory.close()

def create_save_files_directories(timestamp, save_frames):
    if not os.path.exists('frames'):
        os.makedirs('frames')

    if save_frames:
        os.makedirs(f"frames/{timestamp}")

    if not os.path.exists('episode_logs'):
        os.makedirs('episode_logs')

    os.makedirs(f"episode_logs/{timestamp}")

    log_filename = f'episode_logs/{timestamp}/episode_log.csv'

    if not os.path.exists(log_filename):
        with open(log_filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['episode', 'total_reward', 'duration'])

    return log_filename

if __name__ == "__main__":
    train()