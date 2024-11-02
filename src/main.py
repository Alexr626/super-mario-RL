# main.py
import torch
from env import make_env
from agent import Agent
from utils.utils import save_model
from utils.config import *
import numpy as np

def train():
    env = make_env()
    agent = Agent(env.action_space.n)
    for episode in range(NUM_EPISODES):
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
        total_reward = 0
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action.item())
            next_frames_array = np.array(next_state._frames)
            next_state_tensor = torch.tensor(next_frames_array).unsqueeze(0).float().squeeze(-1)
            reward_tensor = torch.tensor([reward], dtype=torch.float32)
            done_tensor = torch.tensor([bool(done)], dtype=torch.uint8)
            agent.memory.push(state, action, reward_tensor, next_state_tensor, done_tensor)
            agent.optimize_model()
            state = next_state_tensor
            total_reward += reward

            if agent.steps_done % TARGET_UPDATE == 0:
                agent.update_target_net()

        if episode % SAVE_INTERVAL == 0:
            save_model(agent.policy_net, f"mario_dqn_{episode}.pth")
        print(f"Episode {episode}: Total Reward = {total_reward}")

    env.close()

if __name__ == "__main__":
    train()
