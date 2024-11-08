import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
import numpy as np
import torch
import torch.nn as nn

class CausalEnvWrapper():
    def __init__(self, env):
        super().__init__(env)

        self.env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
        env = JoypadSpace(env, SIMPLE_MOVEMENT)
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, 84)
        env = FrameStack(env, 4)
        self.state

        self.data_buffer = []

    def step(self, action):
        next_state, reward, terminated, truncated, _ = self.env.step(action)
        done = terminated or truncated

        next_frames_array = np.array(next_state._frames)
        next_state_tensor = torch.tensor(next_frames_array).unsqueeze(0).float().squeeze(-1)
        reward_tensor = torch.tensor([reward], dtype=torch.float32)
        done_tensor = torch.tensor([done], dtype=torch.bool)


        self.data_buffer.append((self.state, action, next_state, reward, done))
        self.state = next_state
        return next_state, reward, done, info

    def reset(self):
        self.state = self.env.reset()
        return self.state
def make_env():
    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, 4)
    return env

# Gets current state of environment based on
def get_state(reset_result):
    if isinstance(reset_result, tuple):
        # Gym >= 0.26
        state, info = reset_result
    else:
        # Gym <= 0.25
        state = reset_result
        info = {}

    return state