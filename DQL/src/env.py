import gym
import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from gym import Wrapper
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace
from utils.config import *

import gym
from gym import Wrapper

class LevelTracker(Wrapper):
    """
    Gym Wrapper to track the current level in Super Mario Bros.
    """

    def __init__(self, env, skip):
        super(LevelTracker, self).__init__(env)
        # Define a mapping from RAM value to level name
        self.skip = skip
        self.level_map = self.create_level_map()

    def create_level_map(self):
        # This is a placeholder mapping. You'll need to adjust it based on the game's encoding.
        # Super Mario Bros. has 8 worlds, each with 4 levels, plus bonus stages.
        level_map = {}
        world_level = [
            'World 1-1', 'World 1-2', 'World 1-3', 'World 1-4',
            'World 2-1', 'World 2-2', 'World 2-3', 'World 2-4',
            'World 3-1', 'World 3-2', 'World 3-3', 'World 3-4',
            'World 4-1', 'World 4-2', 'World 4-3', 'World 4-4',
            'World 5-1', 'World 5-2', 'World 5-3', 'World 5-4',
            'World 6-1', 'World 6-2', 'World 6-3', 'World 6-4',
            'World 7-1', 'World 7-2', 'World 7-3', 'World 7-4',
            'World 8-1', 'World 8-2', 'World 8-3', 'World 8-4',
            'Bonus Stage'
        ]

        # Populate the mapping
        for i, level in enumerate(world_level, start=1):
            ram_value = i  # Assuming RAM values start at 1 for World 1-1
            level_map[ram_value] = level

        return level_map

    def get_current_level(self):
        # Access the unwrapped environment to get RAM
        ram = self.env.unwrapped.ram

        # RAM address 0x075 (decimal 117) holds the current level information
        current_level_ram = ram[117]

        # Decode the RAM value to get the level name
        level = self.level_map.get(current_level_ram, 'unknown')

        return level

    def reset(self, **kwargs):
        state, info = self.env.reset(**kwargs)
        level = self.get_current_level()
        info['level'] = level
        return state, info

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info

def make_env():
    env = gym_super_mario_bros.make(ENV_NAME, render_mode='human' if DISPLAY else 'rgb', apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    # env = LevelTracker(env, skip=4)
    env = SkipFrame(env, skip=4)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, 4)
    return env

class SkipFrame(Wrapper):
    def __init__(self, env, skip):
        super().__init__(env)
        self.skip = skip

    def step(self, action):
        total_reward = 0.0
        done = False
        for _ in range(self.skip):
            next_state, reward, done, trunc, info = self.env.step(action)
            total_reward += reward
            if done:
                break
        return next_state, total_reward, done, trunc, info


def apply_wrappers(env):
    env = SkipFrame(env, skip=4) # Num of frames to apply one action to
    env = ResizeObservation(env, shape=84) # Resize frame from 240x256 to 84x84
    env = GrayScaleObservation(env)
    env = FrameStack(env, num_stack=4, lz4_compress=True) # May need to change lz4_compress to False if issues arise
    return env