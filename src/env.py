import gym
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, RIGHT_ONLY
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace

def make_env():
    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True, render_mode="human")
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, 4)
    return env
