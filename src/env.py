import gym
from gym_super_mario_bros.actions import RIGHT_ONLY
from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
from nes_py.wrappers import JoypadSpace

def make_env():
    env = gym.make('SuperMarioBros-v0', apply_api_compatibility=True)
    env = JoypadSpace(env, RIGHT_ONLY)
    env = GrayScaleObservation(env)
    env = ResizeObservation(env, 84)
    env = FrameStack(env, 4)
    return env
