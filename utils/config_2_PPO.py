# config_1_PPO.py

import torch

# General
BATCH_SIZE = 256
NUM_EPOCHS = 2
GAMMA = 0.95
TARGET_UPDATE = 10000
LEARNING_RATE = 1e-4
NUM_EPISODES = 1000000
MODEL_SAVE_INTERVAL = 100
FRAME_SAVE_INTERVAL = 5000
TRANSITION_SAVE_INTERVAL = 100
MEMORY_CAPACITY = 100000

BETA = 0.01
TAU = 0.95
EPS_PPO = 0.3

