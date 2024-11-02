# config.py
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000000
TARGET_UPDATE = 10000
MEMORY_CAPACITY = 100000
LEARNING_RATE = 1e-4
NUM_EPISODES = 10000
SAVE_INTERVAL = 100