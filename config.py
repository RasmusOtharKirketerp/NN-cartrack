# config.py

import torch
import numpy as np

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE)

# Track and environment settings
TRACK_LENGTH = 500  # Extended length of the track
TRACK_WIDTH = 50    # Extended width of the track
MAX_TIMESTEPS = 2000  # Maximum timesteps per episode
BORDER_THRESHOLD = 1.0  # Threshold for penalizing proximity to the border
OBSTACLE_THRESHOLD = 1.0  # Threshold for penalizing proximity to obstacles

# Obstacle definitions: Each obstacle has a position (center) and size
OBSTACLES = [
    {"center": np.array([TRACK_LENGTH / 2, TRACK_WIDTH / 2], dtype=np.float32), "size": np.array([50.0, 30.0], dtype=np.float32)},  # Middle obstacle
    {"center": np.array([TRACK_LENGTH - 75, TRACK_WIDTH - 5], dtype=np.float32), "size": np.array([30.0, 20.0], dtype=np.float32)},  # Top obstacle
    {"center": np.array([TRACK_LENGTH - 75, 5], dtype=np.float32), "size": np.array([30.0, 20.0], dtype=np.float32)}  # Bottom obstacle
]

# DQN parameters
LEARNING_RATE = 1e-4
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 2000
TARGET_UPDATE = 10
MEMORY_CAPACITY = 20000
NUM_EPISODES = 50000
MAX_SPEED = 1.0
ACCELERATION = 0.07
STEERING = 0.1

# Neural network layers
FC1_UNITS = 128
FC2_UNITS = 256

# Reward and penalty values
REWARD_GOAL = 10000
PENALTY_OFF_TRACK = -10000
PENALTY_OBSTACLE = -1000
PENALTY_BORDER = -500
PENALTY_TIME_LIMIT = -50
FORWARD_REWARD_MULTIPLIER = 50
TIME_PENALTY = -2

# State and action sizes
STATE_SIZE = 12  # Adjusted according to the state representation
ACTION_SIZE = 4  # Number of possible actions: Accelerate, Decelerate, Steer Left, Steer Right

