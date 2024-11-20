import torch
import numpy as np

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE)

# Track and environment settings
TRACK_LENGTH = 300
TRACK_WIDTH = 50
MAX_TIMESTEPS = 1100
BORDER_THRESHOLD = 10.0  # Adjusted for consistency
OBSTACLE_THRESHOLD = 1.0

# Obstacle definitions
OBSTACLES = [
    {"center": np.array([TRACK_LENGTH / 2, TRACK_WIDTH / 2], dtype=np.float32), "size": np.array([50.0, 30.0], dtype=np.float32)},
    {"center": np.array([TRACK_LENGTH - 75, TRACK_WIDTH - 5], dtype=np.float32), "size": np.array([30.0, 20.0], dtype=np.float32)},
    {"center": np.array([TRACK_LENGTH - 75, 5], dtype=np.float32), "size": np.array([30.0, 20.0], dtype=np.float32)}
]

# DQN parameters
LEARNING_RATE = 1e-4  # Slightly reduced for stability
BATCH_SIZE = 64
GAMMA = 0.99  # Increased for better future reward consideration
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1000  # Faster decay
TARGET_UPDATE = 10  # Update target network less frequently
MEMORY_CAPACITY = 30000

NUM_EPISODES = 15000

# Agent parameters
MAX_SPEED = 4.0
ACCELERATION = 0.9
STEERING = 0.8

HIDDEN_LAYERS = [256, 256, 128]  # Use a list to preserve order

# Reward and penalty values
REWARD_GOAL = 100       # Positive reward
PENALTY_OFF_TRACK = 100 # Positive penalty
PENALTY_OBSTACLE = 100  # Positive penalty
PENALTY_BORDER = 10     # Positive penalty
PENALTY_BACKWARDS = 50  # Positive penalty
PENALTY_TIME_LIMIT = 10 # Positive penalty
DELTA_X_REWARD_FACTOR = 200.0  # Positive reward factor
TIME_PENALTY = 0.1      # Positive penalty per timestep


# State and action sizes
STATE_SIZE = 12
ACTION_SIZE = 4

# Drawing parameters
PROCENTAGE_TO_DRAW = 0.9
DRAW_BEST_TRAJECTORIES = int(NUM_EPISODES * PROCENTAGE_TO_DRAW) + 1
