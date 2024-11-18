# config.py

import torch
import numpy as np

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE)

# Track and environment settings
TRACK_LENGTH = 300  # Extended length of the track
TRACK_WIDTH = 50    # Extended width of the track
MAX_TIMESTEPS = 1100  # Maximum timesteps per episode
BORDER_THRESHOLD = 1.0  # Threshold for penalizing proximity to the border
OBSTACLE_THRESHOLD = 1.0  # Threshold for penalizing proximity to obstacles

# Obstacle definitions
OBSTACLES = [
    {"center": np.array([TRACK_LENGTH / 2, TRACK_WIDTH / 2], dtype=np.float32), "size": np.array([50.0, 30.0], dtype=np.float32)},
    {"center": np.array([TRACK_LENGTH - 75, TRACK_WIDTH - 5], dtype=np.float32), "size": np.array([30.0, 20.0], dtype=np.float32)},
    {"center": np.array([TRACK_LENGTH - 75, 5], dtype=np.float32), "size": np.array([30.0, 20.0], dtype=np.float32)}
]

# DQN parameters - adjusted
LEARNING_RATE = 5e-4  # Increased for more impactful updates
BATCH_SIZE = 64
GAMMA = 0.98  # Increased slightly for better discounting of future rewards
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 1500  # Faster decay to help the agent exploit learned actions sooner
TARGET_UPDATE = 5  # More frequent updates to align with policy network
MEMORY_CAPACITY = 30000  # Increased memory capacity to retain more diverse experiences

# Number of episodes to train the agent
NUM_EPISODES = 15000

# Agent parameters
MAX_SPEED = 4.0
ACCELERATION = 0.9  # Reduced slightly to make agent movement more controlled
STEERING = 0.8  # Reduced steering for smoother directional changes

HIDDEN_LAYERS = {256, 256, 128}

# Reward and penalty values - adjusted
REWARD_GOAL = 200000  # Increased to further encourage reaching the goal
PENALTY_OFF_TRACK = -1000  # Reduced slightly to prevent overwhelming negative rewards
PENALTY_OBSTACLE = -90  # Increased for obstacle penalty to discourage hitting obstacles
PENALTY_BORDER = 3000  # Reduced to allow more flexibility without steep penalty
PENALTY_BACKWARDS = -100  # Increased slightly to discourage
PENALTY_TIME_LIMIT = -25  # Reduced time limit penalty to give agent more time to explore
BORDER_THRESHOLD = 10.0  # Increased to allow more flexibility near the borders
# Forward reward encourages agent to move along the track smoothly
DELTA_X_REWARD_FACTOR = 250  # Adjusted to give more weight to forward movement
TIME_PENALTY = -1  # Reduced time penalty for each step to avoid discouraging exploration

# State and action sizes
STATE_SIZE = 12  # Adjusted according to the state representation
ACTION_SIZE = 4  # Number of possible actions: Accelerate, Decelerate, Steer Left, Steer Right

# Drawing parameters
PROCENTAGE_TO_DRAW = 0.9
DRAW_BEST_TRAJECTORIES = int(NUM_EPISODES * PROCENTAGE_TO_DRAW) + 1
