import torch
import numpy as np

# Device settings
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device: ", DEVICE)

# Track and environment settings
TRACK_LENGTH = 300
TRACK_WIDTH = 50
MAX_TIMESTEPS = 200


# Obstacle definitions
OBSTACLES = [
    {"center": np.array([TRACK_LENGTH / 2, TRACK_WIDTH / 2], dtype=np.float32), "size": np.array([50.0, 30.0], dtype=np.float32)},
    {"center": np.array([TRACK_LENGTH - 75, TRACK_WIDTH - 5], dtype=np.float32), "size": np.array([30.0, 20.0], dtype=np.float32)},
    {"center": np.array([TRACK_LENGTH - 75, 5], dtype=np.float32), "size": np.array([30.0, 20.0], dtype=np.float32)}
]

# DQN parameters
LEARNING_RATE = 1e-5  # Slightly reduced for stability
BATCH_SIZE = 64
GAMMA = 0.99  # Increased for better future reward consideration
EPS_START = 0.5
EPS_END = 0.1
EPS_DECAY = 5000  # Faster decay
TARGET_UPDATE = 10  # Update target network less frequently
MEMORY_CAPACITY = 100000

NUM_EPISODES = 500

# Agent parameters
MAX_SPEED = 5
ACCELERATION = 1
STEERING = 0.9

HIDDEN_LAYERS = [32, 32, 32, 32]  # Use a list to preserve order

# Reward and penalty values
REWARD_GOAL = 100  # Positive reward

# Define other values as factors of the REWARD_GOAL
PENALTY_OFF_TRACK = REWARD_GOAL // 1  # Same as REWARD_GOAL
PENALTY_OBSTACLE = REWARD_GOAL * 2
PENALTY_BORDER = REWARD_GOAL // 1
PENALTY_BACKWARDS = REWARD_GOAL // 100  # Approximate to a close factor
PENALTY_TIME_LIMIT = REWARD_GOAL // 20
DELTA_X_REWARD_FACTOR = REWARD_GOAL / 2  # Adjusted for precision, still a factor
TIME_PENALTY = REWARD_GOAL // 1  # 
BORDER_THRESHOLD = REWARD_GOAL // 1000
OBSTACLE_THRESHOLD = REWARD_GOAL // 1000  
PENALTY_TERMINATION = REWARD_GOAL *20



# State and action sizes
STATE_SIZE = 12
ACTION_SIZE = 4

# Drawing parameters
PROCENTAGE_TO_DRAW = 0.5
DRAW_BEST_TRAJECTORIES = int(NUM_EPISODES * PROCENTAGE_TO_DRAW) + 1
