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
LEARNING_RATE = 1e-4  # Slightly reduced for stability
BATCH_SIZE = 64
GAMMA = 0.99  # Increased for better future reward consideration
EPS_START = 1
EPS_END = 0.1
EPS_DECAY = 50000
TARGET_UPDATE = 10  # Update target network less frequently
MEMORY_CAPACITY = 100000

NUM_EPISODES = 300

# Agent parameters
MAX_SPEED = 2
ACCELERATION = 0.1
STEERING = 0.9
STEERING_ANGLE_INCREMENT = np.radians(5)  # Steering change per action (5 degrees)

# Episode-Based Rewards and Penalties
EPISODE_REWARD_GOAL_REACHED = 5000
EPISODE_PENALTY_OFF_TRACK = EPISODE_REWARD_GOAL_REACHED * 2
EPISODE_PENALTY_TIME_LIMIT = 100
EPISODE_PENALTY_FAILURE = 3000

# Step-Based Rewards and Penalties
STEP_REWARD_FORWARD_PROGRESS = 1500
STEP_PENALTY_BORDER = 100
STEP_PENALTY_BACKWARD = 500
STEP_PENALTY_OBSTACLE = EPISODE_REWARD_GOAL_REACHED
STEP_PENALTY_TIME = 1

# Step-Based Thresholds
STEP_THRESHOLD_BORDER = 1.0
STEP_THRESHOLD_OBSTACLE = 1.5

EPISODE_THRESHOLD_MAX_TIMESTEPS = 1000

# State and action sizes
STATE_SIZE = 12
ACTION_SIZE = 4


HIDDEN_LAYERS = [32,64,ACTION_SIZE]  # Use a list to preserve order
# Drawing parameters
PROCENTAGE_TO_DRAW = 0.5
DRAW_BEST_TRAJECTORIES = int(NUM_EPISODES * PROCENTAGE_TO_DRAW) + 1

    
