import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import matplotlib.patches as patches

# Check if CUDA is available and set the device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Simple Car Track Environment with Obstacle
class CarTrackEnv:
    def __init__(self):
        self.track_length = 100  # Length of the track (x-axis)
        self.track_width = 5     # Width of the track (y-axis)
        # Obstacle parameters
        self.obstacle_center = np.array([self.track_length / 2, self.track_width / 2], dtype=np.float32)
        self.obstacle_size = np.array([10.0, 2.0], dtype=np.float32)  # Width and height
        self.reset()

    def reset(self):
        # Car starts at the beginning of the track, centered
        self.position = np.array([0.0, self.track_width / 2.0], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.done = False
        self.timestep = 0
        self.position_history = [self.position.copy()]  # Initialize position history
        self.speed_history = [np.linalg.norm(self.velocity)]
        self.prev_position_x = self.position[0]  # Initialize previous x-position
        self.reached_goal = False
        return self._get_state()

    def _get_state(self):
        # Calculate distances to borders
        distance_to_left_border = self.position[1]
        distance_to_right_border = self.track_width - self.position[1]
        state = np.concatenate((self.position, self.velocity, [distance_to_left_border, distance_to_right_border]))
        return torch.tensor(state, dtype=torch.float32).to(device)

    def step(self, action):
        """
        Actions:
        0 - Accelerate forward
        1 - Decelerate (brake)
        2 - Steer left
        3 - Steer right
        4 - Do nothing
        """
        if self.done:
            return self._get_state(), 0, self.done

        # Define constants
        acceleration = 0.1
        steering = 0.1
        max_speed = 1.0

        # Update velocity based on action
        if action == 0:  # Accelerate
            self.velocity[0] += acceleration
        elif action == 1:  # Decelerate
            self.velocity[0] -= acceleration
        elif action == 2:  # Steer left
            self.velocity[1] -= steering
        elif action == 3:  # Steer right
            self.velocity[1] += steering
        # else: Do nothing

        # Apply friction
        self.velocity *= 0.95

        # Limit speed
        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = (self.velocity / speed) * max_speed

        # Update position
        self.position += self.velocity
        self.position_history.append(self.position.copy())  # Record the position
        self.speed_history.append(speed)

        # Calculate forward progress
        delta_x = self.position[0] - self.prev_position_x

        # Increment timestep
        self.timestep += 1

        # Check for off-track
        off_track = (
            self.position[0] < 0 or
            self.position[0] > self.track_length or
            self.position[1] < 0 or
            self.position[1] > self.track_width
        )

        # Check for finish line
        reached_goal = self.position[0] >= self.track_length

        # Check for collision with obstacle
        obstacle_x_min = self.obstacle_center[0] - self.obstacle_size[0] / 2
        obstacle_x_max = self.obstacle_center[0] + self.obstacle_size[0] / 2
        obstacle_y_min = self.obstacle_center[1] - self.obstacle_size[1] / 2
        obstacle_y_max = self.obstacle_center[1] + self.obstacle_size[1] / 2

        in_obstacle = (
            obstacle_x_min <= self.position[0] <= obstacle_x_max and
            obstacle_y_min <= self.position[1] <= obstacle_y_max
        )

        # Calculate distances to borders
        distance_to_left_border = self.position[1]
        distance_to_right_border = self.track_width - self.position[1]
        min_distance_to_border = min(distance_to_left_border, distance_to_right_border)

        # Penalty for being close to borders
        border_threshold = 1.0  # Distance threshold to start penalizing
        if min_distance_to_border < border_threshold:
            border_penalty = - (border_threshold - min_distance_to_border) * 50
        else:
            border_penalty = 0

        # Determine reward
        if reached_goal:
            reward = 1000  # Reward for reaching the goal
            self.done = True
            self.reached_goal = True
        elif off_track or in_obstacle:
            reward = -1000  # Penalty for going off-track or hitting obstacle
            self.done = True
        else:
            reward = delta_x * 10 - 1 + border_penalty  # Include border penalty

        # Update previous x-position
        self.prev_position_x = self.position[0]

        return self._get_state(), reward, self.done

# Neural Network for DQN
class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 256)
        self.fc3 = nn.Linear(256, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# Replay Memory for Experience Replay
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque(maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(tuple(arg.to('cpu') if isinstance(arg, torch.Tensor) else arg for arg in args))

    def sample(self, batch_size):
        batch = random.sample(self.memory, batch_size)
        return zip(*batch)

    def __len__(self):
        return len(self.memory)

# Hyperparameters
BATCH_SIZE = 64
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.1
EPS_DECAY = 2000
TARGET_UPDATE = 10
MEMORY_CAPACITY = 20000
NUM_EPISODES = 500
LEARNING_RATE = 1e-4

action_size = 5  # Number of actions
state_size = 6   # Position (x, y), velocity (v_x, v_y), distances to borders

# Initialize environment, models, optimizer, and memory
env = CarTrackEnv()

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_CAPACITY)

steps_done = 0
epsilon = EPS_START

# Enable interactive mode
plt.ion()
fig, ax = plt.subplots(figsize=(10, 6))
ax.set_xlim(0, env.track_length + 5)  # Extend x-axis slightly for visibility
ax.set_ylim(-1, env.track_width + 1)  # Extend y-axis limits
ax.set_xlabel('Position X')
ax.set_ylabel('Position Y')
ax.set_title('Car Trajectory Across Episodes')

# Draw track boundaries
ax.axhline(y=0, color='red', linestyle='--', label='Track Boundary')
ax.axhline(y=env.track_width, color='red', linestyle='--')
ax.axvline(x=env.track_length, color='green', linestyle='--', label='Finish Line')

# Shade the track area
ax.fill_between([0, env.track_length], 0, env.track_width, color='lightgrey', alpha=0.5, label='Track Area')

# Draw the obstacle
obstacle_x_min = env.obstacle_center[0] - env.obstacle_size[0] / 2
obstacle_y_min = env.obstacle_center[1] - env.obstacle_size[1] / 2
obstacle_rect = patches.Rectangle(
    (obstacle_x_min, obstacle_y_min),
    env.obstacle_size[0],
    env.obstacle_size[1],
    linewidth=1,
    edgecolor='black',
    facecolor='brown',
    label='Obstacle'
)
ax.add_patch(obstacle_rect)

# Move legend to top right corner
ax.legend(loc='upper right')

# Initialize lists to store data
all_positions = []
all_speeds = []
all_rewards = []
all_episodes = []
line_collections = []
best_reward = float('-inf')
best_episode = None
best_positions = None
best_line = None
successful_episodes_positions = []
successful_lines = []

# Epsilon-greedy action selection with adjusted decay
def select_action(state):
    global steps_done, epsilon
    # Decay epsilon
    epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() < epsilon:
        # Random action (exploration)
        return torch.tensor([[random.randrange(action_size)]], device=device, dtype=torch.long)
    else:
        # Best action (exploitation)
        with torch.no_grad():
            return policy_net(state).max(0)[1].view(1, 1)

# Optimize the model with Double DQN
def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    # Unpack batch
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = transitions

    # Convert batches to tensors
    state_batch = torch.stack(state_batch).to(device)
    action_batch = torch.cat(action_batch).to(device)
    reward_batch = torch.tensor(reward_batch, device=device, dtype=torch.float32)
    next_state_batch = torch.stack(next_state_batch).to(device)
    done_batch = torch.tensor(done_batch, device=device, dtype=torch.bool)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the columns of actions taken
    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute the next state values using Double DQN
    with torch.no_grad():
        # Get actions from policy_net
        next_state_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1)
        # Get Q values from target_net using actions from policy_net
        next_state_values = target_net(next_state_batch).gather(1, next_state_actions).squeeze()

    # Compute the expected Q values
    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (~done_batch))

    # Compute loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.squeeze(), expected_state_action_values)

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # Clip gradients to prevent explosion
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    optimizer.step()

# Training loop
for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0

    for t in range(1000):  # Maximum number of steps per episode
        state = state.to(device)
        action = select_action(state)
        next_state, reward, done = env.step(action.item())
        total_reward += reward

        reward_tensor = torch.tensor([reward], device=device)
        done_tensor = torch.tensor([done], device=device, dtype=torch.bool)

        # Store the transition in memory
        memory.push(state, action, reward_tensor, next_state, done_tensor)

        # Move to the next state
        state = next_state

        # Perform one step of the optimization
        optimize_model()

        if done:
            break

    # Get positions and speeds of the current episode
    positions = np.array(env.position_history)
    speeds = np.array(env.speed_history)

    # Store positions, speeds, and rewards
    all_positions.append(positions)
    all_speeds.append(speeds)
    all_rewards.append(total_reward)
    all_episodes.append(episode)

    if env.reached_goal:
        if total_reward > best_reward:
            if best_line:
                best_line.set_color('blue')
                successful_lines.append(best_line)
            best_reward = total_reward
            best_episode = episode
            best_positions = positions
            best_line, = ax.plot(best_positions[:,0], best_positions[:,1], color='red', linewidth=3, label='Best Episode')
        else:
            line, = ax.plot(positions[:,0], positions[:,1], color='blue', linewidth=2)
            successful_lines.append(line)
            successful_episodes_positions.append(positions)

    print(f"Episode {episode + 1}: Total Reward = {total_reward}")

    # Remove old LineCollections
    if len(line_collections) >= 10:
        old_lc = line_collections.pop(0)
        old_lc.remove()

    # Plot the episode with speed color coding (optional)
    points = positions.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(speeds[:-1])
    lc.set_linewidth(2)
    line_collections.append(ax.add_collection(lc))

    # Refresh the plot
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Update the target network
    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Training complete.")

# Turn off interactive mode
plt.ioff()
plt.show()

# Save the model
torch.save(policy_net.state_dict(), 'car_track_model.pth')
