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

# Simple Car Track Environment with High Maneuverability
class CarTrackEnv:
    def __init__(self):
        self.scale_factor = 1
        self.track_length = 500  # Extended length of the track
        self.track_width = 50     # Extended width of the track
        self.obstacle_center = np.array([self.track_length / 2, self.track_width / 2], dtype=np.float32)
        self.obstacle_size = np.array([50.0, 10.0], dtype=np.float32)  # Wider obstacle
        self.reset()

    def reset(self):
        self.position = np.array([0.0, self.track_width / 2.0], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.done = False
        self.timestep = 0
        self.position_history = [self.position.copy()]
        self.speed_history = [np.linalg.norm(self.velocity)]
        self.prev_position_x = self.position[0]
        self.reached_goal = False
        return self._get_state()

    def _get_state(self):
        distance_to_left_border = self.position[1]
        distance_to_right_border = self.track_width - self.position[1]
        state = np.concatenate((self.position, self.velocity, [distance_to_left_border, distance_to_right_border]))
        return torch.tensor(state, dtype=torch.float32).to(device)

    def step(self, action):
        if self.done:
            return self._get_state(), 0, self.done

        acceleration = 0.1
        steering = 0.1
        max_speed = 1.0

        if action == 0:  # Accelerate
            self.velocity[0] += acceleration
        elif action == 1:  # Decelerate
            self.velocity[0] -= acceleration
        elif action == 2:  # Steer left
            self.velocity[1] -= steering
        elif action == 3:  # Steer right
            self.velocity[1] += steering

        self.velocity *= 0.95

        speed = np.linalg.norm(self.velocity)
        if speed > max_speed:
            self.velocity = (self.velocity / speed) * max_speed

        self.position += self.velocity
        self.position_history.append(self.position.copy())
        self.speed_history.append(speed)

        delta_x = self.position[0] - self.prev_position_x
        self.timestep += 1

        off_track = (
            self.position[0] < 0 or
            self.position[0] > self.track_length or
            self.position[1] < 0 or
            self.position[1] > self.track_width
        )

        reached_goal = self.position[0] >= self.track_length

        obstacle_x_min = self.obstacle_center[0] - self.obstacle_size[0] / 2
        obstacle_x_max = self.obstacle_center[0] + self.obstacle_size[0] / 2
        obstacle_y_min = self.obstacle_center[1] - self.obstacle_size[1] / 2
        obstacle_y_max = self.obstacle_center[1] + self.obstacle_size[1] / 2

        in_obstacle = (
            obstacle_x_min <= self.position[0] <= obstacle_x_max and
            obstacle_y_min <= self.position[1] <= obstacle_y_max
        )

        distance_to_left_border = self.position[1]
        distance_to_right_border = self.track_width - self.position[1]
        min_distance_to_border = min(distance_to_left_border, distance_to_right_border)

        border_threshold = 1.0
        if min_distance_to_border < border_threshold:
            border_penalty = - (border_threshold - min_distance_to_border) * 50
        else:
            border_penalty = 0

        if reached_goal:
            reward = 1000
            self.done = True
            self.reached_goal = True
        elif off_track or in_obstacle:
            reward = -1000
            self.done = True
        else:
            reward = delta_x * 10 - 1 + border_penalty

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

action_size = 5
state_size = 6

env = CarTrackEnv()

policy_net = DQN(state_size, action_size).to(device)
target_net = DQN(state_size, action_size).to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=LEARNING_RATE)
memory = ReplayMemory(MEMORY_CAPACITY)

steps_done = 0
epsilon = EPS_START

plt.ion()
fig, ax = plt.subplots(figsize=(15, 8))
ax.set_xlim(0, env.track_length + 50)
ax.set_ylim(-10, env.track_width + 10)
ax.set_xlabel('Position X')
ax.set_ylabel('Position Y')
ax.set_title('Car Trajectory Across Episodes')

ax.axhline(y=0, color='red', linestyle='--', label='Track Boundary')
ax.axhline(y=env.track_width, color='red', linestyle='--')
ax.axvline(x=env.track_length, color='green', linestyle='--', label='Finish Line')

ax.fill_between([0, env.track_length], 0, env.track_width, color='lightgrey', alpha=0.5, label='Track Area')

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

ax.legend(loc='upper right')

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

def select_action(state):
    global steps_done, epsilon
    epsilon = EPS_END + (EPS_START - EPS_END) * np.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if random.random() < epsilon:
        return torch.tensor([[random.randrange(action_size)]], device=device, dtype=torch.long)
    else:
        with torch.no_grad():
            return policy_net(state).max(0)[1].view(1, 1)

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = transitions

    state_batch = torch.stack(state_batch).to(device)
    action_batch = torch.cat(action_batch).to(device)
    reward_batch = torch.tensor(reward_batch, device=device, dtype=torch.float32)
    next_state_batch = torch.stack(next_state_batch).to(device)
    done_batch = torch.tensor(done_batch, device=device, dtype=torch.bool)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    with torch.no_grad():
        next_state_actions = policy_net(next_state_batch).max(1)[1].unsqueeze(1)
        next_state_values = target_net(next_state_batch).gather(1, next_state_actions).squeeze()

    expected_state_action_values = reward_batch + (GAMMA * next_state_values * (~done_batch))

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.squeeze(), expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    optimizer.step()

for episode in range(NUM_EPISODES):
    state = env.reset()
    total_reward = 0

    for t in range(1000):
        state = state.to(device)
        action = select_action(state)
        next_state, reward, done = env.step(action.item())
        total_reward += reward

        reward_tensor = torch.tensor([reward], device=device)
        done_tensor = torch.tensor([done], device=device, dtype=torch.bool)

        memory.push(state, action, reward_tensor, next_state, done_tensor)

        state = next_state

        optimize_model()

        if done:
            break

    positions = np.array(env.position_history)
    speeds = np.array(env.speed_history)

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

    if len(line_collections) >= 10:
        old_lc = line_collections.pop(0)
        old_lc.remove()

    points = positions.reshape(-1,1,2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    norm = plt.Normalize(0, 1)
    lc = LineCollection(segments, cmap='viridis', norm=norm)
    lc.set_array(speeds[:-1])
    lc.set_linewidth(2)
    line_collections.append(ax.add_collection(lc))

    fig.canvas.draw()
    fig.canvas.flush_events()

    if episode % TARGET_UPDATE == 0:
        target_net.load_state_dict(policy_net.state_dict())

print("Training complete.")

plt.ioff()
plt.show()

torch.save(policy_net.state_dict(), 'car_track_model.pth')
