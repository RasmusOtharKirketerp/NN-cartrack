# nn.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import config  # Import configuration file

# Car Track Environment with High Maneuverability and Multiple Obstacles
class CarTrackEnv:
    def __init__(self):
        self.track_length = config.TRACK_LENGTH
        self.track_width = config.TRACK_WIDTH
        self.obstacles = config.OBSTACLES
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

        # Calculate distances to each obstacle
        obstacle_distances = []
        for obstacle in self.obstacles:
            obstacle_x_min = obstacle["center"][0] - obstacle["size"][0] / 2
            obstacle_x_max = obstacle["center"][0] + obstacle["size"][0] / 2
            obstacle_y_min = obstacle["center"][1] - obstacle["size"][1] / 2
            obstacle_y_max = obstacle["center"][1] + obstacle["size"][1] / 2

            distance_x = max(0, obstacle_x_min - self.position[0], self.position[0] - obstacle_x_max)
            distance_y = max(0, obstacle_y_min - self.position[1], self.position[1] - obstacle_y_max)
            obstacle_distances.extend([distance_x, distance_y])

        state = np.concatenate((self.position, self.velocity, [distance_to_left_border, distance_to_right_border], obstacle_distances))
        return torch.tensor(state, dtype=torch.float32).to(config.DEVICE)

    def step(self, action):
        if self.done:
            return self._get_state(), 0, self.done

        self._update_velocity(action)
        self._apply_friction_and_cap_speed()
        self._update_position()

        delta_x = self.position[0] - self.prev_position_x
        self.timestep += 1  # Increment timestep

        # Check if off-track or reached goal
        off_track = self._check_if_off_track()
        reached_goal = self.position[0] >= self.track_length

        # Calculate reward and penalties
        reward = self._calculate_reward(delta_x, reached_goal, off_track)

        # Update previous x-position
        self.prev_position_x = self.position[0]

        return self._get_state(), reward, self.done

    def _update_velocity(self, action):
        if action == 0:  # Accelerate
            self.velocity[0] += config.ACCELERATION
        elif action == 1:  # Decelerate
            self.velocity[0] -= config.ACCELERATION
        elif action == 2:  # Steer left
            self.velocity[1] -= config.STEERING
        elif action == 3:  # Steer right
            self.velocity[1] += config.STEERING

    def _apply_friction_and_cap_speed(self):
        # Apply friction
        self.velocity *= 0.95

        # Cap the speed at max_speed
        speed = np.linalg.norm(self.velocity)
        if speed > config.MAX_SPEED:
            self.velocity = (self.velocity / speed) * config.MAX_SPEED

    def _update_position(self):
        # Update the car's position based on velocity
        self.position += self.velocity
        self.position_history.append(self.position.copy())
        self.speed_history.append(np.linalg.norm(self.velocity))

    def _check_if_off_track(self):
        return (
            self.position[0] < 0 or
            self.position[0] > self.track_length or
            self.position[1] < 0 or
            self.position[1] > self.track_width
        )

    def _calculate_reward(self, delta_x, reached_goal, off_track):
        border_penalty = self._calculate_border_penalty()
        obstacle_penalty = self._calculate_obstacle_penalty()

        if reached_goal:
            reward = config.REWARD_GOAL  # Reward for reaching the goal
            self.done = True
            self.reached_goal = True
        elif off_track:
            reward = config.PENALTY_OFF_TRACK  # Penalty for going off-track
            self.done = True
        elif self.timestep >= config.MAX_TIMESTEPS:  # End the episode if max timesteps reached
            reward = config.PENALTY_TIME_LIMIT  # Penalty for running out of time
            self.done = True
        else:
            # In _calculate_reward method
            reward = delta_x * FORWARD_REWARD_MULTIPLIER - TIME_PENALTY + border_penalty + obstacle_penalty 

        return reward

    def _calculate_border_penalty(self):
        distance_to_left_border = self.position[1]
        distance_to_right_border = self.track_width - self.position[1]
        min_distance_to_border = min(distance_to_left_border, distance_to_right_border)

        if min_distance_to_border < config.BORDER_THRESHOLD:
            return - (config.BORDER_THRESHOLD - min_distance_to_border) * config.PENALTY_BORDER
        return 0

    def _calculate_obstacle_penalty(self):
        obstacle_penalty = 0
        for obstacle in self.obstacles:
            obstacle_x_min = obstacle["center"][0] - obstacle["size"][0] / 2
            obstacle_x_max = obstacle["center"][0] + obstacle["size"][0] / 2
            obstacle_y_min = obstacle["center"][1] - obstacle["size"][1] / 2
            obstacle_y_max = obstacle["center"][1] + obstacle["size"][1] / 2

            in_obstacle = (
                obstacle_x_min <= self.position[0] <= obstacle_x_max and
                obstacle_y_min <= self.position[1] <= obstacle_y_max
            )

            if in_obstacle:
                obstacle_penalty = config.PENALTY_OBSTACLE
                self.done = True
                break
            else:
                distance_x = max(0, obstacle_x_min - self.position[0], self.position[0] - obstacle_x_max)
                distance_y = max(0, obstacle_y_min - self.position[1], self.position[1] - obstacle_y_max)
                min_distance_to_obstacle = min(distance_x, distance_y)

                if min_distance_to_obstacle < config.OBSTACLE_THRESHOLD:
                    obstacle_penalty -= (config.OBSTACLE_THRESHOLD - min_distance_to_obstacle) * 50

        return obstacle_penalty

class DQN(nn.Module):
    def __init__(self, state_size, action_size):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(state_size, config.FC1_UNITS)
        self.fc2 = nn.Linear(config.FC1_UNITS, config.FC2_UNITS)
        self.fc3 = nn.Linear(config.FC2_UNITS, action_size)

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

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

def select_action(state, epsilon, policy_net):
    if random.random() < epsilon:
        return torch.tensor([[random.randrange(config.ACTION_SIZE)]], device=config.DEVICE, dtype=torch.long)
    else:
        with torch.no_grad():
            return policy_net(state).max(0)[1].view(1, 1)

def optimize_model(memory, policy_net, target_net, optimizer):
    if len(memory) < config.BATCH_SIZE:
        return
    transitions = memory.sample(config.BATCH_SIZE)

    state_batch, action_batch, reward_batch, next_state_batch, done_batch = transitions

    state_batch = torch.stack(state_batch).to(config.DEVICE)
    action_batch = torch.cat(action_batch).to(config.DEVICE)
    reward_batch = torch.tensor(reward_batch, device=config.DEVICE, dtype=torch.float32)
    next_state_batch = torch.stack(next_state_batch).to(config.DEVICE)
    done_batch = torch.tensor(done_batch, device=config.DEVICE, dtype=torch.bool)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    with torch.no_grad():
        next_state_values = target_net(next_state_batch).max(1)[0]
    expected_state_action_values = reward_batch + (config.GAMMA * next_state_values * (~done_batch))

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.squeeze(), expected_state_action_values)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_net.parameters(), 1)
    optimizer.step()

def update_plot(trajectories, x_positions_current, y_positions_current, total_reward_current, ax):
    # 'trajectories' is a list of tuples: (x_positions, y_positions, total_reward)

    # Clear previous trajectories and texts
    # Remove all lines
    #for line in ax.lines[:]:
    #    line.remove()
    # Remove all texts
    #for text in ax.texts[:]:
    #    text.remove()

    # Plot the top 10 trajectories
    for x_positions, y_positions, total_reward in trajectories:
        ax.plot(x_positions, y_positions, color='blue', linewidth=0.5)
        # Add score at the end of each trajectory
        ax.text(x_positions[-1], y_positions[-1], f"{total_reward:.1f}", fontsize=8)

    # Plot the current trajectory in a different color
    ax.plot(x_positions_current, y_positions_current, color='red', linewidth=1.0)
    ax.text(x_positions_current[-1], y_positions_current[-1], f"{total_reward_current:.1f}", fontsize=10, color='red')

    plt.draw()
    plt.pause(0.001)



def train(env, policy_net, target_net, optimizer, memory, ax):
    epsilon = config.EPS_START  # Initialize epsilon
    epsilon_decay = (config.EPS_START - config.EPS_END) / config.EPS_DECAY
    trajectories = []

    for episode in range(config.NUM_EPISODES):
        state = env.reset()
        total_reward = 0

        for t in range(config.MAX_TIMESTEPS):
            state = state.to(config.DEVICE)
            action = select_action(state, epsilon, policy_net)
            next_state, reward, done = env.step(action.item())
            total_reward += reward

            reward_tensor = torch.tensor([reward], device=config.DEVICE)
            done_tensor = torch.tensor([done], device=config.DEVICE, dtype=torch.bool)

            memory.push(state, action, reward_tensor, next_state, done_tensor)

            state = next_state

            optimize_model(memory, policy_net, target_net, optimizer)

            if done:
                break

        # Update epsilon
        if epsilon > config.EPS_END:
            epsilon -= epsilon_decay

        # Update the target network
        if episode % config.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        print(f"Episode {episode + 1}: Total Reward = {total_reward}")

        # Get current trajectory positions
        x_positions_current = [pos[0] for pos in env.position_history]
        y_positions_current = [pos[1] for pos in env.position_history]

        # Store the trajectory
        trajectories.append((x_positions_current, y_positions_current, total_reward))

        # Keep only the top 10 trajectories
        trajectories = sorted(trajectories, key=lambda x: x[2], reverse=True)[:10]

        # Update the plot
        update_plot(trajectories, x_positions_current, y_positions_current, total_reward, ax)



def setup_environment():
    env = CarTrackEnv()
    return env

def setup_model():
    policy_net = DQN(config.STATE_SIZE, config.ACTION_SIZE).to(config.DEVICE)
    target_net = DQN(config.STATE_SIZE, config.ACTION_SIZE).to(config.DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
    memory = ReplayMemory(config.MEMORY_CAPACITY)

    return policy_net, target_net, optimizer, memory

def setup_visualization(env):
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

    # Draw obstacles
    for obstacle in env.obstacles:
        obstacle_x_min = obstacle["center"][0] - obstacle["size"][0] / 2
        obstacle_y_min = obstacle["center"][1] - obstacle["size"][1] / 2
        obstacle_rect = patches.Rectangle(
            (obstacle_x_min, obstacle_y_min),
            obstacle["size"][0],
            obstacle["size"][1],
            linewidth=1,
            edgecolor='black',
            facecolor='brown',
            label='Obstacle'
        )
        ax.add_patch(obstacle_rect)

    ax.legend(loc='upper right')

    return fig, ax

def main():
    env = setup_environment()
    policy_net, target_net, optimizer, memory = setup_model()
    fig, ax = setup_visualization(env)

    train(env, policy_net, target_net, optimizer, memory, ax)

    # Ensure the plot is displayed at the end
    plt.ioff()  # Turn off interactive mode
    plt.show()  # Show the plot

    torch.save(policy_net.state_dict(), 'car_track_model.pth')

if __name__ == "__main__":
    main()
