# nn.py
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
from collections import deque
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import config  # Import configuration file
from tqdm import tqdm

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

        # Update velocity and position based on action
        self._update_velocity(action)
        self._apply_friction_and_cap_speed()
        self._update_position()

        # Calculate forward progress in the x-direction
        delta_x = self.position[0] - self.prev_position_x
        delta_x_reward = max(delta_x * config.DELTA_X_REWARD_FACTOR, 0)  # Reward for forward progress, only positive values

        self.timestep += 1  # Increment timestep

        # Check if the agent is off-track or has reached the goal
        off_track = self._check_if_off_track()
        reached_goal = self.position[0] >= self.track_length

        # Calculate penalties based on conditions
        border_penalty = self._calculate_border_penalty()
        obstacle_penalty = self._calculate_obstacle_penalty()
        
        # Set reward based on conditions
        if reached_goal:
            reward = config.REWARD_GOAL  # Reward for reaching the goal
            self.done = True
            self.reached_goal = True
        elif off_track:
            reward = config.PENALTY_OFF_TRACK  # Penalty for going off-track
            self.done = True
        elif self.timestep >= config.MAX_TIMESTEPS:  # End episode if max timesteps reached
            reward = config.PENALTY_TIME_LIMIT  # Penalty for running out of time
            self.done = True
        else:
            # Regular reward calculation including delta_x_reward, time penalty, and other penalties
            reward = delta_x_reward - config.TIME_PENALTY + border_penalty + obstacle_penalty

        # Update previous x-position for the next step calculation
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
        elif self.timestep >= config.MAX_TIMESTEPS:
            reward = config.PENALTY_TIME_LIMIT  # Penalty for running out of time
            self.done = True
        else:
            # Forward progress reward based on delta_x
            delta_x_reward = delta_x * config.DELTA_X_REWARD_FACTOR
            reward = delta_x_reward - config.TIME_PENALTY + border_penalty + obstacle_penalty

        return reward


    def _calculate_border_penalty(self):
        # Calculate left and right boundary penalties
        distance_to_left_border = self.position[1]
        distance_to_right_border = self.track_width - self.position[1]
        min_distance_to_border = min(distance_to_left_border, distance_to_right_border)

        # Penalize for proximity to left/right borders if too close
        border_penalty = 0
        if min_distance_to_border < config.BORDER_THRESHOLD:
            border_penalty -= (config.BORDER_THRESHOLD - min_distance_to_border) * config.PENALTY_BORDER

        # Check for backward movement from start
        if self.position[0] < self.prev_position_x:  # Going backward in x-direction
            border_penalty -= config.PENALTY_BACKWARDS  # Apply backward penalty

        return border_penalty


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
    def __init__(self, input_size, output_size, hidden_layers):
        super(DQN, self).__init__()

        # List to store layers
        layers = []

        # Input layer
        last_size = input_size

        # Hidden layers
        for layer_size in config.HIDDEN_LAYERS:
            layers.append(nn.Linear(last_size, layer_size))
            layers.append(nn.ReLU())  # Activation function
            last_size = layer_size

        # Output layer
        layers.append(nn.Linear(last_size, output_size))

        # Register layers as a sequential model
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)

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

def update_plot(trajectories, x_positions_current, y_positions_current, total_reward_current, ax,
                top_lines, top_texts, current_line, current_text):
    # Update top trajectories
    for i, (x_positions, y_positions, total_reward) in enumerate(trajectories):
        if i < len(top_lines):
            # Update existing line and text
            top_lines[i].set_data(x_positions, y_positions)
            top_texts[i].set_position((x_positions[-1], y_positions[-1]))
            top_texts[i].set_text(f"{total_reward:.1f}")
        else:
            # Create new line and text
            line, = ax.plot(x_positions, y_positions, color='blue', linewidth=0.5)
            text = ax.text(x_positions[-1], y_positions[-1], f"{total_reward:.1f}", fontsize=8)
            top_lines.append(line)
            top_texts.append(text)

    # Hide extra lines and texts if trajectories list is shorter
    for i in range(len(trajectories), len(top_lines)):
        top_lines[i].set_data([], [])
        top_texts[i].set_text('')

    # Update current trajectory
    current_line.set_data(x_positions_current, y_positions_current)
    current_text.set_position((x_positions_current[-1], y_positions_current[-1]))
    current_text.set_text(f"{total_reward_current:.1f}")

    plt.draw()
    #plt.pause(0.001)




def train(env, policy_net, target_net, optimizer, memory):
    epsilon = config.EPS_START
    epsilon_decay = (config.EPS_START - config.EPS_END) / config.EPS_DECAY
    episode_rewards = []
    average_rewards = []
    epsilon_values = []
    loss_values = []
    q_values = []
    trajectories = []  # Initialize to store trajectories for later plotting

    for episode in tqdm(range(config.NUM_EPISODES), desc="Training Progress"):
        state = env.reset()
        total_reward = 0
        total_loss = 0
        steps = 0
        max_q_value = 0  # Track max Q-value per episode
        delta_x_reward = 0

        trajectory_x = []  # X positions for this episode
        trajectory_y = []  # Y positions for this episode

        for t in range(config.MAX_TIMESTEPS):
            # Store position to track trajectory
            trajectory_x.append(env.position[0])
            trajectory_y.append(env.position[1])

            # Choose action
            state = state.to(config.DEVICE)
            action = select_action(state, epsilon, policy_net)
            next_state, reward, done = env.step(action.item())

            # Track reward components
            delta_x_reward += reward
            #print("delta_x_reward: ", delta_x_reward)   
            #delta_x_reward = delta_x_reward * config.DELTA_X_REWARD_FACTOR
            #delta_x_reward = env._calculate_delta_x_reward()
            border_penalty = env._calculate_border_penalty()
            obstacle_penalty = env._calculate_obstacle_penalty()

            # Accumulate rewards and steps
            total_reward += reward
            steps += 1

            # Track max Q-value for chosen action
            with torch.no_grad():
                q_value = policy_net(state).max().item()
                max_q_value = max(max_q_value, q_value)

            # Optimize model and check if loss is not None
            loss = optimize_model(memory, policy_net, target_net, optimizer)
            if loss is not None:
                total_loss += loss.item()
                #print("total_loss: ", total_loss)

            # Log metrics for this episode

            if done:
                break

        # Update epsilon
        if epsilon > config.EPS_END:
            epsilon -= epsilon_decay

        

        # Store epsilon and Q-values for analysis
        epsilon_values.append(epsilon)
        q_values.append(max_q_value)

        # Add current trajectory to trajectories list
        trajectories.append((trajectory_x, trajectory_y, total_reward))
        #print("Episode: ", episode, "Total Reward: ", total_reward)
        log_metrics(episode, total_reward, episode_rewards, epsilon, total_loss, steps, max_q_value, delta_x_reward, border_penalty, obstacle_penalty)


    

    # Return all collected trajectories
    return trajectories






def plot_trajectories(trajectories, env, episode=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    # Setup visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    ax.set_xlim(0, env.track_length + 50)
    ax.set_ylim(-10, env.track_width + 10)
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    if episode is not None:
        ax.set_title(f'Car Trajectories at Episode {episode}')
    else:
        ax.set_title('Car Trajectories After Training')

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

    # Remove duplicate labels
    handles, labels = ax.get_legend_handles_labels()
    unique_labels = dict(zip(labels, handles))
    ax.legend(unique_labels.values(), unique_labels.keys(), loc='upper right')

    # Plot the top 10 trajectories
    for x_positions, y_positions, total_reward in trajectories:
        ax.plot(x_positions, y_positions, color='blue', linewidth=0.5)
        # Add score at the end of each trajectory
        #ax.text(x_positions[-1], y_positions[-1], f"{total_reward:.1f}", fontsize=8)

    # Save the plot to a file with datetime
    
    if episode is not None:
        filename = f"trajectories_episode_{episode}_{timestamp}.png"
    else:
        filename = f"trajectories_final_{timestamp}.png"
    plt.savefig(filename)
    plt.close(fig)




def setup_environment():
    env = CarTrackEnv()
    return env

def setup_model():
    policy_net = DQN(config.STATE_SIZE, config.ACTION_SIZE, config.HIDDEN_LAYERS).to(config.DEVICE)
    target_net = DQN(config.STATE_SIZE, config.ACTION_SIZE, config.HIDDEN_LAYERS).to(config.DEVICE)
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
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
    print("Training the car track environment with DQN...")
    print("date and time:", datetime.now())
    print("Configuration:", config.__file__)
    env = setup_environment()
    policy_net, target_net, optimizer, memory = setup_model()

    # Train and collect trajectories
    trajectories = train(env, policy_net, target_net, optimizer, memory)

    # After training, plot the final trajectories
    plot_trajectories(trajectories, env)

def log_metrics(episode, total_reward, average_rewards, epsilon, total_loss, steps, max_q_value, delta_x_reward, border_penalty, obstacle_penalty):
   
    # Calculate average reward for the last 100 episodes
    avg_reward = np.mean(average_rewards[-100:]) if len(average_rewards) >= 100 else (np.mean(average_rewards) if average_rewards else 0)


    # Calculate average loss per step
    avg_loss = total_loss / steps if steps > 0 else 0

    # Print metrics
    #print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Avg Reward (100) = {avg_reward:.2f}, "
    #      f"Epsilon = {epsilon:.3f}, Avg Loss = {avg_loss:.4f}, Max Q-Value = {max_q_value:.2f}, Steps = {steps}"
    #      f"Reward Breakdown -> Delta_x Reward: {delta_x_reward:.2f}, Border Penalty: {border_penalty:.2f}, Obstacle Penalty: {obstacle_penalty:.2f}")
    #print(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Avg Reward (100) = {avg_reward:.2f} Steps = {steps}")

    # Append to the moving averages and logs
    average_rewards.append(total_reward)
    # Append log to file
    with open(trlog, "a") as log_file:
        log_file.write(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Avg Reward (100) = {avg_reward:.2f}, "
                       f"Epsilon = {epsilon:.3f}, Avg Loss = {avg_loss:.4f}, Max Q-Value = {max_q_value:.2f}, Steps = {steps}, "
                       f"Reward Breakdown -> Delta_x Reward: {delta_x_reward:.2f}, Border Penalty: {border_penalty:.2f}, Obstacle Penalty: {obstacle_penalty:.2f}\n")
    

def clean_up():
    os.chdir(os.path.dirname(__file__))
    delete_files = [file for file in os.listdir() if file.endswith(".png")]
    for file in delete_files:
        try:
            os.remove(file)
        except FileNotFoundError:
            pass

def initLog():
    
    with open(trlog, "w") as log_file:
        log_file.write("Training Log\n" + timestamp + "\n")
        log_file.write("-----------\n")
        log_file.write(f"Configuration: {config.__file__}\n\n")
        log_file.write("Training Metrics\n")
        log_file.write("---------------\n")
        log_file.write("Configuration Items\n")
        log_file.write("--------------------\n")
        
        # Loop through each attribute in config and log its value
        for key, value in vars(config).items():
            if not key.startswith("__"):  # Skip special attributes
                log_file.write(f"{key}: {value}\n")
                
        log_file.write("\nTraining Progress\n")
        log_file.write("-----------------\n")

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
trlog = f"training_log_{timestamp}.txt"
if __name__ == "__main__":
    print("Training the car track environment with DQN...")
    #clean_up()
    initLog()
    main()
    print("Training complete. Check the 'training_log.txt' file for details.")
