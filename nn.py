# nn.py
import pickle
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
import all_episodes
import config  # Import configuration file
from tqdm import tqdm
from env import CarTrackEnv

log_entries = []  # Store logs in memory to write at the end

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

    return loss  # Return loss for logging

def save_trajectory(episode, trajectory_data):
    # Create a directory to save trajectories if it doesn't exist
    trajectory_dir = 'trajectories'
    os.makedirs(trajectory_dir, exist_ok=True)

    # Define the filename for the trajectory
    filename = os.path.join(trajectory_dir, f'trajectory_episode_{episode + 1}.pkl')

    # Save the trajectory data to the file
    with open(filename, 'wb') as f:
        pickle.dump(trajectory_data, f)


def train(env, policy_net, target_net, optimizer, memory):
    epsilon = config.EPS_START
    epsilon = max(config.EPS_END, epsilon * config.EPS_DECAY)

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

        # Initialize per-episode variables
        delta_x_reward_total = 0
        border_penalty_total = 0
        obstacle_penalty_total = 0

        trajectory_x = []  # X positions for this episode
        trajectory_y = []  # Y positions for this episode

        # Initialize trajectory data for this episode
        trajectory_data = {
            'positions': [],
            'actions': [],
            'rewards': [],
            'dones': [],
        }

        for t in range(config.MAX_TIMESTEPS):
            # Store position to track trajectory
            trajectory_x.append(env.position[0])
            trajectory_y.append(env.position[1])

            # Choose action
            state = state.to(config.DEVICE)
            action = select_action(state, epsilon, policy_net)
            next_state, reward, done, info = env.step(action.item())

            # Accumulate rewards and steps
            total_reward += reward
            steps += 1

            # Accumulate reward components
            delta_x_reward_total += info['delta_x_reward']
            border_penalty_total += info['border_penalty']
            obstacle_penalty_total += info['obstacle_penalty']

            # Track max Q-value for chosen action
            with torch.no_grad():
                q_value = policy_net(state).max().item()
                max_q_value = max(max_q_value, q_value)

            # Store transition in memory
            memory.push(state.cpu(), action.cpu(), torch.tensor([reward], device='cpu'), next_state.cpu(), torch.tensor([done], device='cpu'))

            # Move to the next state
            state = next_state

            # Optimize model and check if loss is not None
            loss = optimize_model(memory, policy_net, target_net, optimizer)
            if loss is not None:
                total_loss += loss.item()

             # Store position and action
            trajectory_data['positions'].append(env.position.copy())
            trajectory_data['actions'].append(action.item())
            trajectory_data['rewards'].append(reward)
            trajectory_data['dones'].append(done)

            if done:
                break
        save_trajectory(episode, trajectory_data)
        # Update epsilon
        if epsilon > config.EPS_END:
            epsilon -= epsilon_decay

        # Update target network periodically
        if episode % config.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # Store epsilon and Q-values for analysis
        epsilon_values.append(epsilon)
        q_values.append(max_q_value)

        # Add current trajectory to trajectories list
        #trajectories.append((trajectory_x, trajectory_y, total_reward))

        # Log metrics for this episode
        log_metrics(episode, total_reward, average_rewards, epsilon, total_loss, steps, max_q_value,
                    delta_x_reward_total, border_penalty_total, obstacle_penalty_total)
        
    # Write all logs to file at the end
    with open(trlog, "w") as log_file:
        log_file.writelines(log_entries)


def setup_environment():
    env = CarTrackEnv()
    return env

def setup_model():
    policy_net = DQN(config.STATE_SIZE, config.ACTION_SIZE, config.HIDDEN_LAYERS).to(config.DEVICE)
    target_net = DQN(config.STATE_SIZE, config.ACTION_SIZE, config.HIDDEN_LAYERS).to(config.DEVICE)
    target_net.load_state_dict(policy_net.state_dict())
    optimizer = torch.optim.Adam(policy_net.parameters(), lr=config.LEARNING_RATE)
    memory = ReplayMemory(config.MEMORY_CAPACITY)
    return policy_net, target_net, optimizer, memory

def main():
    # Delete all files in the trajectories directory
    trajectory_dir = 'trajectories'
    if os.path.exists(trajectory_dir):
        for filename in os.listdir(trajectory_dir):
            file_path = os.path.join(trajectory_dir, filename)
            try:
                if os.path.isfile(file_path):
                    os.unlink(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    
    print("Training the car track environment with DQN...")
    print("Date and time:", datetime.now())
    print("Configuration:", config.__file__)
    env = setup_environment()
    policy_net, target_net, optimizer, memory = setup_model()

    # Train and collect trajectories
    train(env, policy_net, target_net, optimizer, memory)

    # After training, plot the final trajectories
    #plot_trajectories(trajectories, env)
    print("Training complete. Trajectories have been saved to the 'trajectories' directory.")

def log_metrics(episode, total_reward, average_rewards, epsilon, total_loss, steps, max_q_value,
                delta_x_reward, border_penalty, obstacle_penalty):
    # Calculate average reward for the last 100 episodes
    avg_reward = np.mean(average_rewards[-100:]) if len(average_rewards) >= 100 else (np.mean(average_rewards) if average_rewards else 0)

    # Calculate average loss per step
    avg_loss = total_loss / steps if steps > 0 else 0

    # Append to the moving averages and logs
    average_rewards.append(total_reward)

    # Append log to file
    #with open(trlog, "a") as log_file:
    #    log_file.write(f"Episode {episode + 1}: Reward = {total_reward:.2f}, Avg Reward (100) = {avg_reward:.2f}, "
    #                   f"Epsilon = {epsilon:.3f}, Avg Loss = {avg_loss:.4f}, Max Q-Value = {max_q_value:.2f}, Steps = {steps}, "
    #                   f"Reward Breakdown -> Delta_x Reward: {delta_x_reward:.2f}, Border Penalty: {border_penalty:.2f}, Obstacle Penalty: {obstacle_penalty:.2f}\n")
        
    log_entries.append(
            f"Episode {episode + 1}: Reward = {total_reward:.2f}, Avg Reward (100) = {avg_reward:.2f}, "
            f"Epsilon = {epsilon:.3f}, Avg Loss = {avg_loss:.4f}, Max Q-Value = {max_q_value:.2f}, Steps = {steps}, "
            f"Reward Breakdown -> Delta_x Reward: {delta_x_reward:.2f}, Border Penalty: {border_penalty:.2f}, Obstacle Penalty: {obstacle_penalty:.2f}\n"
        )

def initLog():
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
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
    initLog()
    main()
    all_episodes.main()
    

