import os
import pickle
import torch
from env import CarTrackEnv
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from datetime import datetime
import numpy as np

# Load trajectories from files
def load_trajectories(trajectory_dir='trajectories'):
    """
    Load all trajectory files from the specified directory.
    Returns a list of trajectory data dictionaries.
    """
    # Get a list of trajectory files and sort numerically
    trajectory_files = sorted([
        os.path.join(trajectory_dir, f) for f in os.listdir(trajectory_dir)
        if f.startswith('trajectory_episode_') and f.endswith('.pkl')
    ], key=lambda x: int(x.split('_')[-1].split('.')[0]))  # Extract the episode number for sorting

    trajectories = []
    for file in trajectory_files:
        with open(file, 'rb') as f:
            trajectory_data = pickle.load(f)
            trajectories.append(trajectory_data)
    print(f"Trajectory files loaded: {trajectory_files}")
    print(f"Loaded {len(trajectory_files)} trajectory files.")


    return trajectories


# Visualize a single trajectory
def save_trajectory_to_tensorboard(env, trajectory_data, episode_number, writer):
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.set_xlim(0, env.track_length + 50)
    ax.set_ylim(-10, env.track_width + 10)
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_title(f'Trajectory for Episode {episode_number}')

    # Add track boundaries
    ax.axhline(y=0, color='red', linestyle='--', label='Track Boundary')
    ax.axhline(y=env.track_width, color='red', linestyle='--')
    ax.axvline(x=env.track_length, color='green', linestyle='--', label='Finish Line')
    ax.fill_between([0, env.track_length], 0, env.track_width, color='lightgrey', alpha=0.5, label='Track Area')

    # Plot obstacles
    for obstacle in env.obstacles:
        obstacle_x_min = obstacle["center"][0] - obstacle["size"][0] / 2
        obstacle_y_min = obstacle["center"][1] - obstacle["size"][1] / 2
        obstacle_rect = patches.Rectangle(
            (obstacle_x_min, obstacle_y_min),
            obstacle["size"][0],
            obstacle["size"][1],
            linewidth=1,
            edgecolor='black',
            facecolor='brown'
        )
        ax.add_patch(obstacle_rect)

    # Plot trajectory
    positions = trajectory_data['positions']
    x_positions = [pos[0] for pos in positions]
    y_positions = [pos[1] for pos in positions]
    ax.plot(x_positions, y_positions, color='blue', linewidth=2, label='Trajectory')

    ax.legend(loc='upper right')

    # Save the plot to TensorBoard
    fig.canvas.draw()
    image = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    image = image.reshape(fig.canvas.get_width_height()[::-1] + (4,))
    image_tensor = torch.tensor(image, dtype=torch.uint8).permute(2, 0, 1)  # Convert to (C, H, W) format
    writer.add_image(f'Episode_{episode_number}_Trajectory', image_tensor, global_step=episode_number)

    plt.close(fig)  # Close the plot to avoid opening a new window for each episode
    print(f"Episode {episode_number} trajectory saved to TensorBoard.")

# Main visualization function
def visualize_with_tensorboard(trajectory_dir='trajectories', log_dir='runs/cartrack'):
    # Initialize TensorBoard SummaryWriter
    writer = SummaryWriter(log_dir)

    # Initialize environment
    env = CarTrackEnv()

    # Load trajectories
    trajectories = load_trajectories(trajectory_dir)

    if not trajectories:
        print("No trajectories found. Please run the training script first.")
        return

    # Log metrics and plot trajectories
    for episode_number, trajectory_data in enumerate(trajectories, start=1):
        total_reward = sum(trajectory_data['rewards'])
        total_steps = len(trajectory_data['positions'])
        max_q_value = trajectory_data.get('max_q_value', None)  # Check if Q-values are stored

        # Log metrics to TensorBoard
        writer.add_scalar('Episode Reward', total_reward, episode_number)
        writer.add_scalar('Steps Taken', total_steps, episode_number)
        if max_q_value is not None:
            writer.add_scalar('Max Q-Value', max_q_value, episode_number)
        
        print(f"Logging Episode Reward: {total_reward} at step {episode_number}")
        print(f"Logging Steps Taken: {total_steps} at step {episode_number}")
        if max_q_value is not None:
            print(f"Logging Max Q-Value: {max_q_value} at step {episode_number}")


        # Save the trajectory plot to TensorBoard
        save_trajectory_to_tensorboard(env, trajectory_data, episode_number, writer)

    writer.close()
    print(f"TensorBoard logs saved to {log_dir}. Use 'tensorboard --logdir {log_dir}' to visualize.")

if __name__ == '__main__':
    visualize_with_tensorboard()
