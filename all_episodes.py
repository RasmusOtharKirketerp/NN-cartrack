# plot_all_episodes_highlight_top10.py
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib import cm
from matplotlib.colors import Normalize
import matplotlib.lines as mlines
from env import CarTrackEnv  # Ensure this module is accessible
import re
from datetime import datetime

def load_trajectories(trajectory_dir='trajectories'):
    """
    Load all trajectory files from the specified directory.

    Args:
        trajectory_dir (str): Directory containing trajectory files.

    Returns:
        list: A list of trajectory data dictionaries sorted by episode number.
    """
    # Get a list of trajectory files
    trajectory_files = sorted(
        [
            f for f in os.listdir(trajectory_dir)
            if f.startswith('trajectory_episode_') and f.endswith('.pkl')
        ],
        key=lambda x: int(re.search(r'trajectory_episode_(\d+)\.pkl', x).group(1))  # Sort by episode number
    )

    trajectories = []
    for file in trajectory_files:
        file_path = os.path.join(trajectory_dir, file)
        with open(file_path, 'rb') as f:
            trajectory_data = pickle.load(f)
            trajectories.append(trajectory_data)
    return trajectories

def load_episode_info(log_filename):
    """
    Load episode information strings and Rewards from the log file.

    Args:
        log_filename (str): Filename of the training log.

    Returns:
        dict: A dictionary mapping episode numbers to their corresponding log lines and Rewards.
              Format: {episode_num: {'log': log_line, 'reward': reward_value}, ...}
    """
    episode_info = {}
    reward_pattern = re.compile(r'Reward\s*=\s*([-+]?\d*\.\d+|\d+)')  # Regex to extract Reward

    with open(log_filename, 'r') as f:
        lines = f.readlines()

    for line in lines:
        if line.startswith('Episode'):
            try:
                # Extract episode number
                episode_part, _ = line.split(':', 1)
                episode_num = int(episode_part.strip().split(' ')[1])

                # Extract Reward using regex
                reward_match = reward_pattern.search(line)
                if reward_match:
                    reward = float(reward_match.group(1))
                else:
                    reward = None  # If Reward not found

                episode_info[episode_num] = {'log': line.strip(), 'reward': reward}
            except (IndexError, ValueError):
                continue  # Skip malformed lines
    return episode_info

def plot_all_trajectories(env, trajectories, episode_info, top_n=10, output_file='all_episodes_highlight_top10.png'):
    """
    Plot all trajectories on a single plot, highlight the top N episodes based on Reward,
    and annotate their scores.

    Args:
        env (CarTrackEnv): The environment instance containing track and obstacle details.
        trajectories (list): List of trajectory data dictionaries.
        episode_info (dict): Dictionary mapping episode numbers to their log lines and Rewards.
        top_n (int): Number of top episodes to highlight.
        output_file (str): Filename for the saved plot.
    """
    # Initialize the plot
    fig, ax = plt.subplots(figsize=(20, 10))
    plt.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05)  # Adjust margins

    # Set plot limits based on environment
    ax.set_xlim(0, env.track_length + 50)
    ax.set_ylim(-10, env.track_width + 10)
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_title('All Episodes Trajectories with Top 10 Highlights')

    # Draw track boundaries and finish line
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax.axhline(y=env.track_width, color='red', linestyle='--', linewidth=1)
    ax.axvline(x=env.track_length, color='green', linestyle='--', linewidth=1)

    # Fill track area
    ax.fill_between([0, env.track_length], 0, env.track_width, color='lightgrey', alpha=0.5)

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
            facecolor='brown'
        )
        ax.add_patch(obstacle_rect)

    # Define a color map for trajectories
    cmap = cm.get_cmap('viridis')
    norm = Normalize(vmin=1, vmax=len(trajectories))
    colors = [cmap(norm(i)) for i in range(1, len(trajectories) + 1)]

    # Extract Rewards and identify top N episodes
    episode_rewards = {}
    for idx, trajectory in enumerate(trajectories, start=1):
        reward = episode_info.get(idx, {}).get('reward', None)
        if reward is not None:
            episode_rewards[idx] = reward

    # Sort episodes by Reward in descending order
    sorted_episodes = sorted(episode_rewards.items(), key=lambda x: x[1], reverse=True)
    top_episodes = sorted_episodes[:top_n]
    top_episode_nums = set([ep for ep, rw in top_episodes])

    # Plot each trajectory
    for idx, trajectory in enumerate(trajectories, start=1):
        positions = trajectory['positions']
        x_positions = [pos[0] for pos in positions]
        y_positions = [pos[1] for pos in positions]

        if idx in top_episode_nums:
            # Highlight top episodes with thicker lines and distinct color
            ax.plot(x_positions, y_positions, color="blue", linewidth=3, alpha=1.0)
            print(f"Episode {idx} plotted with Reward: {episode_rewards[idx]}")
        else:
            # Plot other episodes with thinner lines and lower opacity
            ax.plot(x_positions, y_positions, color=colors[idx - 1], linewidth=1, alpha=0.3)

    # Annotate top N episodes with their Rewards
    for ep_num, reward in top_episodes:
        trajectory = trajectories[ep_num - 1]
        positions = trajectory['positions']
        end_pos = positions[-1]  # Last position for annotation

        # Offset the text slightly for better visibility
        text_x = end_pos[0]
        text_y = end_pos[1]

        ax.text(
            text_x, text_y,
            f'Ep {ep_num}: {reward}',
            fontsize=9,
            fontweight='bold',
            color='black',
            bbox=dict(facecolor='yellow', alpha=0.6, edgecolor='none', pad=1)
        )

    # Remove all legends and tables (already omitted)
    # Save the plot as a PNG file
    plt.savefig(output_file, bbox_inches='tight', dpi=300)
    plt.show()
    plt.close(fig)  # Close the figure to free memory
    print(f"All episodes have been plotted and saved to '{output_file}'. Top {top_n} episodes highlighted.")

def main():
    """
    Main function to execute the plotting of all episodes with top 10 highlights.
    """
    # Initialize the environment
    env = CarTrackEnv()

    # Load all trajectories
    trajectories = load_trajectories()

    if not trajectories:
        print("No trajectories found. Please ensure that the 'trajectories' directory contains trajectory files.")
        return

    # Parse the training log file
    # Assume the log file is named 'training_log_<timestamp>.txt' and located in the current directory
    log_files = [f for f in os.listdir('.') if f.startswith('training_log_') and f.endswith('.txt')]
    if log_files:
        log_filename = max(log_files, key=os.path.getctime)  # Get the most recent log file
        episode_info = load_episode_info(log_filename)
    else:
        print("No training log file found. Proceeding without episode information.")
        episode_info = {}

    # Plot all episodes and save as PNG with top 10 highlights
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Add the timestamp to the filename
    output_file = f'all_episodes_highlight_top10_{timestamp}.png'
    plot_all_trajectories(env, trajectories, episode_info, top_n=20, output_file=output_file)

if __name__ == '__main__':
    main()
