# visualize.py
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from datetime import datetime
import config  # Import the configuration file
from env import CarTrackEnv  # Ensure this module is accessible

def load_trajectories(trajectory_dir='trajectories'):
    """
    Load all trajectory files from the specified directory.
    Returns a list of trajectory data dictionaries.
    """
    # Get a list of trajectory files
    trajectory_files = sorted([
        os.path.join(trajectory_dir, f) for f in os.listdir(trajectory_dir)
        if f.startswith('trajectory_episode_') and f.endswith('.pkl')
    ])
    trajectories = []
    for file in trajectory_files:
        with open(file, 'rb') as f:
            trajectory_data = pickle.load(f)
            trajectories.append(trajectory_data)
    return trajectories

def load_episode_info(log_filename):
    """
    Load episode information strings from the log file.
    Returns a dictionary mapping episode numbers to their corresponding log lines.
    """
    episode_info = {}
    with open(log_filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('Episode'):
            # Extract episode number
            try:
                episode_part, _ = line.split(':', 1)
                episode_num = int(episode_part.strip().split(' ')[1])
                episode_info[episode_num] = line.strip()
            except (IndexError, ValueError):
                continue  # Skip malformed lines
    return episode_info

def plot_trajectory(env, trajectories, episode_info, episode_number):
    """
    Plot the trajectory for a given episode and display the corresponding info string.
    """
    # Setup visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.25)  # Adjust space for widgets and text

    ax.set_xlim(0, env.track_length + 50)
    ax.set_ylim(-10, env.track_width + 10)
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_title(f'Episode {episode_number}')

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
    unique = dict(zip(labels, handles))
    ax.legend(unique.values(), unique.keys(), loc='upper right')

    # Extract trajectory data
    trajectory_data = trajectories[episode_number - 1]  # Assuming episode_number starts at 1
    positions = trajectory_data['positions']
    x_positions = [pos[0] for pos in positions]
    y_positions = [pos[1] for pos in positions]

    # Plot the trajectory
    line, = ax.plot(x_positions, y_positions, color='blue', linewidth=2, label='Trajectory')

    # Add text box for episode data
    text_ax = fig.add_axes([0.05, 0.05, 0.9, 0.15])
    text_ax.axis('off')  # Hide axis

    def update_episode_info(episode_num):
        """
        Update the information text displayed on the plot based on the selected episode.
        """
        info_string = episode_info.get(episode_num, f"No data available for Episode {episode_num}.")
        text_ax.clear()
        text_ax.text(0.5, 0.5, info_string, horizontalalignment='center',
                    verticalalignment='center', fontsize=12, wrap=True)
        text_ax.axis('off')
        fig.canvas.draw_idle()

    # Initial display of episode data
    update_episode_info(episode_number)

    # Add a slider to select episodes
    ax_episode = plt.axes([0.2, 0.15, 0.65, 0.03])  # Adjust position for text box
    episode_slider = Slider(
        ax=ax_episode,
        label='Episode',
        valmin=1,
        valmax=len(trajectories),
        valinit=episode_number,
        valfmt='%0.0f',
        valstep=1
    )

    def update(val):
        """
        Update the plot and information text when the slider value changes.
        """
        episode = int(episode_slider.val)
        trajectory_data = trajectories[episode - 1]
        positions = trajectory_data['positions']
        x_positions = [pos[0] for pos in positions]
        y_positions = [pos[1] for pos in positions]

        # Update the data of the line
        line.set_data(x_positions, y_positions)
        ax.set_title(f'Episode {episode}')
        ax.figure.canvas.draw_idle()

        # Update episode info text
        update_episode_info(episode)

    episode_slider.on_changed(update)

    # Add keyboard event handler
    def on_key(event):
        """
        Allow left/right arrow keys to navigate between episodes.
        """
        if event.key == 'left':
            current_val = episode_slider.val
            if current_val > episode_slider.valmin:
                episode_slider.set_val(current_val - 1)
        elif event.key == 'right':
            current_val = episode_slider.val
            if current_val < episode_slider.valmax:
                episode_slider.set_val(current_val + 1)

    fig.canvas.mpl_connect('key_press_event', on_key)

    plt.show()

if __name__ == '__main__':
    # Initialize the environment
    env = CarTrackEnv()

    # Load all trajectories
    trajectories = load_trajectories()

    if not trajectories:
        print("No trajectories found. Please run the training script first.")
    else:
        # Parse the training log file
        # Assume the log file is named 'training_log_<timestamp>.txt'
        log_files = [f for f in os.listdir('.') if f.startswith('training_log_') and f.endswith('.txt')]
        if log_files:
            log_filename = max(log_files, key=os.path.getctime)  # Get the most recent log file
            episode_info = load_episode_info(log_filename)
        else:
            print("No training log file found.")
            episode_info = {}

        # Start by displaying the first episode
        initial_episode = 1  # Assuming episodes start at 1
        plot_trajectory(env, trajectories, episode_info, initial_episode)
