# visualize.py
import os
import pickle
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.widgets import Slider
from datetime import datetime
import config  # Import the configuration file
from nn import CarTrackEnv  # Import the environment

def load_trajectories(trajectory_dir='trajectories'):
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

def parse_training_log(log_filename):
    episode_data = {}
    with open(log_filename, 'r') as f:
        lines = f.readlines()
    for line in lines:
        if line.startswith('Episode'):
            # Remove 'Episode ' and split at ': '
            episode_num, rest = line[len('Episode '):].split(': ', 1)
            episode_num = int(episode_num)
            # Split the rest of the line by ', '
            metrics = rest.strip().split(', ')
            data = {}
            for metric in metrics:
                if 'Reward Breakdown ->' in metric:
                    # Handle Reward Breakdown
                    breakdown = metric.split('->')[1].strip()
                    breakdown_items = breakdown.split(', ')
                    for item in breakdown_items:
                        key, value = item.split(': ')
                        data[key.strip()] = float(value)
                else:
                    key_value = metric.split(' = ')
                    if len(key_value) == 2:
                        key, value = key_value
                        key = key.strip()
                        if key == 'Epsilon':
                            data[key] = float(value)
                        elif key == 'Steps':
                            data[key] = int(value)
                        else:
                            try:
                                data[key] = float(value)
                            except ValueError:
                                data[key] = value
            episode_data[episode_num] = data
    return episode_data

def format_value(value, fmt=".2f"):
    if isinstance(value, (int, float)):
        return format(value, fmt)
    else:
        return str(value)

def plot_trajectory(env, trajectories, episode_data, episode_number):
    # Setup visualization
    fig, ax = plt.subplots(figsize=(15, 8))
    plt.subplots_adjust(bottom=0.25)  # Adjust space for widgets and text

    ax.set_xlim(0, env.track_length + 50)
    ax.set_ylim(-10, env.track_width + 10)
    ax.set_xlabel('Position X')
    ax.set_ylabel('Position Y')
    ax.set_title(f'Episode {episode_number + 1}')

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
    trajectory_data = trajectories[episode_number]
    positions = trajectory_data['positions']
    x_positions = [pos[0] for pos in positions]
    y_positions = [pos[1] for pos in positions]

    # Plot the trajectory
    line, = ax.plot(x_positions, y_positions, color='blue', linewidth=2)

    # Add text box for episode data
    text_ax = fig.add_axes([0.05, 0.05, 0.9, 0.15])
    text_ax.axis('off')  # Hide axis

    def update_episode_data(episode_num):
        data = episode_data.get(episode_num + 1, None)
        if data:
            # Extract values and use format_value to safely format them
            reward = format_value(data.get('Reward', 'N/A'))
            avg_reward = format_value(data.get('Avg Reward (100)', 'N/A'))
            epsilon = format_value(data.get('Epsilon', 'N/A'), fmt=".3f")
            avg_loss = format_value(data.get('Avg Loss', 'N/A'), fmt=".4f")
            max_q_value = format_value(data.get('Max Q-Value', 'N/A'))
            steps = data.get('Steps', 'N/A')
            delta_x_reward = format_value(data.get('Delta_x Reward', 'N/A'))
            border_penalty = format_value(data.get('Border Penalty', 'N/A'))
            obstacle_penalty = format_value(data.get('Obstacle Penalty', 'N/A'))

            info_text = (
                f"Reward = {reward}, "
                f"Avg Reward (100) = {avg_reward}, "
                f"Epsilon = {epsilon}, "
                f"Avg Loss = {avg_loss}, "
                f"Max Q-Value = {max_q_value}, "
                f"Steps = {steps}\n"
                f"Delta_x Reward = {delta_x_reward}, "
                f"Border Penalty = {border_penalty}, "
                f"Obstacle Penalty = {obstacle_penalty}"
            )
        else:
            info_text = f"No data available for Episode {episode_num + 1}."

        text_ax.clear()
        text_ax.text(0.5, 0.5, info_text, horizontalalignment='center', verticalalignment='center', fontsize=12)
        text_ax.axis('off')

    # Initial display of episode data
    update_episode_data(episode_number)

    # Add a slider to select episodes
    ax_episode = plt.axes([0.2, 0.15, 0.65, 0.03])  # Adjust position for text box
    episode_slider = Slider(
        ax=ax_episode,
        label='Episode',
        valmin=1,
        valmax=len(trajectories),
        valinit=episode_number + 1,
        valfmt='%0.0f',
        valstep=1
    )

    def update(val):
        episode = int(episode_slider.val) - 1
        trajectory_data = trajectories[episode]
        positions = trajectory_data['positions']
        x_positions = [pos[0] for pos in positions]
        y_positions = [pos[1] for pos in positions]

        # Update the data of the line
        line.set_data(x_positions, y_positions)
        ax.set_title(f'Episode {episode + 1}')
        ax.figure.canvas.draw_idle()

        # Update episode data text
        update_episode_data(episode)

    episode_slider.on_changed(update)

    # Add keyboard event handler
    def on_key(event):
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
    env = CarTrackEnv()
    trajectories = load_trajectories()

    if not trajectories:
        print("No trajectories found. Please run the training script first.")
    else:
        # Parse the training log file
        # Assume the log file is named 'training_log_<timestamp>.txt'
        log_files = [f for f in os.listdir('.') if f.startswith('training_log_') and f.endswith('.txt')]
        if log_files:
            log_filename = max(log_files, key=os.path.getctime)  # Get the most recent log file
            episode_data = parse_training_log(log_filename)
        else:
            print("No training log file found.")
            episode_data = {}

        # Start by displaying the first episode
        plot_trajectory(env, trajectories, episode_data, 0)
