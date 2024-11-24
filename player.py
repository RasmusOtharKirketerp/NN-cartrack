import matplotlib.pyplot as plt
import matplotlib.patches as patches
from env import CarTrackEnv  
import config
import keyboard 
import numpy as np
import time
import h5py
from datetime import datetime
import matplotlib

def draw_environment(ax_env, ax_reward, env, trajectory_x, trajectory_y, total_reward, total_penalties, rewards, breakdown):
    """
    Draw the environment including the trajectory, car position, and reward/penalty breakdown.
    Also updates the reward plot.
    """
    # Clear the axes
    ax_env.clear()
    ax_reward.clear()
    
    # Environment Plot
    # Set plot limits
    ax_env.set_xlim(0, env.track_length + 50)
    ax_env.set_ylim(-10, env.track_width + 10)
    ax_env.set_xlabel("Position X")
    ax_env.set_ylabel("Position Y")
    ax_env.set_title(f"Manual Play: Total Reward = {total_reward}, Total Penalties = {total_penalties}")
    
    # Draw track boundaries and finish line
    ax_env.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax_env.axhline(y=env.track_width, color="red", linestyle="--", linewidth=1)
    ax_env.axvline(x=env.track_length, color="green", linestyle="--", linewidth=1)
    
    # Fill track area
    ax_env.fill_between([0, env.track_length], 0, env.track_width, color="lightgrey", alpha=0.5)
    
    # Draw obstacles
    for obstacle in env.obstacles:
        obstacle_x_min = obstacle["center"][0] - obstacle["size"][0] / 2
        obstacle_y_min = obstacle["center"][1] - obstacle["size"][1] / 2
        obstacle_rect = patches.Rectangle(
            (obstacle_x_min, obstacle_y_min),
            obstacle["size"][0],
            obstacle["size"][1],
            linewidth=1,
            edgecolor="black",
            facecolor="brown",
        )
        ax_env.add_patch(obstacle_rect)
    
    # Draw trajectory
    ax_env.plot(trajectory_x, trajectory_y, color="blue", linewidth=2, label="Trajectory")
    # Draw current car position
    ax_env.plot(env.position[0], env.position[1], "ro", label="Car")
    
    # Add breakdown info to the plot
    breakdown_text = "\n".join([f"{key}: {value}" for key, value in breakdown.items()])
    ax_env.text(
        0.95,
        0.95,
        f"Reward/Penalty Breakdown:\n{breakdown_text}",
        transform=ax_env.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )
    
    # Add legend
    ax_env.legend()
    
    # Reward Plot
    ax_reward.plot(rewards, color='purple', marker='o')
    ax_reward.set_xlabel('Step')
    ax_reward.set_ylabel('Cumulative Reward')
    ax_reward.set_title('Reward over Time')
    ax_reward.grid(True)
    
    plt.tight_layout()
    plt.draw()

def save_plot(fig, timestamp):
    """
    Save the current figure as a PNG image with a timestamp.
    """
    filename = f"CarTrackEnv_Plot_{timestamp}.png"
    fig.savefig(filename)
    print(f"Plot saved as {filename}")

def save_data(filename, trajectory_x, trajectory_y, rewards, step_rewards, delta_x_rewards,
              border_penalties, obstacle_penalties, delta_x_list, positions_x, positions_y,
              velocities_x, velocities_y):
    """
    Save all relevant data to an HDF5 file.
    """
    with h5py.File(filename, 'w') as h5f:
        h5f.create_dataset('trajectory_x', data=np.array(trajectory_x))
        h5f.create_dataset('trajectory_y', data=np.array(trajectory_y))
        h5f.create_dataset('cumulative_rewards', data=np.array(rewards))
        h5f.create_dataset('step_rewards', data=np.array(step_rewards))
        h5f.create_dataset('delta_x_rewards', data=np.array(delta_x_rewards))
        h5f.create_dataset('border_penalties', data=np.array(border_penalties))
        h5f.create_dataset('obstacle_penalties', data=np.array(obstacle_penalties))
        h5f.create_dataset('delta_x', data=np.array(delta_x_list))
        h5f.create_dataset('positions_x', data=np.array(positions_x))
        h5f.create_dataset('positions_y', data=np.array(positions_y))
        h5f.create_dataset('velocities_x', data=np.array(velocities_x))
        h5f.create_dataset('velocities_y', data=np.array(velocities_y))
    print(f"Data saved to {filename}")

def manual_play():
    """
    Play the CarTrackEnv manually using keyboard inputs with reward/penalty breakdown display.
    """
    env = CarTrackEnv(mode='Player', verbose=True)
    state = env.reset()
    
    trajectory_x = []
    trajectory_y = []
    total_reward = 0
    total_penalties = 0
    rewards = []  # To store cumulative rewards
    step_rewards = []
    delta_x_rewards = []
    border_penalties = []
    obstacle_penalties = []
    delta_x_list = []
    positions_x = []
    positions_y = []
    velocities_x = []
    velocities_y = []
    
    # Initialize Matplotlib figure with two subplots
    fig, (ax_env, ax_reward) = plt.subplots(2, 1, figsize=(14, 10))
    plt.ion()  # Interactive mode

    # Maximize the figure window
    backend = matplotlib.get_backend()
    manager = plt.get_current_fig_manager()

    try:
        if backend == 'TkAgg':
            manager.window.state('zoomed')  # For Windows OS
        elif backend == 'wxAgg':
            manager.frame.Maximize(True)
        elif backend == 'Qt5Agg':
            manager.window.showMaximized()
        elif backend == 'MacOSX':
            manager.window.showFullScreen()
        else:
            # Fallback for other backends or systems
            fig.set_size_inches(plt.rcParams['figure.figsize'][0]*2, plt.rcParams['figure.figsize'][1]*2)
    except Exception as e:
        print(f"Could not maximize the window: {e}")

    # Draw the initial environment
    breakdown = {
        "Step Reward": 0,
        "Delta X Reward": 0,
        "Border Penalty": 0,
        "Obstacle Penalty": 0,
        "Delta X": 0,
        "Position": tuple(env.position),
        "Velocity": tuple(env.velocity),
    }
    rewards.append(0)  # Initial cumulative reward
    step_rewards.append(0)
    delta_x_rewards.append(0)
    border_penalties.append(0)
    obstacle_penalties.append(0)
    delta_x_list.append(0)
    positions_x.append(env.position[0])
    positions_y.append(env.position[1])
    velocities_x.append(env.velocity[0])
    velocities_y.append(env.velocity[1])
    draw_environment(ax_env, ax_reward, env, trajectory_x, trajectory_y, total_reward, total_penalties, rewards, breakdown)
    plt.pause(0.01)  # Display the plot before user input
    
    debounce_time = 0.2  # Adjust debounce time (in seconds)
    last_action_time = 0  # Keep track of the last action time
    last_save_time = 0    # Keep track of the last save time
    save_debounce_time = 0.5  # Debounce time for save action
    
    print("Use Arrow Keys: ↑ to Accelerate, ↓ to Decelerate, ← to Steer Left, → to Steer Right.")
    print("Press 'p' to save the plot and data. Press 'Esc' to Exit.")
    
    while not env.done:
        # Get current time
        current_time = time.time()
    
        # Get user input for actions
        action = None
        if current_time - last_action_time > debounce_time:  # Only process input after debounce time
            if keyboard.is_pressed("up"):
                action = 0  # Accelerate
            elif keyboard.is_pressed("down"):
                action = 1  # Decelerate
            elif keyboard.is_pressed("left"):
                action = 2  # Steer Left
            elif keyboard.is_pressed("right"):
                action = 3  # Steer Right
    
            if keyboard.is_pressed("esc"):
                print("Exiting manual play...")
                break
    
        # Check for 'p' key press to save plot and data
        save_action = False
        if current_time - last_save_time > save_debounce_time:
            if keyboard.is_pressed("p"):
                save_action = True
                last_save_time = current_time  # Update last save time to debounce
    
        # Only step the environment if an action is detected
        if action is not None:
            last_action_time = current_time  # Update last action time
            print(f"Action taken: {['Accelerate', 'Decelerate', 'Steer Left', 'Steer Right'][action]}")
    
            # Step the environment
            next_state, reward, done, info = env.step(action)
            state = next_state
    
            # Update trajectory
            trajectory_x.append(env.position[0])
            trajectory_y.append(env.position[1])
    
            # Update rewards and penalties
            total_reward += reward
            total_penalties += info.get('border_penalty', 0) + info.get('obstacle_penalty', 0)
            rewards.append(total_reward)  # Append cumulative reward
            step_rewards.append(reward)
            delta_x_rewards.append(info.get('delta_x_reward', 0))
            border_penalties.append(info.get('border_penalty', 0))
            obstacle_penalties.append(info.get('obstacle_penalty', 0))
            delta_x_list.append(info.get('delta_x', 0))
            positions_x.append(env.position[0])
            positions_y.append(env.position[1])
            velocities_x.append(env.velocity[0])
            velocities_y.append(env.velocity[1])
    
            # Prepare breakdown for display
            breakdown = {
                "Step Reward": reward,
                "Delta X Reward": info.get('delta_x_reward', 0),
                "Border Penalty": info.get('border_penalty', 0),
                "Obstacle Penalty": info.get('obstacle_penalty', 0),
                "Delta X": info.get('delta_x', 0),
                "Position": tuple(info.get('position', [])),
                "Velocity": tuple(info.get('velocity', [])),
            }
    
            # Draw the updated environment and rewards
            draw_environment(ax_env, ax_reward, env, trajectory_x, trajectory_y, total_reward, total_penalties, rewards, breakdown)
            plt.pause(0.01)  # Pause for the update
    
        # Handle save action
        if save_action:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            try:
                # Save the plot
                save_plot(fig, timestamp)
                
                # Prepare filename for HDF5
                hdf5_filename = f"CarTrackEnv_Data_{timestamp}.h5"
                
                # Save the data
                save_data(
                    hdf5_filename,
                    trajectory_x,
                    trajectory_y,
                    rewards,
                    step_rewards,
                    delta_x_rewards,
                    border_penalties,
                    obstacle_penalties,
                    delta_x_list,
                    positions_x,
                    positions_y,
                    velocities_x,
                    velocities_y
                )
            except Exception as e:
                print(f"Error saving data or plot: {e}")
    
    plt.ioff()
    plt.show()
    
    print(f"Game Over! Final Reward: {total_reward:.2f}, Total Penalties: {total_penalties:.2f}")

if __name__ == "__main__":
    manual_play()
