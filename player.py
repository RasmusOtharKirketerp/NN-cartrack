import matplotlib.pyplot as plt
import matplotlib.patches as patches
from env import CarTrackEnv  
import config
import keyboard 
import numpy as np


def draw_environment(ax, env, trajectory_x, trajectory_y, total_reward, total_penalties, breakdown):
    """
    Draw the environment including the trajectory, car position, and reward/penalty breakdown.
    """
    ax.clear()
    
    # Set plot limits
    ax.set_xlim(0, env.track_length + 50)
    ax.set_ylim(-10, env.track_width + 10)
    ax.set_xlabel("Position X")
    ax.set_ylabel("Position Y")
    ax.set_title(f"Manual Play: Total Reward = {total_reward}, Total Penalties = {total_penalties}")

    # Draw track boundaries and finish line
    ax.axhline(y=0, color="red", linestyle="--", linewidth=1)
    ax.axhline(y=env.track_width, color="red", linestyle="--", linewidth=1)
    ax.axvline(x=env.track_length, color="green", linestyle="--", linewidth=1)

    # Fill track area
    ax.fill_between([0, env.track_length], 0, env.track_width, color="lightgrey", alpha=0.5)

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
        ax.add_patch(obstacle_rect)

    # Draw trajectory
    ax.plot(trajectory_x, trajectory_y, color="blue", linewidth=2, label="Trajectory")
    # Draw current car position
    ax.plot(env.position[0], env.position[1], "ro", label="Car")

    # Add breakdown info to the plot
    breakdown_text = "\n".join([f"{key}: {value}" for key, value in breakdown.items()])
    ax.text(
        0.95,
        0.95,
        f"Reward/Penalty Breakdown:\n{breakdown_text}",
        transform=ax.transAxes,
        fontsize=10,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="black"),
    )

    # Add legend
    ax.legend()
    plt.draw()




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

    # Initialize Matplotlib figure
    fig, ax = plt.subplots(figsize=(14, 10))  # Increased figure size for better visibility
    plt.ion()  # Interactive mode

    print("Use Arrow Keys: ↑ to Accelerate, ↓ to Decelerate, ← to Steer Left, → to Steer Right. Press 'Esc' to Exit.")

    while not env.done:
        # Get user input
        action = None
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

        if action is not None:
            # Step the environment
            next_state, reward, done, info = env.step(action)
            state = next_state

            # Update trajectory
            trajectory_x.append(env.position[0])
            trajectory_y.append(env.position[1])

            # Update rewards and penalties
            total_reward += reward
            total_penalties += sum(info.values())

            print("Info Dictionary:", info)
            print("Info Types:", {k: type(v) for k, v in info.items()})



            # Prepare breakdown for display
            # Simplified breakdown preparation
            breakdown = {
                "Step Reward": f"{info['total_reward']:.2f}",
                "Delta X Reward": f"{info['delta_x_reward']:.2f}",
                "Border Penalty": f"{info['border_penalty']:.2f}",
                "Obstacle Penalty": f"{info['obstacle_penalty']:.2f}",
                "Delta X": f"{info['delta_x']:.2f}",
                "Position": f"({info['position'][0]:.2f}, {info['position'][1]:.2f})",
                "Velocity": f"({info['velocity'][0]:.2f}, {info['velocity'][1]:.2f})"
            }









            # Draw the updated environment
            draw_environment(ax, env, trajectory_x, trajectory_y, total_reward, total_penalties, breakdown)
            plt.pause(0.01)  # Pause for the update

    plt.ioff()
    plt.show()

    print(f"Game Over! Final Reward: {total_reward}, Total Penalties: {total_penalties}")


if __name__ == "__main__":
    manual_play()
