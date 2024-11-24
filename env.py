import numpy as np
import torch
import config
# Car Track Environment with High Maneuverability and Multiple Obstacles
class CarTrackEnv:
    def __init__(self, mode="NN", verbose=False):
        """
        Initialize the CarTrackEnv.

        Args:
            mode (str): Mode of operation. "NN" for neural network training, "Player" for manual play.
            verbose (bool): Whether to enable verbose debug statements (Player mode only).
        """
        self.mode = mode  # Set mode: "NN" or "Player"
        self.verbose = verbose  # Enable verbose debug output
        self.track_length = config.TRACK_LENGTH
        self.track_width = config.TRACK_WIDTH
        self.obstacles = config.OBSTACLES
        self.angle = 0.0  # Car's direction in radians
        self.reset()

    def reset(self):
        self.position = np.array([0.0, self.track_width / 2.0], dtype=np.float32)
        self.velocity = np.array([0.0, 0.0], dtype=np.float32)
        self.angle = 0.0  # Reset angle to face straight along the X-axis
        self.done = False
        self.timestep = 0
        self.position_history = [self.position.copy()]
        self.speed_history = [np.linalg.norm(self.velocity)]
        self.prev_position_x = self.position[0]
        self.reached_goal = False

        if self.mode == "Player":
            self._log(f"Environment reset. Starting at position: {self.position}")

        return self._get_state()
    
    def _get_state(self):
        """
        Get the current state of the environment as a tensor.

        Returns:
            torch.Tensor: The current state of the environment.
        """
        # Calculate distances to the left and right borders
        distance_to_left_border = self.position[1]
        distance_to_right_border = self.track_width - self.position[1]

        # Precompute obstacle distances using vectorized operations
        obstacle_positions = np.array([[obs["center"][0], obs["center"][1]] for obs in self.obstacles])
        obstacle_sizes = np.array([[obs["size"][0], obs["size"][1]] for obs in self.obstacles])
        obstacle_min = obstacle_positions - obstacle_sizes / 2  # Min bounds of obstacles
        obstacle_max = obstacle_positions + obstacle_sizes / 2  # Max bounds of obstacles

        # Calculate distances in x and y directions
        distance_x = np.maximum(0, np.maximum(obstacle_min[:, 0] - self.position[0], self.position[0] - obstacle_max[:, 0]))
        distance_y = np.maximum(0, np.maximum(obstacle_min[:, 1] - self.position[1], self.position[1] - obstacle_max[:, 1]))

        # Flatten obstacle distances
        obstacle_distances = np.stack((distance_x, distance_y), axis=1).flatten()

        # Concatenate state components
        state = np.concatenate([
            self.position,  # Current position
            self.velocity,  # Current velocity
            [distance_to_left_border, distance_to_right_border],  # Border distances
            obstacle_distances  # Distances to obstacles
        ])

        # Convert state to a PyTorch tensor and move it to the appropriate device
        return torch.tensor(state, dtype=torch.float32).to(config.DEVICE)

    def _log(self, message):
        """
        Log a message if the mode is Player.

        Args:
            message (str): Message to log.
        """
        if self.mode == "Player":
            print(message)

    def _verbose_log(self, message):
        """
        Log a detailed debug message if verbose mode is enabled.

        Args:
            message (str): Message to log.
        """
        if self.mode == "Player" and self.verbose:
            print(message)

    def step(self, action):
        import math
        if self.done:
            return self._get_state(), 0, self.done, {}

        # Update velocity and position based on action
        self._update_velocity(action)
        self._apply_friction_and_cap_speed()
        self._update_position()

        # Calculate forward progress in the x-direction
        delta_x = self.position[0] - self.prev_position_x
        SCALING_FACTOR = 2  

        if delta_x > 0:
            # Positive forward progress
            delta_x_reward = config.STEP_REWARD_FORWARD_PROGRESS * math.exp(delta_x / SCALING_FACTOR)
        else:
            # Negative backward progress (penalty)
            delta_x_reward = config.STEP_PENALTY_BACKWARD * abs(delta_x) * -1


        self.timestep += 1  # Increment timestep

        # Check if the agent is off-track or has reached the goal
        off_track = self._check_if_off_track()
        reached_goal = self.position[0] >= self.track_length

        # Calculate penalties based on conditions
        border_penalty = self._calculate_border_penalty()  # Positive value
        obstacle_penalty = self._calculate_obstacle_penalty()  # Positive value

        if reached_goal:
            reward = config.EPISODE_REWARD_GOAL_REACHED
            self.done = True
            self.reached_goal = True
            self._log("Goal reached! Reward granted.")
        elif off_track:
            reward = -config.EPISODE_PENALTY_OFF_TRACK
            self.done = True
            self._log("Off-track! Penalty applied.")
        elif self.timestep >= config.EPISODE_THRESHOLD_MAX_TIMESTEPS:
            reward = -config.EPISODE_PENALTY_TIME_LIMIT
            self.done = True
            self._log("Time limit exceeded! Penalty applied.")
        else:
            reward = (
                delta_x_reward
                - config.STEP_PENALTY_TIME
                - border_penalty
                - obstacle_penalty
            )
            if self.done:
                reward -= config.EPISODE_PENALTY_FAILURE

        # Verbose logging for Player mode
        self._verbose_log(
            f"Step {self.timestep}:\n"
            f"  Position: {self.position}\n"
            f"  Velocity: {self.velocity}\n"
            f"  Delta X: {delta_x:.2f}\n"
            f"  Delta X Reward: {delta_x_reward:.2f}\n"
            f"  Border Penalty: {border_penalty:.2f}\n"
            f"  Obstacle Penalty: {obstacle_penalty:.2f}\n"
            f"  Step Reward until this step: {reward:.2f}\n"
        )

        # Update previous x-position for the next step calculation
        self.prev_position_x = self.position[0]

        # Prepare info dict
        info = {
            'position': self.position.copy(),  # Current position
            'velocity': self.velocity.copy(),  # Current velocity
            'delta_x': delta_x,  # Forward progress in X direction
            'delta_x_reward': delta_x_reward,  # Reward for forward movement
            'border_penalty': border_penalty,  # Penalty for border proximity
            'obstacle_penalty': obstacle_penalty,  # Penalty for obstacle proximity
            'total_reward': reward  # Total reward for the current step
        }


        return self._get_state(), reward, self.done, info

    def _update_velocity(self, action):
        speed = np.linalg.norm(self.velocity)

        # Adjust speed based on action
        if action == 0:  # Accelerate
            speed += config.ACCELERATION
        elif action == 1:  # Decelerate
            speed = max(0, speed - config.ACCELERATION)

        # Adjust angle based on action
        if action == 2:  # Steer left
            self.angle += config.STEERING_ANGLE_INCREMENT  # Turn left
        elif action == 3:  # Steer right
            self.angle -= config.STEERING_ANGLE_INCREMENT  # Turn right

        # Normalize the angle to stay within [0, 2π]
        self.angle %= (2 * np.pi)


        # Convert speed and angle to velocity components
        self.velocity[0] = speed * np.cos(self.angle)  # X-component
        self.velocity[1] = speed * np.sin(self.angle)  # Y-component

        # Log updated velocity and angle
        self._verbose_log(f"Velocity updated: {self.velocity}")
        self._verbose_log(f"Angle updated: {self.angle:.2f}")
 

    def _apply_friction_and_cap_speed(self):
        # Apply friction uniformly to the speed
        self.velocity *= 0.95  # Apply friction to both X and Y components

        # Cap the speed at max_speed
        speed = np.linalg.norm(self.velocity)
        if speed > config.MAX_SPEED:
            self.velocity = (self.velocity / speed) * config.MAX_SPEED

        self._verbose_log(f"Speed after friction and cap: {np.linalg.norm(self.velocity):.2f}")
        self._verbose_log(f"Angle after friction and cap: {self.angle:.2f}")

    def _update_position(self):
        # Update the car's position based on velocity
        self.position += self.velocity
        self.position_history.append(self.position.copy())
        self.speed_history.append(np.linalg.norm(self.velocity))

        self._verbose_log(f"Position updated: {self.position}")

    def _check_if_off_track(self):
        off_track = (
            self.position[0] < 0 or
            self.position[0] > self.track_length or
            self.position[1] < 0 or
            self.position[1] > self.track_width
        )
        if off_track:
            self._log("Car is off-track!")
        return off_track

    def _calculate_border_penalty(self):
        return 0

    def _calculate_obstacle_penalty(self):
        for obstacle in self.obstacles:
            obstacle_x_min = obstacle["center"][0] - obstacle["size"][0] / 2
            obstacle_x_max = obstacle["center"][0] + obstacle["size"][0] / 2
            obstacle_y_min = obstacle["center"][1] - obstacle["size"][1] / 2
            obstacle_y_max = obstacle["center"][1] + obstacle["size"][1] / 2

            # Penalize for collisions
            if (
                obstacle_x_min <= self.position[0] <= obstacle_x_max and
                obstacle_y_min <= self.position[1] <= obstacle_y_max
            ):
                self.done = True
                self._log("Collision with obstacle!")
                return config.STEP_PENALTY_OBSTACLE  # Only penalize for collision

        return 0  # No penalty for proximity

