import numpy as np
import torch
import config
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


    def step(self, action):
        import math  # Add this import at the top of your file if not already present
        if self.done:
            return self._get_state(), 0, self.done, {}

        # Update velocity and position based on action
        self._update_velocity(action)
        self._apply_friction_and_cap_speed()
        self._update_position()

        # Calculate forward progress in the x-direction
        delta_x = self.position[0] - self.prev_position_x
        SCALING_FACTOR = 2  
        delta_x_reward = config.DELTA_X_REWARD_FACTOR * math.exp(max(delta_x, 0) / SCALING_FACTOR)

        self.timestep += 1  # Increment timestep

        # Check if the agent is off-track or has reached the goal
        off_track = self._check_if_off_track()
        reached_goal = self.position[0] >= self.track_length

        # Calculate penalties based on conditions
        border_penalty = self._calculate_border_penalty()  # Positive value
        obstacle_penalty = self._calculate_obstacle_penalty()  # Positive value

        if reached_goal:
            reward = config.REWARD_GOAL
            self.done = True
            self.reached_goal = True
        elif off_track:
            reward = -config.PENALTY_OFF_TRACK
            self.done = True
        elif self.timestep >= config.MAX_TIMESTEPS:
            reward = -config.PENALTY_TIME_LIMIT
            self.done = True
        else:
            reward = (
                delta_x_reward
                - config.TIME_PENALTY
                - border_penalty
                - obstacle_penalty
            )
            if self.done:  # Apply termination penalty for failed episodes
                reward -= config.PENALTY_TERMINATION


        # Update previous x-position for the next step calculation
        self.prev_position_x = self.position[0]

        # Prepare info dict
        info = {
            'delta_x_reward': delta_x_reward,
            'border_penalty': border_penalty,
            'obstacle_penalty': obstacle_penalty
        }

        return self._get_state(), reward, self.done, info


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

    def _calculate_border_penalty(self):
        # Calculate left and right boundary distances
        distance_to_left_border = self.position[1]
        distance_to_right_border = self.track_width - self.position[1]
        min_distance_to_border = min(distance_to_left_border, distance_to_right_border)

        # Penalize proximity to borders
        border_penalty = 0
        if min_distance_to_border < config.BORDER_THRESHOLD:
            border_penalty += (config.BORDER_THRESHOLD - min_distance_to_border) * config.PENALTY_BORDER  # Positive penalty

        # Penalize backward movement
        if self.position[0] < self.prev_position_x:
            border_penalty += config.PENALTY_BACKWARDS  # Positive penalty

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
                obstacle_penalty = config.PENALTY_OBSTACLE  # Positive penalty
                self.done = True
                break
            else:
                # Calculate proximity penalty
                distance_x = max(0, obstacle_x_min - self.position[0], self.position[0] - obstacle_x_max)
                distance_y = max(0, obstacle_y_min - self.position[1], self.position[1] - obstacle_y_max)
                min_distance_to_obstacle = min(distance_x, distance_y)

                if min_distance_to_obstacle < config.OBSTACLE_THRESHOLD:
                    obstacle_penalty += (config.OBSTACLE_THRESHOLD - min_distance_to_obstacle) * 50  # Positive penalty

        return obstacle_penalty
