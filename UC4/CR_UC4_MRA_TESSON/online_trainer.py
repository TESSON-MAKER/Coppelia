import time
import math
import matplotlib.pyplot as plt

def calculate_theta(x, y):
    """Calculate the angle theta based on x and y coordinates."""
    if x >= 0:
        return math.atan(y)
    if x < 0:
        return math.atan(-y)

class OnlineTrainer:
    def __init__(self, robot, NN):
        """
        Initialize the OnlineTrainer.

        Args:
            robot (Robot): an instance of the robot following the VrepPioneerSimulation pattern.
            NN (NeuralNetwork): an instance of the neural network.
        """
        self.robot = robot
        self.network = NN
        self.alpha = [1/2, 1/2, 1/(math.pi)]  # Normalization for Cartesian coordinates (-3m to +3m)

        # Lists for tracking errors and positions
        self.errors_x = []
        self.errors_y = []
        self.errors_theta = []
        self.positions = []

    def train(self, target):
        """Train the robot to reach the target position."""
        position = self.robot.get_position()

        # Store the initial position
        self.positions.append((position[0], position[1]))

        # Prepare the input for the neural network
        network_input = [0, 0, 0]
        network_input[0] = (position[0] - target[0]) * self.alpha[0]
        network_input[1] = (position[1] - target[1]) * self.alpha[1]
        network_input[2] = (position[2] - target[2] - calculate_theta(position[0], position[1])) * self.alpha[2]

        while self.running:
            start_time = time.time()
            command = self.network.runNN(network_input)  # Propagate through the network and compute wheel speeds
            
            # Coefficients for error weighting
            alpha_x = 1/2
            alpha_y = 2/3
            alpha_theta = 1.0 / math.pi

            # Calculate errors
            error_x = (position[0] - target[0]) * self.alpha[0]
            error_y = (position[1] - target[1]) * self.alpha[1]
            error_theta = (position[2] - target[2] - calculate_theta(position[0], position[1])) * self.alpha[2]

            # Store the current position and errors
            self.positions.append((position[0], position[1]))
            self.errors_x.append(error_x)
            self.errors_y.append(error_y)
            self.errors_theta.append(error_theta)

            # Calculate the performance metric before and after
            crit_before = (alpha_x**2 * error_x**2 +
                           alpha_y**2 * error_y**2 +
                           alpha_theta**2 * error_theta**2)

            # Apply the motor velocities
            self.robot.set_motor_velocity(command)
            time.sleep(0.050)  # Wait for delta t
            position = self.robot.get_position()  # Get new position at time t+1

            # Update network input for the next step
            network_input[0] = (position[0] - target[0]) * self.alpha[0]
            network_input[1] = (position[1] - target[1]) * self.alpha[1]
            network_input[2] = (position[2] - target[2] - calculate_theta(position[0], position[1])) * self.alpha[2]

            crit_after = (alpha_x**2 * network_input[0]**2 +
                          alpha_y**2 * network_input[1]**2 +
                          alpha_theta**2 * network_input[2]**2)

            if self.training:
                delta_t = (time.time() - start_time)
                grad = [
                    (-1 / delta_t) * (alpha_x**2 * error_x * delta_t * self.robot.r * math.cos(position[2]) +
                                      alpha_y**2 * error_y * delta_t * self.robot.r * math.sin(position[2]) -
                                      alpha_theta**2 * error_theta * delta_t * self.robot.r / (2 * self.robot.R)),
                    (-1 / delta_t) * (alpha_x**2 * error_x * delta_t * self.robot.r * math.cos(position[2]) +
                                      alpha_y**2 * error_y * delta_t * self.robot.r * math.sin(position[2]) +
                                      alpha_theta**2 * error_theta * delta_t * self.robot.r / (2 * self.robot.R))
                ]

                if crit_after < crit_before:
                    self.network.backPropagate(grad, 0.3, 0.0)  # Perform backpropagation if performance improves
                else:
                    self.network.backPropagate(grad, 0.3, 0.0)  # Continue backpropagation regardless

        # Stop the robot after training ends
        self.robot.set_motor_velocity([0, 0])
        self.running = False

    def plot_errors(self):
        """Plot the recorded errors."""
        plt.figure(figsize=(12, 6))

        # Subplot for X errors
        plt.subplot(3, 1, 1)
        plt.plot(self.errors_x, label='X Error')
        plt.xlabel('Iterations')
        plt.ylabel('X Error')
        plt.title('X Error')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        # Subplot for Y errors
        plt.subplot(3, 1, 2)
        plt.plot(self.errors_y, label='Y Error')
        plt.xlabel('Iterations')
        plt.ylabel('Y Error')
        plt.title('Y Error')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        # Subplot for Theta errors
        plt.subplot(3, 1, 3)
        plt.plot(self.errors_theta, label='Theta Error')
        plt.xlabel('Iterations')
        plt.ylabel('Theta Error')
        plt.title('Theta Error')
        plt.legend()
        plt.grid(True)
        plt.xticks(rotation=45)

        plt.tight_layout()
        plt.show()

    def plot_trajectory(self):
        """Plot the robot's trajectory."""
        x_values = [pos[0] for pos in self.positions]
        y_values = [pos[1] for pos in self.positions]

        plt.figure(figsize=(10, 10))
        plt.plot(x_values, y_values, marker='o', linestyle='-', color='b', label='Trajectory')
        plt.title('Robot Trajectory')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.axhline(0, color='black', linewidth=1)  # X-axis
        plt.axvline(0, color='black', linewidth=1)  # Y-axis
        plt.grid(True)
        plt.legend()
        plt.show()
