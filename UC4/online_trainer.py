import time
import math
import matplotlib.pyplot as plt

def theta_s(x, y):
    if x >= 0:
        return math.atan(y)
    if x < 0:
        return math.atan(-y)

class OnlineTrainer:
    def __init__(self, robot, NN):
        """
        Args:
            robot (Robot): a robot instance following the pattern of
                VrepPioneerSimulation
            target (list): the target position [x, y, theta]
        """
        self.robot = robot
        self.network = NN

        self.alpha = [1/2, 1/2, 1/(math.pi)]  # Normalization with Cartesian world limit = -3m to +3m

        self.errors_x = []  # Storage for x errors
        self.errors_y = []  # Storage for y errors
        self.errors_theta = []  # Storage for theta errors
        self.positions = []  # List to store positions for plotting the trajectory

    def train(self, target):
        position = self.robot.get_position()

        # Store the initial position
        self.positions.append((position[0], position[1]))

        network_input = [0, 0, 0]
        network_input[0] = (position[0] - target[0]) * self.alpha[0]
        network_input[1] = (position[1] - target[1]) * self.alpha[1]
        network_input[2] = (position[2] - target[2] - theta_s(position[0], position[1])) * self.alpha[2]

        while self.running:
            start_time = time.time()
            command = self.network.runNN(network_input)  # Propagates error and computes wheel speeds at time t
            
            alpha_x = 1/2
            alpha_y = 2/3
            alpha_theta = 1.0 / (math.pi)

            # Compute errors
            error_x = (position[0] - target[0]) * self.alpha[0]
            error_y = (position[1] - target[1]) * self.alpha[1]
            error_theta = (position[2] - target[2] - theta_s(position[0], position[1])) * self.alpha[2]

            # Store the current position
            self.positions.append((position[0], position[1]))

            # Store errors
            self.errors_x.append(error_x)
            self.errors_y.append(error_y)
            self.errors_theta.append(error_theta)

            crit_before = (alpha_x**2 * error_x**2 +
                           alpha_y**2 * error_y**2 +
                           alpha_theta**2 * error_theta**2)

            self.robot.set_motor_velocity(command)  # Apply wheel speeds at time t
            time.sleep(0.050)  # Wait delta t
            position = self.robot.get_position()  # Get new robot position at time t+1

            network_input[0] = (position[0] - target[0]) * self.alpha[0]
            network_input[1] = (position[1] - target[1]) * self.alpha[1]
            network_input[2] = (position[2] - target[2] - theta_s(position[0], position[1])) * self.alpha[2]

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
                    self.network.backPropagate(grad, 0.3, 0.0)  # Grad, step size, momentum
                else:
                    self.network.backPropagate(grad, 0.3, 0.0)

        self.robot.set_motor_velocity([0, 0])  # Stop after training program ends
        self.running = False

    def plot_errors(self):
        """Plots the recorded errors with added graduations."""
        plt.figure(figsize=(12, 6))

        # Subplot pour les erreurs en X
        plt.subplot(3, 1, 1)
        plt.plot(self.errors_x, label='X Error')
        plt.xlabel('Iterations')
        plt.ylabel('X Error')
        plt.title('Erreur en X')
        plt.legend()
        plt.grid(True)  # Ajout de la grille pour la graduation
        plt.xticks(rotation=45)  # Rotation des graduations de l'axe x

        # Subplot pour les erreurs en Y
        plt.subplot(3, 1, 2)
        plt.plot(self.errors_y, label='Y Error')
        plt.xlabel('Iterations')
        plt.ylabel('Y Error')
        plt.title('Erreur en Y')
        plt.legend()
        plt.grid(True)  # Ajout de la grille pour la graduation
        plt.xticks(rotation=45)  # Rotation des graduations de l'axe y

        # Subplot pour les erreurs en Theta
        plt.subplot(3, 1, 3)
        plt.plot(self.errors_theta, label='Theta Error')
        plt.xlabel('Iterations')
        plt.ylabel('Theta Error')
        plt.title('Erreur en Theta')
        plt.legend()
        plt.grid(True)  # Ajout de la grille pour la graduation
        plt.xticks(rotation=45)  # Rotation des graduations de l'axe z

        plt.tight_layout()
        plt.show()

    def plot_trajectory(self):
        """Plots the trajectory of the robot."""
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
