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

        self.alpha = [1/6, 1/6, 1/(math.pi)]  # Normalisation avec limite du monde cartésien = -3m à +3m

        self.errors_x = []  # Stockage des erreurs sur x
        self.errors_y = []  # Stockage des erreurs sur y
        self.errors_theta = []  # Stockage des erreurs sur theta

    def train(self, target):
        position = self.robot.get_position()

        network_input = [0, 0, 0]
        network_input[0] = (position[0] - target[0]) * self.alpha[0]
        network_input[1] = (position[1] - target[1]) * self.alpha[1]
        network_input[2] = (position[2] - target[2] - theta_s(position[0], position[1])) * self.alpha[2]

        while self.running:
            debut = time.time()
            command = self.network.runNN(network_input)  # Propage erreur et calcule vitesses roues instant t

            alpha_x = 1/2
            alpha_y = 2/3
            alpha_theta = 1.0 / (math.pi)

            # Calcul des erreurs
            error_x = (position[0] - target[0]) * self.alpha[0]
            error_y = (position[1] - target[1]) * self.alpha[1]
            error_theta = (position[2] - target[2] - theta_s(position[0], position[1])) * self.alpha[2]

            # Stockage des erreurs
            self.errors_x.append(error_x)
            self.errors_y.append(error_y)
            self.errors_theta.append(error_theta)

            crit_av = (alpha_x**2 * error_x**2 +
                       alpha_y**2 * error_y**2 +
                       alpha_theta**2 * error_theta**2)

            self.robot.set_motor_velocity(command)  # Applique vitesses roues instant t
            time.sleep(0.050)  # Attend delta t
            position = self.robot.get_position()  # Obtient nouvelle position robot instant t+1

            network_input[0] = (position[0] - target[0]) * self.alpha[0]
            network_input[1] = (position[1] - target[1]) * self.alpha[1]
            network_input[2] = (position[2] - target[2] - theta_s(position[0], position[1])) * self.alpha[2]

            crit_ap = (alpha_x**2 * network_input[0]**2 +
                       alpha_y**2 * network_input[1]**2 +
                       alpha_theta**2 * network_input[2]**2)

            if self.training:
                delta_t = (time.time() - debut)

                grad = [
                    (-1 / delta_t) * (alpha_x**2 * error_x * delta_t * self.robot.r * math.cos(position[2]) +
                                      alpha_y**2 * error_y * delta_t * self.robot.r * math.sin(position[2]) -
                                      alpha_theta**2 * error_theta * delta_t * self.robot.r / (2 * self.robot.R)),

                    (-1 / delta_t) * (alpha_x**2 * error_x * delta_t * self.robot.r * math.cos(position[2]) +
                                      alpha_y**2 * error_y * delta_t * self.robot.r * math.sin(position[2]) +
                                      alpha_theta**2 * error_theta * delta_t * self.robot.r / (2 * self.robot.R))
                ]

                if crit_ap < crit_av:
                    self.network.backPropagate(grad, 0.3, 0.0)  # Grad, pas d'app, moment
                else:
                    self.network.backPropagate(grad, 0.3, 0.0)

        self.robot.set_motor_velocity([0, 0])  # Stop après arrêt du programme d'apprentissage
        self.running = False

    def plot_errors(self):
        """Trace les erreurs enregistrées."""
        plt.figure(figsize=(12, 6))

        plt.subplot(3, 1, 1)
        plt.plot(self.errors_x, label='Erreur X')
        plt.xlabel('Itérations')
        plt.ylabel('Erreur X')
        plt.legend()

        plt.subplot(3, 1, 2)
        plt.plot(self.errors_y, label='Erreur Y')
        plt.xlabel('Itérations')
        plt.ylabel('Erreur Y')
        plt.legend()

        plt.subplot(3, 1, 3)
        plt.plot(self.errors_theta, label='Erreur Theta')
        plt.xlabel('Itérations')
        plt.ylabel('Erreur Theta')
        plt.legend()

        plt.tight_layout()
        plt.show()

# Exemple d'utilisation
# robot = ...  # Initialiser une instance de robot
# NN = ...  # Initialiser une instance de réseau neuronal
# trainer = OnlineTrainer(robot, NN)
# trainer.train([x_target, y_target, theta_target])
# trainer.plot_errors()
