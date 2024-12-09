from BackProp_Python_v2 import NN
from vrep_pioneer_simulation import VrepPioneerSimulation
import matplotlib
from online_trainer import OnlineTrainer
import json
import threading

# Initialize the robot and the neural network
robot = VrepPioneerSimulation()
HL_size = 30  # Number of neurons in the hidden layer
network = NN(3, HL_size, 2)

# Load the previous network if needed
def load_previous_network():
    choice = input('Do you want to load the previous network? (y/n) --> ')
    if choice == 'y':
        with open('last_w.json') as fp:
            json_obj = json.load(fp)

        for i in range(3):
            for j in range(HL_size):
                network.wi[i][j] = json_obj["input_weights"][i][j]
        for i in range(HL_size):
            for j in range(2):
                network.wo[i][j] = json_obj["output_weights"][i][j]

# Initialize the trainer
trainer = OnlineTrainer(robot, network)

# Configure the training
def configure_training():
    choice = ''
    while choice not in ['y', 'n']:
        choice = input('Do you want to learn? (y/n) --> ')
    if choice == 'y':
        trainer.training = True
    else:
        trainer.training = False

# Input the first target
def get_target_input():
    target = input("Enter the first target: x y radian --> ")
    target = [float(x) for x in target.split()]
    print('New target: [%f, %f, %f]' % (target[0], target[1], target[2]))
    return target

# Training and plotting function
def train_and_plot(target):
    continue_running = True
    while continue_running:
        thread = threading.Thread(target=trainer.train, args=(target,))
        trainer.running = True
        thread.start()

        # Prompt to stop the training
        input("Press Enter to stop the current training")
        trainer.running = False

        # Display errors and position after stopping training
        trainer.plot_errors()
        trainer.plot_trajectory()

        # Ask to continue
        choice = ''
        while choice not in ['y', 'n']:
            choice = input("Do you want to continue? (y/n) --> ")

        if choice == 'y':
            choice_learning = ''
            while choice_learning not in ['y', 'n']:
                choice_learning = input('Do you want to learn? (y/n) --> ')
            if choice_learning == 'y':
                trainer.training = True
            elif choice_learning == 'n':
                trainer.training = False

            target = get_target_input()
        elif choice == 'n':
            continue_running = False

# Save weights to a JSON file
def save_weights():
    json_obj = {"input_weights": network.wi, "output_weights": network.wo}
    with open('last_w.json', 'w') as fp:
        json.dump(json_obj, fp)
    print("The last weights have been stored in last_w.json")

# Execute the program
if __name__ == "__main__":
    load_previous_network()
    configure_training()
    target = get_target_input()
    train_and_plot(target)
    save_weights()
