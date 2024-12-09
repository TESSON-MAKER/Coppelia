from BackProp_Python_v2 import NN
from vrep_pioneer_simulation import VrepPioneerSimulation
import matplotlib
from online_trainer import OnlineTrainer
import json
import threading

# Initialisation du robot et du réseau de neurones
robot = VrepPioneerSimulation()
HL_size = 40  # Nombre de neurones dans la couche cachée
network = NN(3, HL_size, 2)

# Chargement du réseau précédent si nécessaire
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

trainer = OnlineTrainer(robot, network)

# Configuration de l'entraînement
choice = ''
while choice != 'y' and choice != 'n':
    choice = input('Do you want to learn? (y/n) --> ')

if choice == 'y':
    trainer.training = True
elif choice == 'n':
    trainer.training = False

# Entrée de la première cible
target = input("Enter the first target: x y radian --> ")
target = target.split()
for i in range(len(target)):
    target[i] = float(target[i])
print('New target: [%f, %f, %f]' % (target[0], target[1], target[2]))

# Boucle d'entraînement
continue_running = True
while continue_running:
    thread = threading.Thread(target=trainer.train, args=(target,))
    trainer.running = True
    thread.start()

    # Demande pour arrêter l'entraînement
    input("Press Enter to stop the current training")
    trainer.running = False

    # Affichage des erreurs et de la position après l'arrêt de l'entraînement
    trainer.plot_errors()
    trainer.plot_trajectory()  # Ajout de la fonction pour afficher la trajectoire

    choice = ''
    while choice != 'y' and choice != 'n':
        choice = input("Do you want to continue? (y/n) --> ")

    if choice == 'y':
        choice_learning = ''
        while choice_learning != 'y' and choice_learning != 'n':
            choice_learning = input('Do you want to learn? (y/n) --> ')
        if choice_learning == 'y':
            trainer.training = True
        elif choice_learning == 'n':
            trainer.training = False
        target = input("Move the robot to the initial point and enter the new target: x y radian --> ")
        target = target.split()
        for i in range(len(target)):
            target[i] = float(target[i])
        print('New target: [%f, %f, %f]' % (target[0], target[1], target[2]))
    elif choice == 'n':
        continue_running = False

# Sauvegarde des poids dans un fichier JSON
json_obj = {"input_weights": network.wi, "output_weights": network.wo}
with open('last_w.json', 'w') as fp:
    json.dump(json_obj, fp)

print("The last weights have been stored in last_w.json")
