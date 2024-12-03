import vrep
import math
import time
import numpy as np

def to_rad(deg):
    return 2 * math.pi * deg / 360

def to_deg(rad):
    return rad * 360 / (2 * math.pi)

# Simulation Configuration
IP_ADDRESS = '127.0.0.1'
PORT = 19997
SCENE_FILE = './Scene Pioneer_murs.ttt'
INITIAL_POSITION = [3, 2, to_rad(0)]

# Initialize connection
print('Program started')
vrep.simxFinish(-1)  # Close all opened connections
client_id = vrep.simxStart(IP_ADDRESS, PORT, True, True, 5000, 5)  # Connect to V-REP

if client_id != -1:
    print(f'Connected to remote API server on {IP_ADDRESS}:{PORT}')

    # Load scene and retrieve handles
    res = vrep.simxLoadScene(client_id, SCENE_FILE, 1, vrep.simx_opmode_oneshot_wait)
    res, pioneer = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx', vrep.simx_opmode_oneshot_wait)
    res, motor_left = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx_leftMotor', vrep.simx_opmode_oneshot_wait)
    res, motor_right = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx_rightMotor', vrep.simx_opmode_oneshot_wait)

    # Initialize sensor handles and detection data
    sensor_handles = np.zeros(16, dtype=int)
    detection_distances = np.zeros(16)

    for i in range(16):
        res, sensor_handle = vrep.simxGetObjectHandle(client_id, f"Pioneer_p3dx_ultrasonicSensor{i+1}", vrep.simx_opmode_blocking)
        sensor_handles[i] = sensor_handle
        res, _, detected_point, _, _ = vrep.simxReadProximitySensor(client_id, sensor_handle, vrep.simx_opmode_streaming)

    # Start simulation
    vrep.simxStartSimulation(client_id, vrep.simx_opmode_oneshot_wait)

    # Initialize variables
    position = INITIAL_POSITION
    left_motor_speed = 1.0
    right_motor_speed = 1.0
    command = [left_motor_speed, right_motor_speed]

    while True:
        # Set motor speeds
        vrep.simxSetJointTargetVelocity(client_id, motor_left, command[0], vrep.simx_opmode_oneshot_wait)
        vrep.simxSetJointTargetVelocity(client_id, motor_right, command[1], vrep.simx_opmode_oneshot_wait)

        # Get robot position and orientation
        res, position_data = vrep.simxGetObjectPosition(client_id, pioneer, -1, vrep.simx_opmode_oneshot_wait)
        position[0], position[1] = position_data[:2]
        res, orientation_data = vrep.simxGetObjectOrientation(client_id, pioneer, -1, vrep.simx_opmode_oneshot_wait)
        position[2] = orientation_data[2]  # In radians

        # Update sensor readings
        for i in range(16):
            res, detection_state, detected_point, _, _ = vrep.simxReadProximitySensor(client_id, sensor_handles[i], vrep.simx_opmode_buffer)
            if detection_state:
                detection_distances[i] = math.sqrt(sum([coord**2 for coord in detected_point]))
            else:
                detection_distances[i] = float('inf')

        # Braitenberg-like control logic
        left_motor_speed = 1.0
        right_motor_speed = 1.0

        for i, distance in enumerate(detection_distances):
            if distance < 0.5:
                if 4 <= i < 8:  # Right sensors
                    left_motor_speed = -1.0
                    right_motor_speed = 1.0
                elif i < 4:  # Left sensors
                    left_motor_speed = 1.0
                    right_motor_speed = -1.0
                else:  # Front sensors
                    left_motor_speed = right_motor_speed = -1.0

        command = [left_motor_speed, right_motor_speed]

    # Terminate simulation
    vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(client_id)
else:
    print(f'Unable to connect to {IP_ADDRESS}:{PORT}')
