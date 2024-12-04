import vrep
import math
import numpy as np

# Conversion functions
def to_rad(deg):
    return 2 * math.pi * deg / 360

def to_deg(rad):
    return rad * 360 / (2 * math.pi)

# Simulation configuration
IP = '127.0.0.1'
PORT = 19997
SCENE = './Pioneer_murs.ttt'
POSITION_INIT = [3, 2, to_rad(0)]

print('Program started')
vrep.simxFinish(-1)  # Close all opened connections
client_id = vrep.simxStart(IP, PORT, True, True, 5000, 5)  # Connect to V-REP

if client_id != -1:
    print(f'Connected to remote API server on {IP}:{PORT}')

    # Load the scene
    res = vrep.simxLoadScene(client_id, SCENE, 1, vrep.simx_opmode_oneshot_wait)

    # Get handles for robot and motors
    res, pioneer = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx', vrep.simx_opmode_oneshot_wait)
    res, left_motor = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx_leftMotor', vrep.simx_opmode_oneshot_wait)
    res, right_motor = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx_rightMotor', vrep.simx_opmode_oneshot_wait)

    # Initialize sensor handles and detection arrays
    sensor_handles = np.zeros(16)
    detection_IR = np.zeros(16)

    # Get sensor handles and start streaming proximity sensor data
    for i in range(1, 17):
        res, sensor_handle = vrep.simxGetObjectHandle(client_id, f"Pioneer_p3dx_ultrasonicSensor{i}", vrep.simx_opmode_blocking)
        sensor_handles[i - 1] = sensor_handle
        res, _, _, _, _ = vrep.simxReadProximitySensor(client_id, sensor_handle, vrep.simx_opmode_streaming)

    # Start the simulation
    vrep.simxStartSimulation(client_id, vrep.simx_opmode_oneshot_wait)

    # Braitenberg parameters
    V0 = 2.0
    NO_DETECTION_DIST = 0.5
    MAX_DETECTION_DIST = 0.2
    detect = [0] * 16
    BRAITENBERG_L = [-0.2, -0.4, -0.6, -0.8, -1, -1.2, -1.4, -1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    BRAITENBERG_R = [-1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]

    try:
        while True:
            # Update detection states and compute activation levels
            for i in range(16):
                res, detection_state, detected_point, _, _ = vrep.simxReadProximitySensor(client_id, int(sensor_handles[i]), vrep.simx_opmode_buffer)
                if detection_state:
                    dist_to_object = math.sqrt(detected_point[0] ** 2 + detected_point[1] ** 2 + detected_point[2] ** 2)
                    if dist_to_object < NO_DETECTION_DIST:
                        detect[i] = 1 - ((dist_to_object - MAX_DETECTION_DIST) / (NO_DETECTION_DIST - MAX_DETECTION_DIST))
                        detect[i] = max(detect[i], 0)
                    else:
                        detect[i] = 0
                else:
                    detect[i] = 0

            # Compute motor velocities
            v_left = V0
            v_right = V0
            for i in range(16):
                v_left += BRAITENBERG_L[i] * detect[i]
                v_right += BRAITENBERG_R[i] * detect[i]

            # Apply motor velocities
            vrep.simxSetJointTargetVelocity(client_id, left_motor, v_left, vrep.simx_opmode_oneshot)
            vrep.simxSetJointTargetVelocity(client_id, right_motor, v_right, vrep.simx_opmode_oneshot)

    except KeyboardInterrupt:
        print("Simulation stopped by user")

    # Stop the simulation
    vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(client_id)
    print("Simulation finished")

else:
    print(f'Unable to connect to {IP}:{PORT}')
