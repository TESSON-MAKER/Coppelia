import vrep
import math
import time
import numpy as np


def to_rad(deg):
    return 2*math.pi*deg/360

def to_deg(rad):
    return rad*360/(2*math.pi)


# simulation config
ip = '127.0.0.1'
port = 19997
scene = './Pioneer_murs.ttt'
position_init = [3,2,to_rad(0)]


print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
client_id=vrep.simxStart(ip,port,True,True,5000,5) # Connect to V-REP

if client_id!=-1:
    print ('Connected to remote API server on %s:%s' % (ip, port))
    res = vrep.simxLoadScene(client_id, scene, 1, vrep.simx_opmode_oneshot_wait)
    res, pioneer = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx', vrep.simx_opmode_oneshot_wait)
    res, left_motor = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx_leftMotor', vrep.simx_opmode_oneshot_wait)
    res, right_motor = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx_rightMotor', vrep.simx_opmode_oneshot_wait)
    

    # For Sensors
    sensor_handles = np.zeros(16)
    detection_IR = np.zeros(16)

    # Reading data for sensors
    for i in range(1, 17):
        res, sensor_handle = vrep.simxGetObjectHandle(client_id, f"Pioneer_p3dx_ultrasonicSensor{i}", vrep.simx_opmode_blocking)
        sensor_handles[i-1] = sensor_handle
        res, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(client_id, sensor_handle, vrep.simx_opmode_streaming)

    vrep.simxStartSimulation(client_id, vrep.simx_opmode_oneshot_wait)

    
    v0 = 2.0
    noDetectionDist = 0.5
    maxDetectionDist = 0.2
    detect = [0] * 16
    braitenbergL = [-0.2, -0.4, -0.6, -0.8, -1, -1.2, -1.4, -1.6, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    braitenbergR = [-1.6, -1.4, -1.2, -1, -0.8, -0.6, -0.4, -0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


    while True:
        for i in range(16):
            res, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(client_id, int(sensor_handles[i]), vrep.simx_opmode_buffer)
            if detectionState:
                distToObject = math.sqrt(detectedPoint[0] ** 2 + detectedPoint[1] ** 2 + detectedPoint[2] ** 2)
                if distToObject < noDetectionDist:
                    detect[i] = 1 - ((distToObject - maxDetectionDist) / (noDetectionDist - maxDetectionDist))
                    detect[i] = max(detect[i], 0)
                else:
                    detect[i] = 0
            else:
                detect[i] = 0
        
        vLeft = v0
        vRight = v0
        for i in range(16):
            vLeft += braitenbergL[i] * detect[i]
            vRight += braitenbergR[i] * detect[i]

        # Apply motor velocities
        vrep.simxSetJointTargetVelocity(client_id, left_motor, vLeft, vrep.simx_opmode_oneshot)
        vrep.simxSetJointTargetVelocity(client_id, right_motor, vRight, vrep.simx_opmode_oneshot)

    # Terminate
    vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(client_id)

else:
    print('Unable to connect to %s:%s' % (ip, port))