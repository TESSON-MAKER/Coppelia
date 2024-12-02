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
scene = './Pioneer.ttt'
position_init = [3,2,to_rad(0)]

noDetectionDist=0.5
maxDetectionDist=0.2

braitenbergL={-0.2,-0.4,-0.6,-0.8,-1,-1.2,-1.4,-1.6, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}
braitenbergR={-1.6,-1.4,-1.2,-1,-0.8,-0.6,-0.4,-0.2, 0.0,0.0,0.0,0.0,0.0,0.0,0.0,0.0}

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
    sensor_handles=np.zeros(16)
    sensors_handles=np.zeros(16)
    detectStatus = np.zeros(16)
    detection_IR = np.zeros(16)
    
    # Reading data for sensors
    for i in range(1,17) : 
        res , sensor_handle = vrep.simxGetObjectHandle(client_id, "Pioneer_p3dx_ultrasonicSensor" + str(i), vrep.simx_opmode_blocking)
        sensor_handles[i-1] = sensor_handle
        res, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(client_id, sensor_handle, vrep.simx_opmode_streaming)
  

    vrep.simxStartSimulation(client_id, vrep.simx_opmode_oneshot_wait)

    position = position_init 

    command =  [0.6,0.6]
    
    continue_running = True
    
    while(continue_running):
    #Ask for stop running
      #  input("Press enter  to stop the simulation")
               
        current_time = vrep.simxGetLastCmdTime(client_id)
            
        vrep.simxSetJointTargetVelocity(client_id, left_motor, command[0], vrep.simx_opmode_oneshot_wait)
        vrep.simxSetJointTargetVelocity(client_id, right_motor, command[1], vrep.simx_opmode_oneshot_wait)
            
        # recuperation position du robot
        res, tmp = vrep.simxGetObjectPosition(client_id, pioneer, -1, vrep.simx_opmode_oneshot_wait)
        position[0] = tmp[0]
        position[1] = tmp[1]
        res, tmp = vrep.simxGetObjectOrientation(client_id, pioneer, -1, vrep.simx_opmode_oneshot_wait)
        position[2] = tmp[2] # en radian 
    
        # recuperation capteurs IR
        for i in range(1,17) : 
            res, detectionState, detectedPoint, detectedObjectHandle, detectedSurfaceNormalVector = vrep.simxReadProximitySensor(client_id, int(sensor_handles[i-1]), vrep.simx_opmode_buffer)
            distToObject = math.sqrt(math.pow(detectedPoint[0], 2) + math.pow(detectedPoint[1], 2) + math.pow(detectedPoint[2], 2)) # Calculate distance to obstacle relative to each sensor 
            detection_IR[i-1]=distToObject
            print(detectedPoint)
            print(distToObject)
            
            
        if (res>0) and (distToObject<noDetectionDist):
            if (distToObject<maxDetectionDist):
                distToObject=maxDetectionDist
                
                
        for i in range(1,17) : 
                    command[0]=command[0]+braitenbergL[i]*detection_IR[i]
                    command[1]=command[1]+braitenbergR[i]*detection_IR[i]


        delta_t = (vrep.simxGetLastCmdTime(client_id)-current_time)/1000
        #print(delta_t, position[0], position[1])
           
    
    #continue_running = False    
    # terminate
    vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(client_id)

else:
    print('Unable to connect to %s:%s' % (ip, port))
