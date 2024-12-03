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
scene = './Scene Pioneer_murs.ttt'
position_init = [0,0,to_rad(0)]


print ('Program started')
vrep.simxFinish(-1) # just in case, close all opened connections
client_id=vrep.simxStart(ip,port,True,True,5000,5) # Connect to V-REP

if client_id!=-1:
    print ('Connected to remote API server on %s:%s' % (ip, port))
    res = vrep.simxLoadScene(client_id, scene, 1, vrep.simx_opmode_oneshot_wait)
    res, pioneer = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx', vrep.simx_opmode_oneshot_wait)
    res, left_motor = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx_leftMotor', vrep.simx_opmode_oneshot_wait)
    res, right_motor = vrep.simxGetObjectHandle(client_id, 'Pioneer_p3dx_rightMotor', vrep.simx_opmode_oneshot_wait)

    vrep.simxStartSimulation(client_id, vrep.simx_opmode_oneshot_wait)

    position = position_init 

    command =  [0.,0.]
    
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
    
                
        delta_t = (vrep.simxGetLastCmdTime(client_id)-current_time)/1000
        print(delta_t, position[0], position[1])
           
    
    #continue_running = False    
    # terminate
    vrep.simxStopSimulation(client_id, vrep.simx_opmode_oneshot_wait)
    vrep.simxFinish(client_id)

else:
    print('Unable to connect to %s:%s' % (ip, port))
