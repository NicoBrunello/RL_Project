import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    

    def __init__(self, dim_action):
        self.dim_action = dim_action

    def act(self, ob, reward, done, vision_on, theta):

        cov= np.identity(3) *0.1
        #print("ACT!")

        # Get an Observation from the environment.
        # Each observation vectors are numpy array.
        # focus, opponents, track sensors are scaled into [0, 1]. When the agent
        # is out of the road, sensor variables return -1/200.
        # rpm, wheelSpinVel are raw values and then needed to be preprocessed.
        # vision is given as a tensor with size of (64*64, 3) = (4096, 3) <-- rgb
        # and values are in [0, 255]
        if vision_on is False:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel = ob
        else:
            focus, speedX, speedY, speedZ, opponents, rpm, track, wheelSpinVel, vision = ob
        
        #Preparing the state
        # Scale rpm between [0-1] 
        n_rpm = np.tanh(rpm)
        #Scale the wheelSpinVel vector
        n_wheelSpinVel = np.tanh(wheelSpinVel)

        ob_theta1 = [focus[0], speedX, speedY, speedZ, opponents[0], rpm, track[0], wheelSpinVel[0]]
        ob_theta2 = [focus[0], speedX, speedY, speedZ, opponents[0], rpm, track[0], wheelSpinVel[0]]
        ob_theta3 = [focus[0], speedX, speedY, speedZ, opponents[0], rpm, track[0], wheelSpinVel[0]]
        
        for i in range(self.dim_action):
            theta[i] =  (theta[i]-min(theta[i]))/(max(theta[i])-min(theta[i])) -0.5
        #Get the average action theta * ob
        #todo: transform vectors in values,  now just to try out the code I take the first value vector feature
        av_theta =  [np.dot(theta[0],ob_theta1),
                    np.dot(theta[1],ob_theta2),
                    np.dot(theta[2],ob_theta3)]
        print("av_theta--> "+ str(av_theta))
        #av_theta =np.dot(theta[0,1],focus[0]) + np.dot(theta[0,2],speedX) + np.dot(theta[0,2],speedY) + np.dot(theta[0,3],speedZ) + np.dot(theta[0,4],opponents[0]) + np.dot(theta[0,5],n_rpm) + np.dot(theta[0,6],track[0]) + np.dot(theta[0,7],n_wheelSpinVel[0]) 
        #Normalize av_theta 
        av_theta =  (av_theta-min(av_theta))/(max(av_theta)-min(av_theta)) -0.5
        
        #Sample the action from a gaussian distribution with mean equals to av_theta
        action =np.random.multivariate_normal(av_theta, cov)
        action =  (action-min(action))/(max(action)-min(action)) -0.5
        #return the action
        print(np.tanh(action))     
        return (np.tanh(action), action, av_theta, [ob_theta1, ob_theta2, ob_theta3])



