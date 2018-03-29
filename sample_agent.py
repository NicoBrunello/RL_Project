import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    

    def __init__(self, dim_action):
        self.dim_action = dim_action

    def act(self, ob, reward, done, vision_on, theta):

        cov= np.identity(8) *0.1
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


        #Get the average action theta * ob
        # Values of ob must be prepared
        av_theta =[np.dot(theta[0,1],focus[0]),
                    np.dot(theta[0,2],speedX),
                    np.dot(theta[0,2],speedY),
                    np.dot(theta[0,3],speedZ),
                    np.dot(theta[0,4],opponents[0]),
                    np.dot(theta[0,5],rpm),
                    np.dot(theta[0,6],track[0]),
                    np.dot(theta[0,7],wheelSpinVel[0]) ] 
        #Sample the action from a gaussian distribution with mean value av_theta
        action =np.random.multivariate_normal(av_theta, cov)
        #return the action
        print(np.tanh(action))
        
        return np.tanh(action)
