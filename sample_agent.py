import numpy as np
import matplotlib.pyplot as plt

class Agent(object):
    

    def __init__(self, dim_action):
        self.dim_action = dim_action

    def act(self, ob, reward, done, vision_on, theta):

        cov= np.identity(3) *0.01
        action= np.zeros(3)
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
        # Normalize rpm between [0-1] 
        n_rpm = np.tanh(rpm)
        
        #Normalize the wheelSpinVel vector
        n_wheelSpinVel = np.tanh(wheelSpinVel)
        
        #Normalize speed
        n_speedX= ((speedX - (-90)) / (180) ) -0.5
        n_speedY= ((speedY - (-90)) / (180) ) -0.5
        n_speedZ= ((speedZ - (-90)) / (180) ) -0.5

        ob_theta = np.asarray(focus)
        ob_theta = np.append(ob_theta, [n_speedX, n_speedY, n_speedZ])
        #ob_theta = np.append(ob_theta, [opponents])
        #ob_theta = np.append(ob_theta, n_rpm)
        ob_theta  = np.append(ob_theta, [track])
        ob_theta = np.append(ob_theta, [n_wheelSpinVel])

        #print("Ob_theta --------------")
        #print(str(ob_theta))
       # print("-------------------------")

        #Get the average action theta * ob
        av_theta =  np.inner(theta, ob_theta)

        #Sample the action from a gaussian distribution with mean equals to av_theta
        action =np.random.multivariate_normal(av_theta, cov)

        #action = np.clip(action,-1,1)
        #return the action
        return (action, av_theta, ob_theta)



