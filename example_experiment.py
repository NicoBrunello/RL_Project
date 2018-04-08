from gym_torcs import TorcsEnv
from sample_agent import Agent
import numpy as np
import matplotlib as pl

vision = False
episode_count = 3
max_steps = 100
reward = 0
done = False
step = 2
# Theta represent the policy
theta = np.ndarray(shape=(8,3), dtype=(float))
#Learning rate
alpha=0.0001

def compute_gradient(a_s_vector, av_theta, J ):
    delta_theta =[[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0],[0,0,0,0,0,0,0,0]]
    for i in range(len(a_s_vector)):
        action=a_s_vector[i][1] - av_theta
        state = a_s_vector[i][0]
        delta_theta = delta_theta + (np.multiply(action[0] - av_theta[0],  state)  / (0.1*0.1))
    baseline= ((delta_theta**2) * J)/(delta_theta**2)
    gradient = delta_theta * (J-baseline**8)
    return gradient


# Generate a Torcs environment
print ("Creating Torcs environment")
env = TorcsEnv(vision=vision, throttle=False)
print("Torcs env created--------------------")
agent = Agent(3)  # now we use steering only, but we might use throttle and gear

#Init theta vector

theta=np.random.normal(1, 0.1,(3,8))
performance = np.array([0])

print("TORCS Experiment Start.")
for i in range(episode_count):
    print("Episode : " + str(i))

    if np.mod(i, 3) == 0:
        # Sometimes you need to relaunch TORCS because of the memory leak error
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()

    total_reward = 0.
    states = np.array([[0,0,0]])
    J= 0;

    for j in range(max_steps):

        action, raw_action, av_theta, ob_theta = agent.act(ob, reward, done, vision, theta)
        ob, reward, done, _ = env.step(action)

        #print("\n-------------------------------------------------------")
        #print(ob)
        #print("\n-------------------------------------------------------")
        total_reward += reward
        ## update the vector of trajectories
        states = np.append(states, [[ob_theta, raw_action, reward]], axis=0 )
        ## update performance
        J= J + (0.99**j) * (reward) 
        step += 1
        if done:
            break

    performance = np.append(performance, [J])
    gradient = compute_gradient(states, av_theta, J)
    theta = theta + alpha * gradient
    print("theta ------------->"+str(theta))
    #print(str(J))
    print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    print("----------------------------------------   J = " + str(J) + " L = "+ str(performance.size))
    print("")

pl.pyplot.plot(range(episode_count + 1) ,performance)
pl.pyplot.show()

env.end()  # This is for shutting down TORCS
print("Finish.")


