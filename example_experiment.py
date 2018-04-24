from gym_torcs import TorcsEnv
from sample_agent import Agent
import numpy as np
import matplotlib as pl

vision = False
episode_count = 500
max_steps = 10000
reward = 0
done = False
step = 0
# Theta represent the policy
theta = np.ndarray(shape=(3,68), dtype=(float))
#Learning rate
alpha=0.00000001
#Number of episode to compute the average gradient, this let the variance decreases
avg_episode = 10

def compute_gradient(traj, baseline ):
    gradient = np.zeros((3,68))
    for k in range(avg_episode):
        gradient = gradient + ( traj[k][0] *(traj[k][1]-(baseline)))
    gradient = gradient / avg_episode
    return gradient

def compute_Baseline(a_s_vector, av_theta, J):
    delta_theta =np.zeros((3,68))
    for i in range(len(a_s_vector)-1):
        action=a_s_vector[i+1][1] - a_s_vector[i+1][3]
        state = a_s_vector[i+1][0]
        delta_theta = delta_theta + (np.outer(action,  state) / (0.1*0.1))
    baseline_num= ((delta_theta**2) *J)
    baseline_den= (delta_theta**2)
    return baseline_num, baseline_den, delta_theta

# Generate a Torcs environment
print ("Creating Torcs environment")
env = TorcsEnv(vision=vision, throttle=False)
print("Torcs env created--------------------")
agent = Agent(3)  # now we use steering only, but we might use throttle and gear

#Init theta vector
theta=np.random.normal(0, 0.01,(3,68))

performance = np.array([0])

#Baselines sum
baseline_n =0
baseline_d =0

#Vector to compute gradient
traj=np.array([[0,0]])

print("TORCS Experiment Start.")
for i in range(episode_count):
    print("Episode : " + str(i))

    if np.mod(i, 3) == 0:
        # Sometimes you need to relaunch TORCS because of the memory leak error
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()


    total_reward = 0.
    states = np.array([[0,0,0,0]])
    J= 0
    step =0
    delta_theta=0

    for j in range(max_steps):

        action, av_theta, ob_theta = agent.act(ob, reward, done, vision, theta)
        ob, reward, done, _ = env.step(action)

        #print("\n-------------------------------------------------------")
        #print(ob)
        #print("\n-------------------------------------------------------")
        total_reward += reward

        ## update the vector of trajectories
        states = np.append(states, [[ob_theta, action, reward, av_theta]], axis=0 )
        ## update performance
        J= J + (0.99**j) * (reward) 
        step += 1
        if done:
            break

    performance = np.append(performance, [J])
    
    
    #Compute Baseline for current episode, delta_theta is the component for compute gradient with the average baseline
    #Component il delta_theta*J, do so just to avoid passing J function again
    baseline_num, baseline_den, delta_theta = compute_Baseline(states, av_theta, J)
    
    #Sum Baselines to get the average
    baseline_n= baseline_n + baseline_num
    baseline_d= baseline_d + baseline_den

    #Append to compute gradient later
    traj=np.append(traj,[[delta_theta, J]], axis=0)
    #Append performance

    if i%avg_episode ==0  and i!=0:
        #Average baseline
        n_avg = baseline_num/avg_episode
        d_avg= baseline_den/ avg_episode
        baseline= n_avg/d_avg
        
        gradient = compute_gradient(traj, baseline)
        theta = theta + alpha*gradient
        
        print("Gradient----_>" + str(gradient))
        traj=np.array([[0,0]])

    #print(str(J))
    print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    print("----------------------------------------   J = " + str(J) + " L = "+ str(performance.size))
    print("")

pl.pyplot.plot(range(episode_count + 1) ,performance)
pl.pyplot.show()

env.end()  # This is for shutting down TORCS
print("Finish.")


