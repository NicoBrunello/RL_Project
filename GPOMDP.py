from gym_torcs import TorcsEnv
from sample_agent import Agent
import numpy as np
import matplotlib as pl

vision = False
episode_count = 5000
max_steps = 50000
reward = 0
done = False
step = 0

#Learning rate
alpha = 0.000000001
#Number of episode to compute the average gradient, this let the variance decreases
avg_episode = 100
#Number of states
n_states= 31
#Number of action
n_action = 3
# Theta represent the policy
theta = np.ndarray(shape=(n_action,n_states), dtype=(float))

def compute_baseline(traj, max_current_steps):
    counter = 0
    single_baseline_n = np.array([0])
    single_baseline_d = np.array([0])
    baseline= np.zeros(max_current_steps)
    for k in range(avg_episode):
        for j in range(max_current_steps):
            if j < traj[k][3]:
                temp_n = traj[k][1]
                temp_d = traj[k][2] 
                single_n = single_baseline_n[j] + temp_n[j]
                single_d = single_baseline_d[j] + temp_d[j]
                single_baseline_n = np.append(single_baseline_n, single_n) 
                single_baseline_d = np.append(single_baseline_d, single_d)
            else: 
                single_baseline_n = np.append(single_baseline_n, [0])
    single_baseline_n = single_baseline_n / avg_episode
    single_baseline_d = single_baseline_d / avg_episode
    for k in range(max_current_steps):
        if single_baseline_n[k] ==0 or single_baseline_d[k]==0:
            baseline[k] = 0
        else: 
            basline[k]= single_baseline_n[k]/ single_baseline_d[k]
    return baseline

def compute_Delta_Theta(a_s_vector):
    delta_theta = np.zeros((n_action,31))
    for i in range(len(a_s_vector)-1):
        action = a_s_vector[i+1][1] - a_s_vector[i+1][2]
        state = a_s_vector[i+1][0]
        delta_theta = delta_theta + (np.outer(action,  state) / (0.1*0.1))
    return delta_theta

def compute_gradient(traj, baseline, max_current_steps):
    gradient = np.zeros((n_action, n_states))
    for i in range(avg_episode):
        single_gradient = np.zeros((n_action,n_states))
        for j in range(max_current_steps):
            if j < traj[i+1][3]:
                single_traj = traj[i+1]
                single_gradient = single_gradient +  single_traj[0][j] * (single_traj[4][j] - baseline[j])   
        gradient = gradient + single_gradient
    return gradient / avg_episode

# Generate a Torcs environment
print ("Creating Torcs environment")
env = TorcsEnv(vision=vision, throttle=False)
print("Torcs env created--------------------")
agent = Agent(3)  # now we use steering only, but we might use throttle and gear

#Init theta vector
theta = np.random.normal(0, 0.01,(n_action,n_states))

performance = np.array([0])

#Baselines sum
baseline_n = 0
baseline_d = 0

#max of steps per trajectory
max_current_steps = 0

#Vector to compute gradient
traj = np.array([[0, 0, 0, 0, 0]])

print("TORCS Experiment Start.")
for i in range(episode_count):
    print("Episode : " + str(i))

    if np.mod(i, 3) == 0:
        # Sometimes you need to relaunch TORCS because of the memory leak error
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()


    total_reward = 0.
    states = np.array([[0, 0, 0]])
    partial_delta_theta = np.array([np.zeros((n_action,n_states))])
    J = 0
    step = 0
    delta_theta = 0
    baseline_n = np.array([np.zeros((n_action,n_states))])
    baseline_d = np.array([np.zeros((n_action,n_states))])
    J_vector = np.array([[0]])

    for step in range(max_steps):

        action, av_theta, ob_theta = agent.act(ob, reward, done, vision, theta)
        ob, reward, done, _ = env.step(action)

        #print("\n-------------------------------------------------------")
        #print(ob)
        #print("\n-------------------------------------------------------")
        total_reward += reward

        ## update the vector of trajectories
        states = np.append(states, [[ob_theta, action, av_theta]], axis=0 )
        ## update performance
        J= J + (0.99**step) * (reward) 

        #Compute delta_theta till the current partial sum
        delta_theta = delta_theta + compute_Delta_Theta(states)

        #Append partial sum of delta_theta to compute gradient later
        partial_delta_theta = np.append(partial_delta_theta, [delta_theta], axis = 0)

        #Compute baseline numerator till the current step
        baseline_n = np.append(baseline_n, [(np.multiply(delta_theta, delta_theta))* J], axis=0 )

        #Compute baseline denominator till current step
        baseline_d = np.append(baseline_d, [(np.multiply(delta_theta, delta_theta))], axis=0 )

        #Append J of current step
        J_vector = np.append(J_vector, [[J]])

        step += 1
        if done:
            break

    performance = np.append(performance, [J])

    if step > max_current_steps : 
        max_current_steps = step 
    
    traj = np.append(traj, [[partial_delta_theta, baseline_n, baseline_d, step, J_vector]], axis=0)


    if i % avg_episode == 0  and i != 0:
        
        #Average baseline
        baseline = compute_baseline(traj, max_current_steps)
        #Compute gradient
        gradient = compute_gradient(traj, baseline, max_current_steps)
        
        #Update policy
        theta= np.sum([theta, alpha*gradient], axis=0)
        
        print("Gradient----_>" + str(gradient))
        traj = np.array([[0, 0, 0, 0, 0]])          
        max_current_steps = 0

    #print(str(J))
    print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    print("----------------------------------------   J = " + str(J) + " L = "+ str(performance.size))
    print("")

pl.pyplot.plot(range(episode_count + 1) ,performance)
pl.pyplot.show()

env.end()  # This is for shutting down TORCS
print("Finish.")


