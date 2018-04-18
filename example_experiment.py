from gym_torcs import TorcsEnv
from sample_agent import Agent
import numpy as np
import matplotlib as pl

vision = False
episode_count = 1000
max_steps = 10000
reward = 0
done = False
step = 0
# Theta represent the policy
theta = np.ndarray(shape=(8,3), dtype=(float))
#Learning rate
alpha=0.0001

def compute_gradient(a_s_vector, av_theta, J ):
    delta_theta =np.zeros((3,68))
    for i in range(len(a_s_vector)-1):
        action=a_s_vector[i+1][1] - a_s_vector[i+1][3]
        state = a_s_vector[i+1][0]
        delta_theta = delta_theta + (np.outer(action,  state) / (0.1*0.1))
    baseline_num= ((delta_theta**2) *J)
    baseline_den= (delta_theta**2)
    gradient = delta_theta 
    return gradient, baseline_num, baseline_den


# Generate a Torcs environment
print ("Creating Torcs environment")
env = TorcsEnv(vision=vision, throttle=False)
print("Torcs env created--------------------")
agent = Agent(3)  # now we use steering only, but we might use throttle and gear

#Init theta vector
theta=np.random.normal(0, 0.01,(3,68))

performance = np.array([0])

grad_vector= []

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
    gradient =np.zeros((3,68))
    baseline_num_vector =np.zeros(1);
    baseline_den_vector =np.zeros(1);
    av_theta =0

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
    gradient, baseline_num, baseline_den = compute_gradient(states, av_theta, J)
    print("Gradient----_>" + str(gradient))


    if (i %  2) ==0 :
        grad_vector.append([gradient])        
        print("Firstttttttt--->"+str(grad_vector))
        baseline_num_vector = np.append(baseline_num_vector, baseline_num)
        baseline_den_vector = np.append(baseline_den_vector, baseline_den)
        baseline_num_vector = sum(baseline_num_vector)
        baseline_den_vector = sum(baseline_den_vector)
        baseline_num_vector = baseline_num_vector / 2
        baseline_den_vector = baseline_den_vector / 2
        grad= np.sum(grad_vector)
        grad= grad / 2
        print("Before---___>" + str(grad))
        #TODO: which J ??
        grad = grad * (J-(baseline_num_vector/baseline_den_vector))
        #Update policy
        theta = theta + (alpha * grad)
        print("theta-------->" + str(theta))
        grad= np.zeros((3,68))
    else:
        baseline_num_vector = np.append(baseline_num_vector, baseline_num)
        baseline_den_vector = np.append(baseline_den_vector, baseline_den)
        grad_vector.append([gradient])


    #print(str(J))
    print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    print("----------------------------------------   J = " + str(J) + " L = "+ str(performance.size))
    print("")

pl.pyplot.plot(range(episode_count + 1) ,performance)
pl.pyplot.show()

env.end()  # This is for shutting down TORCS
print("Finish.")


