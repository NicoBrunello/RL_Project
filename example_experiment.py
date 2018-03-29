from gym_torcs import TorcsEnv
from sample_agent import Agent
import numpy as np

vision = False
episode_count = 500
max_steps = 100
reward = 0
done = False
step = 2
theta = np.ndarray(shape=(1,8), dtype=(float))


# Generate a Torcs environment
print ("Creating Torcs environment")
env = TorcsEnv(vision=vision, throttle=False)
print("Torcs env created--------------------")
agent = Agent(1)  # steering onlyt_n = np.array(shape=(1,8), dtype=(float))

#Init theta vector

theta=np.random.normal(1, 0.1,(1,8))

print("TORCS Experiment Start.")
for i in range(episode_count):
    print("Episode : " + str(i))

    if np.mod(i, 3) == 0:
        # Sometimes you need to relaunch TORCS because of the memory leak error
        ob = env.reset(relaunch=True)
    else:
        ob = env.reset()


    total_reward = 0.

    for j in range(max_steps):

        action = agent.act(ob, reward, done, vision, theta)
        ob, reward, done, _ = env.step(action)
        print("\n-------------------------------------------------------")
        #print(ob)
        print("\n-------------------------------------------------------")
        total_reward += reward

        step += 1
        if done:
            break

    print("TOTAL REWARD @ " + str(i) +" -th Episode  :  " + str(total_reward))
    print("Total Step: " + str(step))
    print("")

env.end()  # This is for shutting down TORCS
print("Finish.")


#def compute_gradient(a_s_vector, t_n, max_steps )
#pi_theta = np.random.multiariate_normal(t_n, cov)

