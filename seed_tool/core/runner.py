from core.env_wrapper import EnvironmentWrapper
from core import mod_utils as utils
import numpy as np
import torch


#Rollout evaluate an agent in a complete game
def rollout_worker(worker_id, task_pipe, result_pipe, net_repo):

    env = EnvironmentWrapper(1)
    net = net_repo[0]
    while True:
        seed = task_pipe.recv() #Wait until a signal is received  to start rollout
        env.respawn(seed)

        fitness = 0.0; total_frame = 0
        state = env.reset()
        state = utils.to_tensor(np.array(state)).unsqueeze(0)

        while True: #unless done

            action = net.forward(state)
            action = utils.to_numpy(action)

            next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment


            next_state = utils.to_tensor(np.array(next_state)).unsqueeze(0)
            fitness += reward

            state = next_state

            #DONE FLAG IS Received
            if done:
                total_frame = env.istep
                break

        #Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([worker_id, seed, fitness, total_frame])




