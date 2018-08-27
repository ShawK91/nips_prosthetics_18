import opensim as osim
from osim.env import ProstheticsEnv
from core import mod_utils as utils
import numpy as np, sys, os

DONE_GAMMA = 0.98

#Rollout evaluate an agent in a complete game
def rollout_worker(worker_id, task_pipe, result_pipe, noise, exp_list, store_transition=True, num_eval =1, skip_step=1):
    worker_id = worker_id
    while True:
        task = task_pipe.recv()
        net_id = task[0]; net = task[1]

        blockPrint()
        env = ProstheticsEnv(visualize=False); exit_flag = False

        fitness = 0.0; cumulative_frame=0; frame = 0
        state = env.reset(); rollout_trajectory = []
        state = utils.to_tensor(np.array(state)).unsqueeze(0);
        while True: #Infinite

            action = net.forward(state)
            action = utils.to_numpy(action)
            if noise != None: action += noise.noise()

            next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment
            #next_state.append(frame)
            next_state = utils.to_tensor(np.array(next_state)).unsqueeze(0)
            fitness += reward; cumulative_frame+=1; frame+=1

            if store_transition:
                rollout_trajectory.append([utils.to_numpy(state), action,
                                 utils.to_numpy(next_state), np.reshape(np.array([reward]), (1,1)),
                                 0.0, np.reshape(np.array([int(done)]), (1,1)) ])
            state = next_state
            if done:
                # Process compute done-probs
                if frame < 299:  # Forgive trajectories longer than 299
                    for i, entry in enumerate(rollout_trajectory): entry[4] = np.reshape(np.array([DONE_GAMMA ** (frame - i)]), (1, 1))
                    for entry in rollout_trajectory: exp_list.append([entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]])         #Send back to main through exp_list
                    rollout_trajectory = []

                if exit_flag: break
                else:
                    exit_flag = True
                    env.reset()


        fitness /= 2.0; cumulative_frame /= 2.0
        result_pipe.send([net_id, fitness, cumulative_frame])



# Disable
def blockPrint():
    sys.stdout = open(os.devnull, 'w')

# Restore
def enablePrint():
    sys.stdout = sys.__stdout__
