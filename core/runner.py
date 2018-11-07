from core.env_wrapper import EnvironmentWrapper
from core import mod_utils as utils
import numpy as np
import core.reward_shaping as rs


#Rollout evaluate an agent in a complete game
def rollout_worker(worker_id, task_pipe, result_pipe, noise, exp_list, pop, num_evals=1):
    """Rollout Worker runs a simulation in the environment to generate experiences and fitness values

        Parameters:
            worker_id (int): Specific Id unique to each worker spun
            task_pipe (pipe): Receiver end of the task pipe used to receive signal to start on a task
            result_pipe (pipe): Sender end of the pipe used to report back results
            noise (object): A noise generator object
            exp_list (shared list object): A shared list object managed by a manager that is used to store experience tuples
            pop (shared list object): A shared list object managed by a manager used to store all the models (actors)
            difficulty (int): Difficulty of the task
            use_rs (bool): Use behavioral reward shaping?
            store_transition (bool): Log experiences to exp_list?

        Returns:
            None
    """

    worker_id = worker_id; env = EnvironmentWrapper(1)


    while True:
        _ = task_pipe.recv() #Wait until a signal is received  to start rollout
        net = pop[worker_id] #Get the current model state from the population

        fitness = 0.0; total_frame = 0
        for _ in range(num_evals):

            state = env.reset()
            state = utils.to_tensor(np.array(state)).unsqueeze(0)
            while True:

                action = net.forward(state)
                action = utils.to_numpy(action)
                if noise != None: action += noise.noise()

                next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment

                next_state = utils.to_tensor(np.array(next_state)).unsqueeze(0)
                fitness += reward

                #If storing transitions
                rollout_trajectory.append([utils.to_numpy(state), action,
                                 utils.to_numpy(next_state), np.reshape(np.array([reward]), (1,1)),
                                           np.reshape(np.array([-1.0]), (1, 1)), np.reshape(np.array([int(done)]), (1,1))])

                state = next_state

                #DONE FLAG IS Received
                if done:
                    total_frame += env.istep

                    # Forgive trajectories that did not end within 2 steps of maximum allowed
                    if env.istep < 998:
                        for i, entry in enumerate(rollout_trajectory): entry[4] = np.reshape(np.array([len(rollout_trajectory) - i ]), (1, 1))

                    #Push experiences to main
                    for entry in rollout_trajectory: exp_list.append([entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]])
                    rollout_trajectory = []


                    ############## FOOT Z AXIS PENALTY ##########
                    break

        fitness /= num_evals
        total_frame /= num_evals

        #Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([worker_id, fitness, total_frame, []])




