from core.env_wrapper import EnvironmentWrapper
from core import mod_utils as utils
import numpy as np
from core import reward_shaping as rs_module


#Rollout evaluate an agent in a complete game
def rollout_worker(worker_id, task_pipe, result_pipe, noise, exp_list, difficulty, use_rs, store_transition=True):
    worker_id = worker_id; env = EnvironmentWrapper(difficulty, rs=use_rs)
    lfoot = []; rfoot = []; lknee =[]; rknee = []

    while True:
        task = task_pipe.recv()
        net_id = task[0]; net = task[1]

        fitness = 0.0; total_frame=0; shaped_fitness = 0.0
        state = env.reset(); rollout_trajectory = []
        state = utils.to_tensor(np.array(state)).unsqueeze(0); exit_flag = True
        while True: #Infinite

            action = net.forward(state)
            action = utils.to_numpy(action)
            if noise != None: action += noise.noise()

            next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment
            if use_rs:
                lfoot.append(env.foot_pos[0]); rfoot.append(env.foot_pos[1])
                lknee.append(env.knee_pos[0]); rknee.append(env.knee_pos[1])


            next_state = utils.to_tensor(np.array(next_state)).unsqueeze(0)
            fitness += reward

            if store_transition:
                rollout_trajectory.append([utils.to_numpy(state), action,
                                 utils.to_numpy(next_state), np.reshape(np.array([reward]), (1,1)),
                                 -1.0, np.reshape(np.array([int(done)]), (1,1))])
            state = next_state

            #DONE FLAG IS Received
            if done:
                total_frame += env.istep
                # Process compute done-distances
                if store_transition:

                    # Forgive trajectories that did not end within 2 steps of maximum allowed
                    if env.istep < 298 and difficulty == 0 or env.istep <998 and difficulty != 0:
                        for i, entry in enumerate(rollout_trajectory): entry[4] = np.reshape(np.array([len(rollout_trajectory) - i ]), (1, 1))

                    #Push experiences to main
                    for entry in rollout_trajectory: exp_list.append([entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]])         #Send back to main through exp_list
                    rollout_trajectory = []

                    #Process Reward shaping components and compute shaped_fitness
                    if use_rs:
                        shaped_fitness += rs_module.knee_bend(lknee, lfoot)
                        shaped_fitness += rs_module.knee_bend(rknee, rfoot)
                        lfoot = []; rfoot = []; lknee = []; rknee = []


                if exit_flag: break
                else:
                    exit_flag = True
                    state = env.reset()
                    state = utils.to_tensor(np.array(state)).unsqueeze(0)


        fitness /= 1.0; total_frame /= 1.0; shaped_fitness /= 1.0
        result_pipe.send([net_id, fitness, total_frame, shaped_fitness])




