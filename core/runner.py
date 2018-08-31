from core.env_wrapper import EnvironmentWrapper
from core import mod_utils as utils
import numpy as np

#Rollout evaluate an agent in a complete game
def rollout_worker(worker_id, task_pipe, result_pipe, noise, exp_list, difficulty, store_transition=True):
    worker_id = worker_id; env = EnvironmentWrapper(difficulty)
    while True:
        task = task_pipe.recv()
        #print('Task Received for', worker_id)
        net_id = task[0]; net = task[1]
        #env.respawn()
        fitness = 0.0; total_frame=0; current_frame = 0
        state = env.reset(); rollout_trajectory = []
        state = utils.to_tensor(np.array(state)).unsqueeze(0); exit_flag = True
        while True: #Infinite

            action = net.forward(state)
            action = utils.to_numpy(action)
            if noise != None: action += noise.noise()

            next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment
            #next_state.append(frame)
            next_state = utils.to_tensor(np.array(next_state)).unsqueeze(0)
            fitness += reward; total_frame+=1; current_frame+=1

            if store_transition:
                rollout_trajectory.append([utils.to_numpy(state), action,
                                 utils.to_numpy(next_state), np.reshape(np.array([reward]), (1,1)),
                                 -1.0, np.reshape(np.array([int(done)]), (1,1)) ])
            state = next_state
            if done:
                # Process compute done-probs
                if store_transition:
                    if current_frame < 298 and difficulty == 0 or current_frame <998 and difficulty != 0:  # Forgive trajectories that did not end within 2 steps of maximum allowed
                        for i, entry in enumerate(rollout_trajectory): entry[4] = np.reshape(np.array([current_frame - i]), (1, 1))

                    for entry in rollout_trajectory: exp_list.append([entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]])         #Send back to main through exp_list
                    rollout_trajectory = []

                if exit_flag: break
                else:
                    exit_flag = True
                    current_frame = 0
                    state = env.reset()
                    state = utils.to_tensor(np.array(state)).unsqueeze(0)


        fitness /= 1.0; total_frame /= 1.0
        result_pipe.send([net_id, fitness, total_frame])




