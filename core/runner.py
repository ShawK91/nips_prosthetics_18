from core import mod_utils as utils
import numpy as np

#Rollout evaluate an agent in a complete game
def rollout_worker(task_q, result_q, env, noise, exp_list, store_transition=True, num_eval =3, skip_step=1):

    while True:
        task = task_q.get()
        id = task[0]; net = task[1]

        fitness = 0.0; traj_len = 0
        for eval in range(num_eval):
            state = env.reset();
            state.append(0)
            state = utils.to_tensor(np.array(state)).unsqueeze(0);
            last_action = None
            for frame in range(1, 1000000000): #Infinite
                if frame % skip_step == 0 or frame == 1:
                    action = net.forward(state)
                    last_action = action
                else: action = last_action

                action = utils.to_numpy(action)
                if noise != None: action += noise.noise()

                next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment
                next_state.append(frame)
                next_state = utils.to_tensor(np.array(next_state)).unsqueeze(0)
                fitness += reward; traj_len+=1

                if store_transition:
                    exp_list.append([utils.to_numpy(state), action,
                                     utils.to_numpy(next_state), np.reshape(np.array([reward]), (1,1)), id, np.reshape(np.array([int(done)]), (1,1)) ])

                state = next_state
                if done: break
                #if frame == 10: break     #TEST

        fitness /= num_eval; traj_len /= num_eval

        result_q.put([id, fitness, traj_len])
        task_q.task_done()



