from core.env_wrapper import EnvironmentWrapper
from core import mod_utils as utils
import numpy as np
import core.reward_shaping as rs
import torch, random



def process_act(action_prob):
    action = action_prob[:,0:19] > action_prob[:,19:]

    mask = torch.cat((action, 1 - action), 1).float()
    action_prob = action_prob * mask

    return action, action_prob




#Rollout evaluate an agent in a complete game
def rollout_worker(worker_id, task_pipe, result_pipe, noise, exp_list, pop, difficulty, use_rs, store_transition=True, use_synthetic_targets=False, xbias=None, zbias=None, phase_len=100, traj_container=None, ep_len=1005, JGS=False):
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

    worker_id = worker_id; env = EnvironmentWrapper(difficulty, rs=use_rs, use_synthetic_targets=use_synthetic_targets, xbias=xbias, zbias=zbias, phase_len=phase_len, jgs=JGS)

    if use_rs:
        lfoot = [];
        rfoot = [];
        if difficulty == 0:
            ltibia = []; rtibia = []; pelvis_x = []; pelvis_y = []
            ltibia_angle = []; lfemur_angle = []; rtibia_angle =[]; rfemur_angle = []
            head_x = []

    while True:
        _ = task_pipe.recv() #Wait until a signal is received  to start rollout
        net = pop[worker_id] #Get the current model state from the population

        fitness = 0.0; total_frame=0
        state = env.reset(); rollout_trajectory = []
        state = utils.to_tensor(np.array(state)).unsqueeze(0)

        while True: #unless done

            net_out = net.forward(state)

            #Exploration
            if noise != None:
                if random.random() < noise:
                    rand_ind = np.arange(38)
                    np.random.shuffle(rand_ind)
                    net_out = net_out[:,rand_ind]


            action, action_prob = process_act(net_out)
            action = utils.to_numpy(action)





            next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment

            if use_rs: #If using behavioral reward shaping
                lfoot.append(env.lfoot_xyz);
                rfoot.append(env.rfoot_xyz)
                if difficulty == 0:
                    ltibia.append(env.ltibia_xyz); rtibia.append(env.rtibia_xyz)
                    pelvis_y.append(env.pelvis_y); pelvis_x.append(env.pelvis_x);
                    lfemur_angle.append(env.lfemur_angle); ltibia_angle.append(env.ltibia_angle)
                    rfemur_angle.append(env.rfemur_angle); rtibia_angle.append(env.rtibia_angle)
                    head_x.append(env.head_x)


            next_state = utils.to_tensor(np.array(next_state)).unsqueeze(0)
            fitness += reward

            #If storing transitions
            if store_transition:
                rollout_trajectory.append([utils.to_numpy(state), utils.to_numpy(action_prob),
                                 utils.to_numpy(next_state), np.reshape(np.array([reward]), (1,1)),
                                           np.reshape(np.array([-1.0]), (1, 1)), np.reshape(np.array([int(done)]), (1,1))])

            state = next_state

            #DONE FLAG IS Received
            if done or env.istep >= ep_len:
                total_frame += env.istep

                #Proximal Policy Optimization (Process and push as trajectories)
                if traj_container != None:
                    traj_container.append(rollout_trajectory)


                if store_transition:

                    # Forgive trajectories that did not end within 2 steps of maximum allowed
                    if env.istep < ep_len-2:
                        for i, entry in enumerate(rollout_trajectory): entry[4] = np.reshape(np.array([len(rollout_trajectory) - i ]), (1, 1))

                    #Push experiences to main
                    for entry in rollout_trajectory: exp_list.append([entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]])
                    rollout_trajectory = []





                #ROUND 2 JGS
                shaped_fitness = [env.istep + env.xjgs, env.istep + env.zjgs, env.istep -(abs(env.zjgs) * abs(env.xjgs)), env.istep * 10 + env.xjgs + env.zjgs]
                # Foot Criss-cross
                lfoot = np.array(lfoot);
                rfoot = np.array(rfoot);
                criss_cross = rs.foot_z_rs(lfoot, rfoot)
                lfoot = [];
                rfoot = [];
                fitness = criss_cross * fitness


                break

        #Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([worker_id, fitness, total_frame, shaped_fitness])




