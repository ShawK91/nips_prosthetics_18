from core.env_wrapper import EnvironmentWrapper
from core import mod_utils as utils
import numpy as np
import core.reward_shaping as rs


#Rollout evaluate an agent in a complete game
def rollout_worker(worker_id, task_pipe, result_pipe, noise, exp_list, pop, difficulty, use_rs, store_transition=True, use_synthetic_targets=False, xbias=None, zbias=None, phase_len=100):
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

    worker_id = worker_id; env = EnvironmentWrapper(difficulty, rs=use_rs, use_synthetic_targets=use_synthetic_targets, xbias=xbias, zbias=zbias, phase_len=phase_len)
    nofault_endstep = phase_len * 4
    if use_rs:
        if difficulty == 0:
            lfoot = []; rfoot = []; ltibia = []; rtibia = []; pelvis_x = []; pelvis_y = []
            ltibia_angle = []; lfemur_angle = []; rtibia_angle =[]; rfemur_angle = []
            head_x = []

    while True:
        _ = task_pipe.recv() #Wait until a signal is received  to start rollout
        net = pop[worker_id] #Get the current model state from the population

        fitness = 0.0; total_frame=0; shaped_fitness = 0.0
        state = env.reset(); rollout_trajectory = []
        state = utils.to_tensor(np.array(state)).unsqueeze(0); exit_flag = True
        while True: #unless done

            action = net.forward(state)
            action = utils.to_numpy(action)
            if noise != None: action += noise.noise()

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
                rollout_trajectory.append([utils.to_numpy(state), action,
                                 utils.to_numpy(next_state), np.reshape(np.array([reward]), (1,1)),
                                           np.reshape(np.array([-1.0]), (1, 1)), np.reshape(np.array([int(done)]), (1,1))])

            state = next_state

            #DONE FLAG IS Received
            if done or env.istep > 200 or (use_synthetic_targets == True and env.istep >= nofault_endstep):
                total_frame += env.istep


                #Behavioral Reward Shaping
                if use_rs:

                    lfoot = np.array(lfoot);
                    rfoot = np.array(rfoot);
                    foot_z = rs.foot_z_rs(lfoot, rfoot)

                    # Reset
                    lfoot = [];
                    rfoot = []


                    fitness = fitness - env.istep * 9.0

                    ######## Scalarization RS #######
                    if env.zminus_pen > 0: zminus_fitness =  0.2 * env.istep - env.zminus_pen
                    else: zminus_fitness = 0.0

                    if env.zplus_pen > 0: zplus_fitness = 0.2 * env.istep - env.zplus_pen
                    else: zplus_fitness = 0.0

                    x_fitness = 0.25 * env.istep - env.x_pen


                    #Behavioral RS
                    pelvis_swingx = rs.pelvis_swing(np.array(env.vel_traj), use_synthetic_targets, phase_len)
                    pelv_swing_fit = fitness + pelvis_swingx

                    #Make the scaled fitness list
                    shaped_fitness = [zplus_fitness, zminus_fitness, x_fitness, pelv_swing_fit]

                    fitness = fitness * foot_z if fitness > 0 else fitness


                else: shaped_fitness = []


                ############## FOOT Z AXIS PENALTY ##########
                if exit_flag: break
                else:
                    exit_flag = True
                    state = env.reset()
                    state = utils.to_tensor(np.array(state)).unsqueeze(0)

        #Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([worker_id, fitness, total_frame, shaped_fitness])




