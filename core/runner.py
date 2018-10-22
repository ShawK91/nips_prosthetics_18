from core.env_wrapper import EnvironmentWrapper
from core import mod_utils as utils
import numpy as np
import core.reward_shaping as rs


#Rollout evaluate an agent in a complete game
def rollout_worker(worker_id, task_pipe, result_pipe, noise, exp_list, pop, difficulty, use_rs, store_transition=True, use_synthetic_targets=False, xbias=None, zbias=None, phase_len = 100):
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
                if difficulty == 0:
                    ltibia.append(env.ltibia_xyz); rtibia.append(env.rtibia_xyz)
                    pelvis_y.append(env.pelvis_y); pelvis_x.append(env.pelvis_x);
                    lfoot.append(env.lfoot_xyz); rfoot.append(env.rfoot_xyz)
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
            if done or (use_synthetic_targets == True and env.istep >= nofault_endstep):
                total_frame += env.istep

                if store_transition:

                    # Forgive trajectories that did not end within 2 steps of maximum allowed
                    if env.istep < 298 and difficulty == 0 or env.istep <998 and difficulty != 0 and use_synthetic_targets != True or env.istep < (nofault_endstep-2) and difficulty != 0 and use_synthetic_targets:
                        for i, entry in enumerate(rollout_trajectory): entry[4] = np.reshape(np.array([len(rollout_trajectory) - i ]), (1, 1))

                    #Push experiences to main
                    for entry in rollout_trajectory: exp_list.append([entry[0], entry[1], entry[2], entry[3], entry[4], entry[5]])
                    rollout_trajectory = []

                #Behavioral Reward Shaping
                if use_rs:
                    if difficulty == 0: #Round 1
                        lfoot = np.array(lfoot); rfoot = np.array(rfoot); ltibia = np.array(ltibia); rtibia = np.array(rtibia); pelvis_y = np.array(pelvis_y); pelvis_x = np.array(pelvis_x); head_x=np.array(head_x)
                        lfemur_angle = np.degrees(np.array(lfemur_angle)); rfemur_angle = np.degrees(np.array(rfemur_angle))
                        ltibia_angle = np.degrees(np.array(ltibia_angle)); rtibia_angle = np.degrees(np.array(rtibia_angle))

                        #Compute Shaped fitness
                        shaped_fitness = env.istep + rs.final_footx(pelvis_x, lfoot, rfoot) * 100.0  #rs.thighs_swing(lfemur_angle, rfemur_angle)/360.0 +

                        #Compute trajectory wide constraints
                        hard_shape_w = rs.pelvis_height_rs(pelvis_y) * rs.foot_z_rs(lfoot, rfoot) * rs.knee_bend(ltibia_angle, lfemur_angle, rtibia_angle, rfemur_angle) * rs.head_behind_pelvis(head_x)

                        #Apply constraint to fitness/shaped_fitness
                        shaped_fitness = shaped_fitness * hard_shape_w if shaped_fitness >0 else shaped_fitness
                        #fitness = fitness *  hard_shape_w if fitness > 0 else fitness

                        #Reset
                        lfoot = []; rfoot = []; ltibia = []; rtibia = []; pelvis_x = []; pelvis_y = []; head_x =[]
                        ltibia_angle = []; lfemur_angle = []; rtibia_angle = []; rfemur_angle = []

                    #ROUND 2
                    else:
                        if env.zminus_pen > 0: zminus_fitness =  0.2 * env.istep - env.zminus_pen
                        else: zminus_fitness = 0.0

                        if env.zplus_pen > 0: zplus_fitness = 0.2 * env.istep - env.zplus_pen
                        else: zplus_fitness = 0.0

                        x_fitness = 0.25 * env.istep - env.x_pen
                        scaled_fit = fitness - env.istep * 9.8 #Shaped fitness reweighs the importance between survival and folllowing the x/z target vel
                        shaped_fitness = [zplus_fitness, zminus_fitness, x_fitness, scaled_fit]
                        fitness = fitness - env.istep * 9.0

                else: shaped_fitness = []


                ############## FOOT Z AXIS PENALTY ##########
                if exit_flag: break
                else:
                    exit_flag = True
                    state = env.reset()
                    state = utils.to_tensor(np.array(state)).unsqueeze(0)

        #Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([worker_id, fitness, total_frame, shaped_fitness])




