from core.env_wrapper import EnvironmentWrapper
from core import mod_utils as utils
import numpy as np
import core.reward_shaping as rs


#Rollout evaluate an agent in a complete game
def rollout_worker(worker_id, task_pipe, result_pipe, noise, exp_list, pop, difficulty, use_rs, store_transition=True, use_synthetic_targets=False, xbias=None, zbias=None, phase_len=100, traj_container=None, ep_len=1005, JGS=False, num_trials=5):
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
    nofault_endstep = phase_len * 4
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

        fitnesses = []; total_frames = 0
        for _ in range(num_trials):

            fitness = 0.0
            state = env.reset()
            state = utils.to_tensor(np.array(state)).unsqueeze(0)
            while True: #unless done

                action = net.forward(state)
                action = utils.to_numpy(action)
                if noise != None: action += noise.noise()

                next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment



                next_state = utils.to_tensor(np.array(next_state)).unsqueeze(0)
                fitness += reward


                state = next_state

                #DONE FLAG IS Received
                if done:
                    total_frames += env.istep
                    fitnesses.append(fitness)
                    break




        lower_fit =  min(fitnesses); total_frames /= float(num_trials); mean_fit = sum(fitnesses)/float(num_trials); max_fit = max(fitnesses)

        #Send back id, fitness, total length and shaped fitness using the result pipe
        result_pipe.send([worker_id, lower_fit, total_frames, [mean_fit, max_fit]])




