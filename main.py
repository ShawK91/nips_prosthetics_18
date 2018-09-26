import numpy as np, os, time, random, torch, sys
from core.neuroevolution import SSNE
from core.models import Actor
from core import mod_utils as utils
from core.runner import rollout_worker
from core.ounoise import OUNoise
#os.environ["CUDA_VISIBLE_DEVICES"]='3'
from torch.multiprocessing import Process, Pipe, Manager

USE_RS = False
DIFFICULTY = 1

class Buffer():
    """Cyclic Buffer stores experience tuples from the rollouts

        Parameters:
            save_freq (int): Period for saving data to drive
            save_folder (str): Folder to save data to
            capacity (int): Maximum number of experiences to hold in cyclic buffer

        """
    def __init__(self, save_freq, save_folder):
        self.save_freq = save_freq; self.folder = save_folder
        self.s = []; self.ns = []; self.a = []; self.r = []; self.done_dist = []; self.done = []
        self.num_entries = 0

    def push(self, s, ns, a, r, done_dist, done): #f: FITNESS, t: TIMESTEP, done: DONE
        """Add an experience to the buffer

            Parameters:
                s (ndarray): Current State
                ns (ndarray): Next State
                a (ndarray): Action
                r (ndarray): Reward
                done_dist (ndarray): Temporal distance to done (#action steps after which the skselton fell over)
                done (ndarray): Done
                shaped_r (ndarray): Shaped Reward (includes both temporal and behavioral shaping)


            Returns:
                None
        """

        #Append new tuple
        self.s.append(s); self.ns.append(ns); self.a.append(a); self.r.append(r); self.done_dist.append(done_dist); self.done.append(done)
        self.num_entries += 1

    def __len__(self):
        return len(self.s)

    def save(self):
        """Method to save experiences to drive

               Parameters:
                   None

               Returns:
                   None
           """

        tag = str(int(self.num_entries / self.save_freq))
        np.savez_compressed(self.folder + 'clean_buffer_' + tag,
                            state=np.vstack(self.s),
                            next_state=np.vstack(self.ns),
                            action = np.vstack(self.a),
                            reward = np.vstack(self.r),
                            done_dist = np.vstack(self.done_dist),
                            done=np.vstack(self.done))
        print ('MEMORY BUFFER WITH', len(self.s), 'SAMPLES SAVED WITH TAG', tag)
        #Empty buffer
        self.s = []; self.ns = []; self.a = []; self.r = []; self.done_dist = []; self.done = []

class Parameters:
    def __init__(self):
        """Parameter class stores all parameters for policy gradient

        Parameters:
            None

        Returns:
            None
        """

        self.seed = 1991
        self.num_action_rollouts = 0

        #NeuroEvolution stuff
        self.pop_size = 50
        self.elite_fraction = 0.1
        self.crossover_prob = 0.15
        self.mutation_prob = 0.90
        self.extinction_prob = 0.005  # Probability of extinction event
        self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 0  # 1-Gaussian, 2-Laplace, 3-Uniform


        #Save Results
        self.state_dim = 415; self.action_dim = 19 #Hard coded
        if DIFFICULTY == 0: self.save_foldername = 'R_Skeleton/'
        else: self.save_foldername = 'R2_Skeleton/'
        self.metric_save = self.save_foldername + 'metrics/'
        self.model_save = self.save_foldername + 'models/'
        self.rl_models = self.save_foldername + 'rl_models/'
        self.data_folder = self.save_foldername + 'data/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
        if not os.path.exists(self.metric_save): os.makedirs(self.metric_save)
        if not os.path.exists(self.model_save): os.makedirs(self.model_save)
        if not os.path.exists(self.rl_models): os.makedirs(self.rl_models)
        if not os.path.exists(self.data_folder): os.makedirs(self.data_folder)

class ERL_Agent:
    """Main ERL class containing all methods for CERL

        Parameters:
        args (int): Parameter class with all the parameters

    """

    def __init__(self, args):
        self.args = args
        self.evolver = SSNE(self.args)

        #MP TOOLS
        self.manager = Manager()

        #Init population
        self.pop = self.manager.list()
        for _ in range(args.pop_size):
            self.pop.append(Actor(args))
            #self.pop[-1].apply(utils.init_weights)
        self.best_policy = Actor(args)
        #Turn off gradients and put in eval mode
        for actor in self.pop:
            actor = actor.cpu()
            actor.eval()

        self.load_seed(args.model_save, self.pop)

        #Init RL Agent
        self.replay_buffer = Buffer(100000, self.args.data_folder)

        #MP TOOLS
        self.exp_list = self.manager.list()
        self.evo_task_pipes = [Pipe() for _ in range(args.pop_size)]
        self.evo_result_pipes = [Pipe() for _ in range(args.pop_size)]

        self.evo_workers = [Process(target=rollout_worker, args=(i, self.evo_task_pipes[i][1], self.evo_result_pipes[i][1], None, self.exp_list, self.pop, DIFFICULTY, USE_RS, True)) for i in range(args.pop_size)]

        for worker in self.evo_workers: worker.start()

        #Trackers
        self.buffer_added = 0; self.best_score = 0.0; self.frames_seen = 0.0; self.best_shaped_score = 0.0
        self.eval_flag = [True for _ in range(args.pop_size)]

    def load_seed(self, dir, pop):
        """Read models from drive and sync it into the population

            Parameters:
                  dir (str): Folder location to pull models from
                  pop (shared_list): population of models

            Returns:
                None


        """
        list_files = os.listdir(dir)
        print(list_files)
        for i, model in enumerate(list_files):
            try:
                pop[i].load_state_dict(torch.load(dir + model))
                pop[i].eval()
            except:
                print(model, 'Failed to load')

    def add_experience(self, state, action, next_state, reward, done_probs, done):
        """Process and send experiences to be added to the buffer

              Parameters:
                  state (ndarray): Current State
                  next_state (ndarray): Next State
                  action (ndarray): Action
                  reward (ndarray): Reward
                  done_dist (ndarray): Temporal distance to done (#action steps after which the skselton fell over)
                  done (ndarray): Done

              Returns:
                  None
          """

        self.buffer_added += 1
        self.replay_buffer.push(state, next_state, action, reward, done_probs, done)
        if self.buffer_added % 100000 == 0: self.replay_buffer.save()

    def train(self, gen):
        """Main training loop to do rollouts, neureoevolution, and policy gradients

            Parameters:
                gen (int): Current epoch of training

            Returns:
                None
        """


        ################ ROLLOUTS ##############
        #Start Evo rollouts
        for id, actor in enumerate(self.pop):
            if self.eval_flag[id]:
                self.evo_task_pipes[id][0].send(True)
                self.eval_flag[id] = False


        ########## SOFT -JOIN ROLLOUTS ############
        all_fitness = []; all_net_ids = []; all_eplens = []; all_shaped_fitness = []
        while True:
            for i in range(self.args.pop_size):
                if self.evo_result_pipes[i][0].poll():
                    entry = self.evo_result_pipes[i][0].recv()
                    all_fitness.append(entry[1]); all_net_ids.append(entry[0]); all_eplens.append(entry[2]); self.frames_seen+= entry[2]; all_shaped_fitness.append(entry[3])
                    self.eval_flag[i] = True

            # Soft-join (50%)
            if len(all_fitness) / self.args.pop_size >= 0.7: break


        # Add ALL EXPERIENCE COLLECTED TO MEMORY concurrently
        for _ in range(len(self.exp_list)):
            exp = self.exp_list.pop()
            self.add_experience(exp[0], exp[1], exp[2], exp[3], exp[4], exp[5])
        ######################### END OF PARALLEL ROLLOUTS ################


        ############ PROCESS MAX FITNESS #############
        champ_index = all_net_ids[all_fitness.index(max(all_fitness))]
        if max(all_fitness) > self.best_score:
            self.best_score = max(all_fitness)
            utils.hard_update(self.best_policy, self.pop[champ_index])
            torch.save(self.pop[champ_index].state_dict(), self.args.save_foldername + 'models/' + 'erl_best')
            print("Best policy saved with score", '%.2f'%max(all_fitness))


        #Save champion periodically
        if gen % 5 == 0 and max(all_fitness) > (self.best_score-100):
            torch.save(self.pop[champ_index].state_dict(), self.args.save_foldername + 'models/' + 'champ')
            print("Champ saved with score ", '%.2f'%max(all_fitness))

        if USE_RS:
            max_shaped_fit = max(all_shaped_fitness)
            if max_shaped_fit > self.best_shaped_score:
                self.best_shaped_score = max_shaped_fit
                shaped_champ_ind = all_net_ids[all_shaped_fitness.index(max(all_shaped_fitness))]
                torch.save(self.pop[shaped_champ_ind].state_dict(), self.args.save_foldername + 'models/' + 'shaped_erl_best')
                print("Best Shaped ERL policy saved with true score", '%.2f' % all_fitness[all_shaped_fitness.index(max(all_shaped_fitness))], 'and shaped score of ', '%.2f' % max_shaped_fit)

        else:
            max_shaped_fit = max(all_shaped_fitness)

        #NeuroEvolution's probabilistic selection and recombination step
        self.evolver.epoch(self.pop, all_net_ids, all_fitness, all_shaped_fitness)

        # Synch RL Agent to NE periodically
        if gen % 5 == 0:
            self.evolver.sync_rl(self.args.rl_models, self.pop)

        return max(all_fitness), all_eplens[all_fitness.index(max(all_fitness))], all_fitness, all_eplens, all_shaped_fitness, max_shaped_fit, all_eplens[all_shaped_fitness.index(max(all_shaped_fitness))]

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    frame_tracker = utils.Tracker(parameters.metric_save, ['erl'], '.csv')  #Tracker class to log progress

    #Set seeds
    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)

    #INITIALIZE THE MAIN AGENT CLASS
    agent = ERL_Agent(parameters) #Initialize the agent
    print('Running osim-rl',  ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim, 'using ERL for ', 'Round 1' if DIFFICULTY == 0 else 'Round 2')

    time_start = time.time()
    for gen in range(1, 1000000000): #Infinite generations
        gen_time = time.time()

        #Train one iteration
        best_score, test_len, all_fitness, all_eplen, all_shaped_fit, shaped_score, shaped_len = agent.train(gen)

        #PRINT PROGRESS
        print('Score:','%.2f'%best_score, ' Avg:','%.2f'%frame_tracker.all_tracker[0][1],'Time:','%.2f'%(time.time()-gen_time),
              'Champ_len', '%.2f'%test_len, 'Best_yet', '%.2f'%agent.best_score, 'Shaped_Score', '%.2f'%shaped_score, 'Shaped_len', '%.2f'%shaped_len)

        # PRINT MORE DETAILED STATS PERIODICALLY
        if gen % 5 == 0:
            tmp_fit = np.array(all_fitness); tmp_len = np.array(all_eplen)
            fit_min, fit_mean, fit_std = np.min(tmp_fit), np.mean(tmp_fit), np.std(tmp_fit)
            len_min, len_mean, len_std, len_max = np.min(tmp_len), np.mean(tmp_len), np.std(tmp_len), np.max(tmp_len)
            print()
            print('#Frames Seen/Buffer', int(agent.frames_seen/1000), int(agent.buffer_added/1000), 'Pop Stats: Fitness min/mu/std', '%.2f'%fit_min, '%.2f'%fit_mean, '%.2f'%fit_std, 'Len min/max/mu/std', '%.2f'%len_min, '%.2f'%len_max, '%.2f'%len_mean, '%.2f'%len_std)
            ind_sortmax = sorted(range(len(all_fitness)), key=all_fitness.__getitem__); ind_sortmax.reverse()
            print ('Fitnesses: ', ['%.2f'%all_fitness[i] for i in ind_sortmax])
            print ('Lens:', ['%.1f'%all_eplen[i] for i in ind_sortmax])
            print('Shaped:', ['%.2f' % all_shaped_fit[i] for i in ind_sortmax])
            print()

        #Update score to trackers
        frame_tracker.update([best_score], agent.buffer_added)













