import numpy as np, os, time, random, torch, sys
from core.neuroevolution import SSNE
from core.models import Actor
from core import mod_utils as utils
from core.runner import rollout_worker
from core.ounoise import OUNoise
#os.environ["CUDA_VISIBLE_DEVICES"]='3'
from torch.multiprocessing import Process, Pipe, Manager

#sys.stdout = open('erl_log', 'w')
SEED = ['R_Skeleton/models/erl_best', 'R_Skeleton/models/champ']

class Buffer():

    def __init__(self, save_freq, save_folder):
        self.save_freq = save_freq; self.folder = save_folder
        self.s = []; self.ns = []; self.a = []; self.r = []; self.done_dist = []; self.done = []
        self.num_entries = 0

    def push(self, s, ns, a, r, done_dist, done): #f: FITNESS, t: TIMESTEP, done: DONE
        #Append new tuple
        self.s.append(s); self.ns.append(ns); self.a.append(a); self.r.append(r); self.done_dist.append(done_dist); self.done.append(done)
        self.num_entries += 1

    def __len__(self):
        return len(self.s)

    def save(self):
        tag = str(int(self.num_entries / self.save_freq))
        np.savez_compressed(self.folder + 'buffer_' + tag,
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

        self.seed = 1991
        self.num_action_rollouts = 4

        #NeuroEvolution stuff
        self.pop_size = 40
        self.elite_fraction = 0.1
        self.crossover_prob = 0.2
        self.mutation_prob = 0.85
        self.extinction_prob = 0.005  # Probability of extinction event
        self.extinction_magnituide = 0.5  # Probabilty of extinction for each genome, given an extinction event
        self.weight_magnitude_limit = 10000000
        self.mut_distribution = 0  # 1-Gaussian, 2-Laplace, 3-Uniform


        #Save Results
        self.state_dim = 415; self.action_dim = 19 #Simply instantiate them here, will be initialized later
        self.save_foldername = 'R_Skeleton/'
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
    def __init__(self, args):
        self.args = args
        self.evolver = SSNE(self.args)

        #Init population
        self.pop = []
        for _ in range(args.pop_size):
            self.pop.append(Actor(args))
            #self.pop[-1].apply(utils.init_weights)
        self.best_policy = Actor(args)
        #Turn off gradients and put in eval mode
        for actor in self.pop:
            actor = actor.cpu()
            actor.eval()
        if len(SEED) != 0:
            for i in range(len(SEED)):
                try:
                    self.pop[i].load_state_dict(torch.load(SEED[i]))
                    self.pop[i] = self.pop[i].cpu()
                    print (SEED[i], 'loaded')
                except: 'SEED LOAD FAILED'

        #Init RL Agent
        self.replay_buffer = Buffer(100000, self.args.data_folder)
        self.noise_gens = [OUNoise(args.action_dim), None] #First generator is standard while the second has no noise
        for i in range(3): self.noise_gens.append(OUNoise(args.action_dim, scale=random.random()/8.0,
                                                          mu = 0.0, theta=random.random()/5.0, sigma=random.random()/3.0)) #Other generators are non-standard and spawn with random params

        #MP TOOLS
        self.manager = Manager()
        self.exp_list = self.manager.list()

        self.evo_task_pipes = [Pipe() for _ in range(args.pop_size)]
        self.evo_result_pipes = [Pipe() for _ in range(args.pop_size)]
        self.rl_task_pipes = [Pipe() for _ in range(args.num_action_rollouts)]
        self.rl_result_pipes= [Pipe() for _ in range(args.num_action_rollouts)]

        #self.all_envs = [ProstheticsEnv(visualize=False, integrator_accuracy=INTEGRATOR_ACCURACY) for _ in range(args.pop_size+args.num_action_rollouts)]
        self.evo_workers = [Process(target=rollout_worker, args=(i, self.evo_task_pipes[i][1], self.evo_result_pipes[i][1], None, self.exp_list, 0, True)) for i in range(args.pop_size)]
        self.rl_workers = [Process(target=rollout_worker, args=(args.pop_size+i, self.rl_task_pipes[i][1], self.rl_result_pipes[i][1], self.noise_gens[i], self.exp_list, 0, True)) for i in range(args.num_action_rollouts)]

        for worker in self.rl_workers: worker.start()
        for worker in self.evo_workers: worker.start()

        #Trackers
        self.buffer_added = 0; self.best_score = 0.0; self.frames_seen = 0.0
        self.eval_flag = [True for _ in range(args.pop_size)]
        self.rl_eval_flag = [True]; self.rl_score =[]; self.rl_len = []

    def add_experience(self, state, action, next_state, reward, done_probs, done):
        self.buffer_added += 1
        self.replay_buffer.push(state, next_state, action, reward, done_probs, done)
        if self.buffer_added % 100000 == 0: self.replay_buffer.save()

    def train(self, gen):
        gen_start = time.time(); print()
        ################ ROLLOUTS ##############
        #Start Evo rollouts
        for id, actor in enumerate(self.pop):
            if self.eval_flag[id]:
                self.evo_task_pipes[id][0].send([id, actor])
                self.eval_flag[id] = False

        # # Start RL rollouts
        if self.rl_eval_flag:
            self.eval_flag[id] = False
            self.rl_score = []; self.rl_len = []
            for id in range(self.args.num_action_rollouts): self.rl_task_pipes[id][0].send([id, self.best_policy])

        all_fitness = []; all_net_ids = []; all_eplens = []
        while True:
            for i in range(self.args.pop_size):
                if self.evo_result_pipes[i][0].poll():
                    entry = self.evo_result_pipes[i][0].recv()
                    all_fitness.append(entry[1]); all_net_ids.append(entry[0]); all_eplens.append(entry[2]); self.frames_seen+= entry[2]
                    self.eval_flag[i] = True

            # Soft-join (50%)
            if len(all_fitness) / self.args.pop_size >= 0.7: break


        for i in range(self.args.num_action_rollouts):
            if self.rl_result_pipes[i][0].poll():
                entry = self.rl_result_pipes[i][0].recv()
                self.rl_score.append(entry[1])
                self.rl_len.append(entry[2])
                self.frames_seen += entry[2]
            if len(self.rl_score) == self.args.num_action_rollouts:
                self.rl_eval_flag = True


        # Add ALL EXPERIENCE COLLECTED TO MEMORY concurrently
        for _ in range(len(self.exp_list)):
            exp = self.exp_list.pop()
            self.add_experience(exp[0], exp[1], exp[2], exp[3], exp[4], exp[5])


        ######################### END OF PARALLEL ROLLOUTS ################
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

        #NeuroEvolution's probabilistic selection and recombination step
        self.evolver.epoch(self.pop, all_net_ids, all_fitness, all_eplens)

        # Synch RL Agent to NE periodically
        if gen % 5 == 0:
            rl_sync_time = time.time()
            self.evolver.sync_rl(self.args.rl_models, self.pop)

        return max(all_fitness), all_eplens[all_fitness.index(max(all_fitness))], all_fitness, all_eplens

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    frame_tracker = utils.Tracker(parameters.metric_save, ['erl'], '.csv')  # Initiate tracker

    #Initialize the environment
    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)
    agent = ERL_Agent(parameters) #Initialize the agent
    print('Running osim-rl',  ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim, 'using ERL')

    time_start = time.time()
    for gen in range(1, 1000000000): #Infinite generations
        gen_time = time.time()
        best_score, test_len, all_fitness, all_eplen = agent.train(gen)
        print('#Frames Seen/Buffer', int(agent.frames_seen/1000), int(agent.buffer_added/1000), 'Score:','%.2f'%best_score, ' Avg:','%.2f'%frame_tracker.all_tracker[0][1],'Time:','%.2f'%(time.time()-gen_time),
              'Champ_len', '%.2f'%test_len, 'Best_yet', '%.2f'%agent.best_score)
        if gen % 5 == 0:
            tmp_fit = np.array(all_fitness); tmp_len = np.array(all_eplen)
            fit_min, fit_mean, fit_std = np.min(tmp_fit), np.mean(tmp_fit), np.std(tmp_fit)
            len_min, len_mean, len_std, len_max = np.min(tmp_len), np.mean(tmp_len), np.std(tmp_len), np.max(tmp_len)
            print()
            print('Pop Stats: Fitness min/mu/std', '%.2f'%fit_min, '%.2f'%fit_mean, '%.2f'%fit_std, 'Len min/max/mu/std', '%.2f'%len_min, '%.2f'%len_max, '%.2f'%len_mean, '%.2f'%len_std, 'Rl_ESD:',
              '%.2f'%(agent.evolver.rl_res['elites']/agent.evolver.num_rl_syncs), '%.2f'%(agent.evolver.rl_res['selects']/agent.evolver.num_rl_syncs), '%.2f'%(agent.evolver.rl_res['discarded']/agent.evolver.num_rl_syncs), )
            ind_sortmax = sorted(range(len(all_fitness)), key=all_fitness.__getitem__); ind_sortmax.reverse()
            print ('Fitnesses: ', ['%.1f'%all_fitness[i] for i in ind_sortmax])
            print ('Lens:', ['%.1f'%all_eplen[i] for i in ind_sortmax])
            print ('Action_rollouts_fitnesses', ['%.2f'%i for i in agent.rl_score])
            print ('Action_rollouts_lens', ['%.2f'%i for i in agent.rl_len])
            print()

        frame_tracker.update([best_score], agent.buffer_added)













