import numpy as np, os, time, random, torch, sys
from core.neuroevolution import SSNE
from core.models import Actor
from core import mod_utils as utils
from core.runner import rollout_worker
from core.ounoise import OUNoise
os.environ["CUDA_VISIBLE_DEVICES"]='3'
TOTAL_FRAMES = 1000000 * 10
from osim.env import ProstheticsEnv
from multiprocessing import Process, JoinableQueue, Manager

SEED = 'R_Skeleton/models/best_policy'

class Buffer():

    def __init__(self, capacity, save_folder):
        self.capacity = capacity; self.folder = save_folder
        self.s = []; self.ns = []; self.a = []; self.r = []; self.f = []; self.done = []
        self.position = 0; self.num_entries = 0

    def push(self, s, ns, a, r, f, done): #f: FITNESS, t: TIMESTEP, done: DONE
        """Saves a transition."""
        if len(self.s) < self.capacity: #Expand if memory not full
            self.s.append(None); self.ns.append(None); self.a.append(None); self.r.append(None); self.f.append(None); self.done.append(None)

        #Append new tuple
        self.s[self.position] = s; self.ns[self.position] = ns; self.a[self.position] = a; self.r[self.position] = r; self.f[self.position] = f; self.done[self.position] = done
        self.position = (self.position + 1) % self.capacity #Update position pointer
        self.num_entries += 1

    def __len__(self):
        return len(self.s)


    def save(self):
        tag = str(int(self.num_entries / self.capacity))
        np.savez_compressed(self.folder + 'buffer_' + tag,
                            state=np.vstack(self.s[0:self.position+1]),
                            next_state=np.vstack(self.ns[0:self.position+1]),
                            action = np.vstack(self.a[0:self.position+1]),
                            reward = np.vstack(self.r[0:self.position+1]),
                            fitness = np.vstack(self.f[0:self.position+1]),
                            done=np.vstack(self.done[0:self.position+1]))
        print ('MEMORY BUFFER WITH', len(self.s), 'SAMPLES SAVED WITH TAG', tag)

class Parameters:
    def __init__(self):

        #RL params
        self.gamma = 0.99; self.tau = 0.001
        self.seed = 4
        self.batch_size = 512
        self.buffer_size = 1000000
        self.frac_frames_train = 1.0

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
        self.state_dim = 159; self.action_dim = 19 #Simply instantiate them here, will be initialized later
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
        for actor in self.pop: actor.eval()
        if SEED != None: self.pop[0].load_state_dict(torch.load(SEED))

        #Init RL Agent
        self.replay_buffer = Buffer(args.buffer_size, self.args.data_folder)
        self.noise_gens = [OUNoise(args.action_dim), None] #First generator is standard while the second has no noise
        for i in range(3): self.noise_gens.append(OUNoise(args.action_dim, scale=random.random()/2, mu = 0.5*(random.random()-0.5), theta=random.random()/2, sigma=random.random())) #Other generators are non-standard and spawn with random params

        #MP TOOLS
        self.manager = Manager()

        self.exp_list = self.manager.list()
        self.evo_task_q = JoinableQueue(); self.evo_res_q = JoinableQueue()
        self.rl_task_q = JoinableQueue(); self.rl_res_q = JoinableQueue()

        self.all_envs = [ProstheticsEnv(visualize=False) for i in range(args.pop_size+5)]
        self.rl_workers = [Process(target=rollout_worker, args=(self.rl_task_q, self.rl_res_q, self.all_envs[i], self.noise_gens[i], self.exp_list, True)) for i in range(5)]
        self.evo_workers = [Process(target=rollout_worker, args=(self.evo_task_q, self.evo_res_q, self.all_envs[5+i], None, self.exp_list, True)) for i in range(args.pop_size)]

        for worker in self.rl_workers: worker.start()
        for worker in self.evo_workers: worker.start()

        #Trackers
        self.num_games = 0; self.num_frames = 0
        self.buffer_added = 0
        self.best_score = 0.0

    def add_experience(self, state, action, next_state, reward, fitness, done):
        self.buffer_added += 1
        self.replay_buffer.push(state, next_state, action, reward, fitness, done)
        if self.buffer_added % 100000 == 0: self.replay_buffer.save()

    def train(self, gen):
        gen_frames = 0.0
        ################ ROLLOUTS ##############
        #Start Evo rollouts
        for id, actor in enumerate(self.pop): self.evo_task_q.put([id, actor])

        # Start RL rollouts
        for _ in range(len(self.rl_workers)): self.rl_task_q.put([-1, self.best_policy])

        # Join and process RL ROLLOUTS
        rl_score = []
        self.rl_task_q.join()
        while not self.rl_res_q.empty():
            entry = self.rl_res_q.get()
            rl_score.append(entry[1])
            gen_frames += entry[2]

        #Join and process Evo Rollouts
        self.evo_task_q.join()
        all_fitness = [0.0 for _ in range(len(self.pop))]; all_eplen = [0.0 for _ in range(len(self.pop))]
        while not self.evo_res_q.empty():
            entry = self.evo_res_q.get()
            all_fitness[entry[0]] = entry[1]
            all_eplen[entry[0]] = entry[2]
            gen_frames += entry[2]

        #Add ALL EXPERIENCE COLLECTED TO MEMORY

        for _ in range(len(self.exp_list)):
            exp = self.exp_list.pop()
            fit = all_fitness[int(exp[4])]
            fit = np.reshape(np.array([fit]), (1,1))
            self.add_experience(exp[0], exp[1], exp[2], exp[3], fit, exp[5])


        ######################### END OF PARALLEL ROLLOUTS ################

        champ_index = all_fitness.index(max(all_fitness))
        if max(all_fitness) > self.best_score:
            self.best_score = max(all_fitness)
            utils.hard_update(self.best_policy, self.pop[champ_index])

        #Save champion periodically
        if gen % 10 == 0:
            torch.save(agent.pop[champ_index].state_dict(), parameters.save_foldername + 'models/' + 'champ')
            torch.save(agent.best_policy.state_dict(), parameters.save_foldername + 'models/' + 'erl_best')
            print("Progress Saved")

        #NeuroEvolution's probabilistic selection and recombination step
        self.evolver.epoch(self.pop, all_fitness, all_eplen)


        # Synch RL Agent to NE periodically
        if gen % 5 == 0: self.evolver.sync_rl(self.args.rl_models, self.pop)

        tmp_fit = np.array(all_fitness); tmp_len = np.array(all_eplen)
        fit_min, fit_mean, fit_std = np.min(tmp_fit), np.mean(tmp_fit), np.std(tmp_fit)
        len_min, len_mean, len_std, len_max = np.min(tmp_len), np.mean(tmp_len), np.std(tmp_len), np.max(tmp_len)

        return max(all_fitness), gen_frames, max(rl_score), all_eplen[champ_index], [fit_min, fit_mean, fit_std, len_min, len_mean, len_max, len_std]

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    frame_tracker = utils.Tracker(parameters.metric_save, ['erl'], '.csv')  # Initiate tracker

    #Initialize the environment
    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)
    agent = ERL_Agent(parameters) #Initialize the agent
    print('Running osim-rl',  ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim, 'using ERL')

    time_start = time.time(); num_frames = 0.0
    for gen in range(1, 1000000000): #Infinite generations
        gen_time = time.time()
        best_score, gen_frames, rl_score, test_len, pop_stat = agent.train(gen)
        num_frames += gen_frames
        print('#Frames/k', int(num_frames/1000), 'Score:','%.2f'%best_score, ' Avg:','%.2f'%frame_tracker.all_tracker[0][1],'Time:','%.2f'%(time.time()-gen_time),
              'Champ_len', '%.2f'%test_len, 'Best_yet', '%.2f'%agent.best_score)
        if gen % 5 == 0:
            print()
            print('Pop Stats: Fitness min/mu/std', '%.2f'%pop_stat[0], '%.2f'%pop_stat[1], '%.2f'%pop_stat[2], 'Len min/max/mu/std', '%.2f'%pop_stat[3], '%.2f'%pop_stat[6], '%.2f'%pop_stat[4], '%.2f'%pop_stat[5], 'Rl_ESD:',
              '%.2f'%(agent.evolver.rl_res['elites']/agent.evolver.num_rl_syncs), '%.2f'%(agent.evolver.rl_res['selects']/agent.evolver.num_rl_syncs), '%.2f'%(agent.evolver.rl_res['discarded']/agent.evolver.num_rl_syncs), )
            print()
        frame_tracker.update([best_score], num_frames)
        if num_frames > TOTAL_FRAMES: break












