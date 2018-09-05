import numpy as np, os, time, random, torch, sys
from core.neuroevolution import SSNE
from core.models import Actor

from core.runner import rollout_worker
from core.ounoise import OUNoise
import torch
import numpy as np, os, time, random, sys
from core import mod_utils as utils
from osim.env import ProstheticsEnv
from multiprocessing import Process, JoinableQueue
from core import off_policy_gradient as pg
from core import models
os.environ["CUDA_VISIBLE_DEVICES"]='3'
TOTAL_FRAMES = 1000000 * 10
from osim.env import ProstheticsEnv
from multiprocessing import Process, Pipe, Manager

SEED = True
FORCE_SEED = True
FORCE_SEED_FNAME = 'rl_models/td3_best'
algo = 'DDPG'    #1. TD3
                 #2. DDPG
                 #3. TD3_max   - TD3 but using max rather than min
DATA_LIMIT = 5000000
#FAIRLY STATIC
SAVE = True
QUICK_TEST = False
ITER_PER_EPOCH = 100
SAVE_THRESHOLD = 500
TEST_EVALS = 2
INTEGRATOR_ACCURACY=5e-5*2
if QUICK_TEST:
    ITER_PER_EPOCH = 1
    SEED = False
    DATA_LIMIT = 1000

#Algo Choice process
if True:
    if algo == 'TD3':
        critic_fname = 'td3_critic'
        actor_fname = 'td3_actor'
        log_fname = 'td3_epoch'
        best_fname = 'td3_best'
    elif algo == 'DDPG':
        critic_fname = 'ddpg_critic'
        actor_fname = 'ddpg_actor'
        log_fname = 'ddpg_epoch'
        best_fname = 'ddpg_best'
    elif algo == 'TD3_max':
        critic_fname = 'td3_max_critic'
        actor_fname = 'td3_max_actor'
        log_fname = 'td3_max_epoch'
        best_fname = 'td3_max_best'
    else:
        sys.exit('Incorrect Algo choice')

class Buffer():

    def __init__(self, capacity, save_folder):
        self.capacity = capacity; self.folder = save_folder
        self.s = []; self.ns = []; self.a = []; self.r = []; self.done_probs = []; self.done = []
        self.num_entries = 0

    def push(self, s, ns, a, r, done_probs, done): #f: FITNESS, t: TIMESTEP, done: DONE
        #Append new tuple
        self.s.append(s); self.ns.append(ns); self.a.append(a); self.r.append(r); self.done_probs.append(done_probs); self.done.append(done)
        if len(self.s) > self.capacity:
            self.s.pop(0); self.ns.pop(0); self.a.pop(0); self.r.pop(0); self.done_probs.pop(0); self.done.pop(0)
        self.num_entries += 1

    def __len__(self):
        return len(self.s)

    def save(self):
        tag = str(int(self.num_entries / self.capacity))
        save_ind = self.num_entries % self.capacity
        np.savez_compressed(self.folder + 'pg_buffer_' + tag,
                            state=np.vstack(self.s)[-save_ind:],
                            next_state=np.vstack(self.ns)[-save_ind:],
                            action = np.vstack(self.a)[-save_ind:],
                            reward = np.vstack(self.r)[-save_ind:],
                            fitness = np.vstack(self.done_probs)[-save_ind:],
                            done=np.vstack(self.done)[-save_ind:])
        print ('MEMORY BUFFER WITH', len(self.s), 'SAMPLES SAVED WITH TAG', tag)

    def sample(self, batch_size):
        ind = random.sample(range(self.__len__()), batch_size)
        return torch.Tensor(np.vstack([self.s[i] for i in ind])), torch.Tensor(np.vstack([self.ns[i] for i in ind])), torch.Tensor(np.vstack([self.a[i] for i in ind])), torch.Tensor(np.vstack([self.r[i] for i in ind]))

class Parameters:
    def __init__(self):

        #MetaParams
        self.is_cuda= True
        self.algo = algo
        self.num_action_rollouts = 3

        #RL params
        self.seed = 949
        self.batch_size = 256
        self.action_loss = False
        self.action_loss_w = 1.0

        #TD3
        self.gamma = 0.995; self.tau = 0.001
        self.init_w = False
        self.policy_ups_freq = 2
        self.policy_noise = 0.03
        self.policy_noise_clip = 0.1

        #TR Constraints
        self.critic_constraint = False
        self.critic_constraint_w = None
        self.q_clamp = None
        self.policy_constraint = None
        self.policy_constraint_w = 5.0

        #Target Networks
        self.hard_update_freq = 10000
        self.hard_update = False

        #Save Results
        self.state_dim = 158; self.action_dim = 19 #Simply instantiate them here, will be initialized later
        self.save_foldername = 'R_Skeleton/'
        self.metric_save = self.save_foldername + 'metrics/'
        self.model_save = self.save_foldername + 'models/'
        self.rl_models = self.save_foldername + 'rl_models/'
        self.data_folder = self.save_foldername + 'data_old/'
        self.aux_save = self.save_foldername + 'auxiliary/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
        if not os.path.exists(self.metric_save): os.makedirs(self.metric_save)
        if not os.path.exists(self.model_save): os.makedirs(self.model_save)
        if not os.path.exists(self.rl_models): os.makedirs(self.rl_models)
        if not os.path.exists(self.data_folder): os.makedirs(self.data_folder)
        if not os.path.exists(self.aux_save): os.makedirs(self.aux_save)

class FULL_PG:
    def __init__(self, args):
        self.args = args
        self.rl_agent = pg.TD3_DDPG(args)

        if SEED:
            try:
                if FORCE_SEED:
                    self.rl_agent.actor.load_state_dict(torch.load('R_Skeleton/' + FORCE_SEED_FNAME))
                    self.rl_agent.actor_target.load_state_dict(torch.load('R_Skeleton/' + FORCE_SEED_FNAME))
                else:
                    self.rl_agent.actor.load_state_dict(torch.load(args.rl_models + best_fname))
                    self.rl_agent.actor_target.load_state_dict(torch.load(args.rl_models + best_fname))

                print('Actor successfully loaded from the R_Skeleton folder for', best_fname)
            except: print('Loading Actors failed')
            try:
                self.rl_agent.critic_target.load_state_dict(torch.load(args.model_save + critic_fname))
                self.rl_agent.critic.load_state_dict(torch.load(args.model_save + critic_fname))
                print('Critic successfully loaded from the R_Skeleton folder for', critic_fname)
            except: print ('Loading Critics failed')

        self.rl_agent.actor_target.cuda(); self.rl_agent.critic_target.cuda(); self.rl_agent.critic.cuda(); self.rl_agent.actor.cuda()


        ############ LOAD DATA AND CONSTRUCT GENERATORS ########
        self.replay_buffer = Buffer(10000000, self.args.data_folder)
        self.noise_gens = [OUNoise(args.action_dim), None] #First generator is standard while the second has no noise
        for i in range(3): self.noise_gens.append(OUNoise(args.action_dim, scale=random.random()/10, mu = 0, theta=random.random()/5.0, sigma=random.random()/4.0)) #Other generators are non-standard and spawn with random params


        #MP TOOLS
        self.manager = Manager()
        self.exp_list = self.manager.list()

        self.rl_task_pipes = [Pipe() for _ in range(args.num_action_rollouts)]
        self.rl_result_pipes= [Pipe() for _ in range(args.num_action_rollouts)]

        self.all_envs = [ProstheticsEnv(visualize=False, integrator_accuracy=INTEGRATOR_ACCURACY) for _ in range(args.num_action_rollouts)]
        self.rl_workers = [Process(target=rollout_worker, args=(i, self.rl_task_pipes[i][1], self.rl_result_pipes[i][1], self.all_envs[i], self.noise_gens[i], self.exp_list, True)) for i in range(args.num_action_rollouts)]

        for worker in self.rl_workers: worker.start()

        self.rollout_policies = [models.Actor(args) for _ in range(self.args.num_action_rollouts)]
        self.best_policy = models.Actor(args); self.best_score = 0.0

        #Trackers
        self.buffer_added = 0; self.best_score = 0.0
        self.rl_eval_flag = [True for i in range(self.args.num_action_rollouts)]; self.rl_scores =[]; self.rl_lens = []

    def add_experience(self, state, action, next_state, reward, done_probs, done):
        self.buffer_added += 1
        self.replay_buffer.push(state, next_state, action, reward, done_probs, done)
        if self.buffer_added % 100000 == 0: self.replay_buffer.save()

    def train(self, gen, frame_tracker):
        ################ ROLLOUTS ##############
        #Start RL rollouts
        for id in range(self.args.num_action_rollouts):
            if self.rl_eval_flag[id]:
                utils.hard_update(self.rollout_policies[id], self.rl_agent.actor)
                self.rl_task_pipes[id][0].send([id, self.rollout_policies[id]])
                self.rl_eval_flag[id] = False

        #Policy Gradient step
        if self.replay_buffer.__len__() > 5* self.args.batch_size:
            for _ in range(ITER_PER_EPOCH):
                s, ns, a, r = self.replay_buffer.sample(self.args.batch_size)
                s=s.cuda(); ns=ns.cuda(); a=a.cuda(); r=r.cuda()
                self.rl_agent.update_parameters(s, ns, a, r, num_epoch=1)


        #Process Rollout results if any
        all_fitness = []; all_net_ids = []; all_eplens = []
        for id in range(self.args.num_action_rollouts):
            if self.rl_result_pipes[id][0].poll():
                entry = self.rl_result_pipes[id][0].recv()
                all_fitness.append(entry[1]); all_net_ids.append(entry[0]); all_eplens.append(entry[2])
                self.rl_eval_flag[id] = True

        # Add ALL EXPERIENCE COLLECTED TO MEMORY concurrently
        for _ in range(len(self.exp_list)):
            exp = self.exp_list.pop()
            self.add_experience(exp[0], exp[1], exp[2], exp[3], exp[4], exp[5])

        ######### COMPUTE TEST SCORE IF ANY ############
        if 0 in all_net_ids:
            test_score = all_fitness[all_net_ids.index(0)]
            test_len = all_eplens[all_net_ids.index(0)]
            self.rl_scores.append(test_score); self.rl_lens.append(test_len)
            frame_tracker.update([test_score], agent.buffer_added)
            if test_score > self.best_score:
                self.best_score = test_score
                utils.hard_update(self.best_policy, self.rollout_policies[0])
                torch.save(self.best_policy.state_dict(), self.args.save_foldername + 'rl_models/' + 'pg_best')
                print("Best policy saved with score", test_score)


        return all_fitness, all_eplens, 0 in all_net_ids

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    frame_tracker = utils.Tracker(parameters.metric_save, ['full_pg'], '.csv')  # Initiate tracker

    #Initialize the environment
    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)
    agent = FULL_PG(parameters) #Initialize the agent
    print('Running osim-rl',  ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim, 'using FULL PG')

    time_start = time.time()
    for gen in range(1, 1000000000): #Infinite generations
        gen_time = time.time()
        all_fitness, all_eplen, test_complete = agent.train(gen, frame_tracker)
        if not test_complete: continue
        print('#Frames/k', int(agent.buffer_added/1000), ' Avg:','%.2f'%frame_tracker.all_tracker[0][1],'Time:','%.2f'%(time.time()-gen_time),
              'Best_yet', '%.2f'%agent.best_score,
              'Last_test','%.2f'%agent.rl_scores[-1] if len(agent.rl_scores)!=0 else None, 'Last_len','%.2f'%agent.rl_lens[-1] if len(agent.rl_lens)!=0 else None, best_fname,
              'Critic_loss:', '%.2f' % utils.list_mean(agent.rl_agent.critic_loss) if len(agent.rl_agent.critic_loss) != 0 else None,
              'Q_Val min/max', '%.2f' % utils.list_mean(agent.rl_agent.critic_loss_min)if len(agent.rl_agent.critic_loss_min) != 0 else None,
              '%.2f' % utils.list_mean(agent.rl_agent.critic_loss_max)if len(agent.rl_agent.critic_loss_max) != 0 else None,
              'Policy_loss:', '%.2f' % utils.list_mean(agent.rl_agent.policy_loss) if len(agent.rl_agent.policy_loss) != 0 else None)



















