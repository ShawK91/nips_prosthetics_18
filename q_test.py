import torch
import numpy as np, os, time, random, sys
from core import mod_utils as utils
from multiprocessing import Process, Pipe
from core import off_policy_gradient as pg
from core import models
os.environ["CUDA_VISIBLE_DEVICES"]='3'
from core.runner import rollout_worker
from core.mod_utils import list_mean, pprint

critic_fname = 'R_Skeleton/models/td3_critic0.95_-50.0_ADV_DMASK'
actor_fname = 'R_Skeleton/rl_models/td3_best0.95_-50.0_ADV_DMASK'



SEED = False
SEED_CHAMP = False
algo = 'TD3'    #1. TD3
                 #2. DDPG
                 #3. TD3_max   - TD3 but using max rather than min
DATA_LIMIT = 500000
RS_DONE_W = -40.0; RS_PROPORTIONAL_SHAPE = False

DONE_GAMMA = 0.8



class Parameters:
    def __init__(self):

        #MetaParams
        self.is_cuda= True
        self.algo = algo

        #RL params
        self.seed = 1991
        self.batch_size = 128
        self.action_loss = False
        self.action_loss_w = 1.0

        #TD3
        self.gamma = 0.996; self.tau = 0.001
        self.init_w = False
        self.policy_ups_freq = 2
        self.policy_noise = 0.03
        self.policy_noise_clip = 0.1

        self.use_advantage = True

        #TR Constraints
        self.critic_constraint = None
        self.critic_constraint_w = None
        self.q_clamp = None
        self.policy_constraint = None
        self.policy_constraint_w = None

        #Target Networks
        self.hard_update_freq = None
        self.hard_update = False

        #Save Results
        self.state_dim = 415; self.action_dim = 19 #Simply instantiate them here, will be initialized later
        self.save_foldername = 'R_Skeleton/'
        self.metric_save = self.save_foldername + 'metrics/'
        self.model_save = self.save_foldername + 'models/'
        self.rl_models = self.save_foldername + 'rl_models/'
        self.data_folder = self.save_foldername + 'data/'
        self.aux_save = self.save_foldername + 'auxiliary/'
        if not os.path.exists(self.save_foldername): os.makedirs(self.save_foldername)
        if not os.path.exists(self.metric_save): os.makedirs(self.metric_save)
        if not os.path.exists(self.model_save): os.makedirs(self.model_save)
        if not os.path.exists(self.rl_models): os.makedirs(self.rl_models)
        if not os.path.exists(self.data_folder): os.makedirs(self.data_folder)
        if not os.path.exists(self.aux_save): os.makedirs(self.aux_save)

        self.critic_fname = None; self.actor_fname = None; self.log_fname = None; self.best_fname = None

class Buffer():

    def __init__(self, capacity):
        self.capacity = capacity
        self.s = []; self.ns = []; self.a = []; self.r = []; self.done_dist =[]
        self.num_entries = 0
        self.loaded_files = []

    def __len__(self): return self.num_entries

    def sample(self, batch_size):
        ind = random.sample(range(self.__len__()), batch_size)
        return self.s[ind], self.ns[ind], self.a[ind], self.r[ind]

    def load(self, data_folder):

        ######## READ DATA #########
        list_files = os.listdir(data_folder)
        while len(list_files) < 1: continue #Wait for Data indefinitely
        print (list_files)

        for index, file in enumerate(list_files):
            if file not in self.loaded_files:
                data = np.load(data_folder + file)
                self.loaded_files.append(file)
                s = data['state']; ns = data['next_state']; a = data['action']; r = data['reward']; done_dist = data['done_dist']

                # Reward Shaping for premature falling
                if RS_PROPORTIONAL_SHAPE:
                    rs_flag = np.where(done_dist != -1) #All tuples which lead to premature convergence
                    #r[rs_flag] = r[rs_flag] - ((DONE_GAMMA ** done_dist[rs_flag]) * abs(r[rs_flag]))
                    r[rs_flag] = r[rs_flag] - (DONE_GAMMA ** done_dist[rs_flag]) * RS_DONE_W  # abs(r[rs_flag]))

                else:
                    rs_flag = np.where(done_dist == np.min(done_dist)) #All tuple which was the last experience in premature convergence
                    r[rs_flag] = RS_DONE_W


                if isinstance(self.s, list):
                    self.s = torch.Tensor(s); self.ns = torch.Tensor(ns); self.a = torch.Tensor(a); self.r = torch.Tensor(r); self.done_dist = torch.Tensor(done_dist)
                else:
                    self.s = torch.cat((self.s, torch.Tensor(s)), 0)
                    self.ns = torch.cat((self.ns, torch.Tensor(ns)), 0)
                    self.a = torch.cat((self.a, torch.Tensor(a)), 0)
                    self.r = torch.cat((self.r, torch.Tensor(r)), 0)
                    self.done_dist = torch.cat((self.done_dist, torch.Tensor(done_dist)), 0)

                self.num_entries = len(self.s)
                if self.num_entries >= DATA_LIMIT: break


        print('BUFFER LOADED WITH', self.num_entries, 'SAMPLES')

        self.s = self.s.pin_memory()
        self.ns = self.ns.pin_memory()
        self.a = self.a.pin_memory()
        self.r = self.r.pin_memory()
        #self.fit = torch.cat(self.fit).pin_memory()
        self.done_dist = self.done_dist.pin_memory()


args = Parameters()
agent = pg.TD3_DDPG(args)

agent.critic.load_state_dict(torch.load(critic_fname))
agent.actor.load_state_dict(torch.load(actor_fname))

buffer = Buffer(10000000)
buffer.load(args.data_folder)




tags = ['Worst', 'Bad', 'High_reward']
print()

#Test
for i in range(3):
    if i == 0: state_ind = [np.where(buffer.r == RS_DONE_W)[0]];
    if i == 1: state_ind = [np.where(buffer.r < RS_DONE_W)[0]];
    if i == 2: state_ind = [np.where(buffer.r > 20.0)[0]];

    states = buffer.s[state_ind].cuda()
    actions = agent.actor.forward(states)
    q1, q2, vals= agent.critic.forward(states, actions)
    print(tags[i], 'Vol:', len(state_ind[0]), 'Mean Q-vals', utils.pprint(torch.mean(q1).item()), 'Mean Val', utils.pprint(torch.mean(vals).item()), 'MEan Q-V', utils.pprint(torch.mean(q1).item()-torch.mean(vals).item()) )












