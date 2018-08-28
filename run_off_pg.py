import torch
import numpy as np, os, time, random, sys
from core import mod_utils as utils
from multiprocessing import Process, Pipe
from core import off_policy_gradient as pg
from core import models
os.environ["CUDA_VISIBLE_DEVICES"]='3'
from core.runner import rollout_worker


SEED = False
SEED_CHAMP = False
algo = 'DDPG'    #1. TD3
                 #2. DDPG
                 #3. TD3_max   - TD3 but using max rather than min
DATA_LIMIT = 50000000
RS_DONE_W = -10.0; RS_PROPORTIONAL_SHAPE = False
#FAIRLY STATIC
SAVE = True
QUICK_TEST = False
ITER_PER_EPOCH = 200
SAVE_THRESHOLD = 100
DONE_GAMMA = 0.98
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


class Parameters:
    def __init__(self):

        #MetaParams
        self.is_cuda= True
        self.algo = algo

        #RL params
        self.seed = 949
        self.batch_size = 256
        self.action_loss = False
        self.action_loss_w = 1.0

        #TD3
        self.gamma = 0.997; self.tau = 0.001
        self.init_w = False
        self.policy_ups_freq = 2
        self.policy_noise = 0.03
        self.policy_noise_clip = 0.1

        self.use_advantage = True

        #TR Constraints
        self.critic_constraint = None
        self.critic_constraint_w = 5.0
        self.q_clamp = None
        self.policy_constraint = None
        self.policy_constraint_w = 5.0

        #Target Networks
        self.hard_update_freq = 10000
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
                    r[rs_flag] = r[rs_flag] - ((DONE_GAMMA ** done_dist[rs_flag]) * abs(r[rs_flag]))

                else:
                    rs_flag = np.where(done_dist == np.min(done_dist)) #All tuple which was the last experience in premature convergence
                    r[rs_flag] = RS_DONE_W


                if isinstance(self.s, list):
                    self.s = torch.Tensor(s); self.ns = torch.Tensor(ns); self.a = torch.Tensor(a); self.r = torch.Tensor(r)
                else:
                    self.s = torch.cat((self.s, torch.Tensor(s)), 0)
                    self.ns = torch.cat((self.ns, torch.Tensor(ns)), 0)
                    self.a = torch.cat((self.a, torch.Tensor(a)), 0)
                    self.r = torch.cat((self.r, torch.Tensor(r)), 0)
                    #self.done = torch.cat((self.done, done), 0)

                self.num_entries = len(self.s)
                if self.num_entries > DATA_LIMIT: break


        print('BUFFER LOADED WITH', self.num_entries, 'SAMPLES')

        self.s = self.s.pin_memory()
        self.ns = self.ns.pin_memory()
        self.a = self.a.pin_memory()
        self.r = self.r.pin_memory()
        #self.fit = torch.cat(self.fit).pin_memory()
        #self.done = self.done.pin_memory()

class PG_ALGO:
    def __init__(self, args):
        self.args = args
        self.rl_agent = pg.TD3_DDPG(args)

        if SEED:
            try:
                if SEED_CHAMP:
                    self.rl_agent.actor.load_state_dict(torch.load('R_Skeleton/models/champ'))
                    self.rl_agent.actor_target.load_state_dict(torch.load('R_Skeleton/models/champ'))
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
        self.buffer = Buffer(10000000)
        self.buffer.load(self.args.data_folder)


        #RL ROLLOUT PROCESSOR
        #self.res_list = Manager().list(); self.job_count = 0
        self.rl_task_sender, self.rl_task_receiver = Pipe(); self.rl_res_sender, self.rl_res_receiver = Pipe()
        self.rl_worker = Process(target=rollout_worker, args=(0, self.rl_task_receiver, self.rl_res_sender, None, None, 0, False))
        self.rl_worker.start()
        self.rollout_policy = models.Actor(args)
        self.best_policy = models.Actor(args); self.best_score = 0.0
        self.job_count = 0; self.job_done = 0
        self.rollout_len = None; self.rollout_score = None

    def train(self, gen):

        if gen % 500 == 0: self.buffer.load(self.args.data_folder)
        #Start RL test Rollout
        if self.job_count <= self.job_done:
            self.rollout_policy.cuda()
            self.rl_agent.hard_update(self.rollout_policy, self.rl_agent.actor)
            self.rollout_policy.cpu()
            self.rl_task_sender.send([-1, self.rollout_policy])
            self.rollout_len = None; self.rollout_score = None
            self.job_count += 1

        ############################## RL LEANRING DURING POPULATION EVALUATION ##################
        #RL TRAIN
        for _ in range(ITER_PER_EPOCH):
            s, ns, a, r = self.buffer.sample(self.args.batch_size)
            s=s.cuda(); ns=ns.cuda(); a=a.cuda(); r=r.cuda()
            self.rl_agent.update_parameters(s, ns, a, r, num_epoch=1)


        ################################ EO RL LEARNING ########################

        if gen % 200 == 0 and QUICK_TEST != True and SAVE:
            #torch.save(self.rl_agent.actor.state_dict(), parameters.rl_models + actor_fname)
            torch.save(self.rl_agent.critic.state_dict(), parameters.model_save + critic_fname)
            print("Critic Saved periodically")

        #Periodically reload buffer
        #if gen % 5000 == 0: self.buffer.load(self.args.data_folder)

        # Prcoess rollout scores
        if self.rl_res_receiver.poll():
            entry = self.rl_res_receiver.recv()
            self.job_done += 1
            self.rollout_len = entry[2]
            self.rollout_score = entry[1]
            if self.rollout_score > self.best_score and not QUICK_TEST:
                self.best_score = self.rollout_score
                self.rl_agent.hard_update(self.best_policy, self.rollout_policy)
                if self.best_score > SAVE_THRESHOLD:
                    torch.save(self.best_policy.state_dict(), parameters.rl_models + best_fname)
                    print("Best policy saved with score ", self.best_score)

def shape_filename(fname, args):
    fname = fname + str(parameters.gamma) + '_'
    if RS_PROPORTIONAL_SHAPE: fname = fname + 'RS_PROP'
    else: fname = fname + str(RS_DONE_W)
    if args.use_advantage: fname = fname + '_ADV'
    return fname




if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    frame_tracker = utils.Tracker(parameters.metric_save, [log_fname], '.csv')  # Initiate tracker
    ml_tracker = utils.Tracker(parameters.aux_save, [log_fname+'critic_loss', log_fname+'policy_loss', log_fname+'action_loss'], '.csv')  # Initiate tracker

    #Initialize the environment

    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)
    agent = PG_ALGO(parameters) #Initialize the agent
    print('Running', algo,  ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim, 'using Data')

    time_start = time.time(); num_frames = 0.0

    #################################################### FILENAMES
    critic_fname = shape_filename(critic_fname, parameters)
    actor_fname = shape_filename(actor_fname, parameters)
    log_fname = shape_filename(log_fname, parameters)
    best_fname = shape_filename(best_fname, parameters)
    ####################################################


    for epoch in range(1, 1000000000): #Infinite generations
        gen_time = time.time()
        agent.train(epoch)
        print('Ep:', epoch,  'Last Score:','%.2f'%frame_tracker.all_tracker[0][1],
              'Time:','%.1f'%(time.time()-gen_time), 'Test_len', '%.1f'%agent.rollout_len if agent.rollout_len != None else None,  'Best_Score', '%.1f'%agent.best_score,
              'Critic_loss:', '%.2f'%utils.list_mean(agent.rl_agent.critic_loss), 'Q_Val min/max', '%.2f'%utils.list_mean(agent.rl_agent.critic_loss_min), '%.2f'%utils.list_mean(agent.rl_agent.critic_loss_max), 'Policy_loss:', '%.2f'%utils.list_mean(agent.rl_agent.policy_loss) if len(agent.rl_agent.policy_loss)!=0 else None )
        if epoch % 5 == 0: #Special Stats
            print()
            print ('Action_loss:', '%.2f'%utils.list_mean(agent.rl_agent.action_loss) if len(agent.rl_agent.action_loss)!=0 else None,
                   'Buffer_size/mil', '%.1f'%(agent.buffer.num_entries/1000000.0), 'Algo:', best_fname, 'Q_clamp', parameters.q_clamp,
                   'Gamma', parameters.gamma,
                   'RS_PROP', RS_PROPORTIONAL_SHAPE,
                   'ADVANTAGE', parameters.use_advantage)
            print()

        frame_tracker.update([agent.rollout_score], epoch)
        try:
            if len(agent.rl_agent.policy_loss) > 0 and not QUICK_TEST: ml_tracker.update([agent.rl_agent.critic_loss[-1], agent.rl_agent.policy_loss[-1], agent.rl_agent.action_loss[-1]], epoch)
        except: None










