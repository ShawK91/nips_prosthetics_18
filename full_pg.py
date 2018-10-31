from core import off_policy_gradient as pg
from core import models
from core.mod_utils import list_mean, pprint, str2bool
import core.reward_shaping as rs
import numpy as np, os, time, random, torch
from core import mod_utils as utils
from core.runner import rollout_worker
import core.ounoise as OU_handle
from torch.multiprocessing import Process, Pipe, Manager
#os.environ["CUDA_VISIBLE_DEVICES"]='3'
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-seed_policy', help='Where to seed from if any: none--> No seeding; no_entry --> R2_Skeleton/models/erl_best', default='R2_Skeleton/models/erl_best')
parser.add_argument('-save_folder', help='Primary save folder to save logs, data and policies',  default='R2_Skeleton')
parser.add_argument('-num_workers', type=int,  help='#Rollout workers',  default=12)
parser.add_argument('-shorts', type=str2bool,  help='#Short run',  default=False)
parser.add_argument('-mem_cuda', type=str2bool,  help='#Store buffer in GPU?',  default=False)


SEED = vars(parser.parse_args())['seed_policy']
SAVE_FOLDER = vars(parser.parse_args())['save_folder'] + '/'
NUM_WORKERS = vars(parser.parse_args())['num_workers']
USE_SYNTHETIC_TARGET = vars(parser.parse_args())['shorts']
XBIAS = False; ZBIAS = False; PHASE_LEN = 100
MEM_CUDA = vars(parser.parse_args())['mem_cuda']

#MACROS
SAVE_RS = False #When reward shaping is on, whether to save the best shaped performer or the true best performer
SAVE_THRESHOLD = 100 #Threshold for saving best policies
QUICK_TEST = False #DEBUG MODE
DIFFICULTY = 1 #Difficulty of the environment: 0 --> Round 1 and 1 --> Round 2


class Parameters:
    """Parameter class stores all parameters for policy gradient

    Parameters:
        None

    Returns:
        None
    """

    def __init__(self):

        #FAIRLY STATIC
        self.num_action_rollouts = NUM_WORKERS #Controls how many runners it uses to perform parallel rollouts
        self.is_cuda= True
        self.algo = 'TD3'    #1. TD3
                             #2. DDPG
        self.seed = 7
        self.batch_size = 256 #Batch size for learning
        self.gamma = 0.99 #Discount rate
        self.tau = 0.001 #Target network soft-update rate

        self.use_advantage = True #Use Advantage Function (Q-V)
        self.use_done_mask = True #Use done mask
        self.init_w = False #Whether to initialize model weights using Kaimiling Ne

        #Policy Gradient steps
        self.iter_per_epoch = 1000

        #TD3
        self.policy_ups_freq = 2 #Number of Critic updates per actor update
        self.policy_noise = 0.03 #Noise in policy output when computing Bellman update (smoothes the Critic)
        self.policy_noise_clip = 0.1 #Hard clip for ^^^

        ######### REWARD SHAPING ##########

        #Temporal Reward Shaping (flowing reward backward across a trajectory)
        self.rs_done_w = 0 #Penalty for the last transition that leads to falling (except within the last timestep)
        self.rs_proportional_shape = False #Flow the done_penalty backwards through the trajectory
        self.done_gamma= 0.93 #Discount factor for flowing back the done_penalty

        #Behavioral Reward Shaping (rs to encode behavior constraints)
        self.use_behavior_rs = False #Use behavioral reward shaping
        if self.use_behavior_rs:
            # R1
            if DIFFICULTY == 0:
                self.footz_w= -5.0 #No foot criss-crossing the z-axis
                self.kneefoot_w = -7.5 #Knee remains in front of foot plus the tibia is always bend backward
                self.pelv_w = -5.0 #Pelvis is below 0.8m (crouched)
                self.footy_w = 0.0 #Foot is below 0.1m (don't raise foot too high)
                self.head_w = -5.0 #Head is behind the pelvis in x

        #Trust-region Constraints
        self.trust_region_actor = False
        self.critic_constraint = None
        self.critic_constraint_w = None
        self.q_clamp = None
        self.policy_constraint = None
        self.policy_constraint_w = None

        #Target Networks
        self.hard_update_freq = None
        self.hard_update = False

        #Action loss (entropy analogy for continous action space)
        self.action_loss = False; self.action_loss_w = 1.0

        self.state_dim = 415; self.action_dim = 19 #HARDCODED FOR THIS PROBLEM
        #Save Results
        if DIFFICULTY == 0: self.save_foldername = 'R_Skeleton/'
        else: self.save_foldername = SAVE_FOLDER

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

        self.critic_fname = 'td3_critic' if self.algo == 'TD3' else 'ddpg_critic'
        self.actor_fname = 'td3_actor' if self.algo == 'TD3' else 'ddpg_actor'
        self.log_fname = 'td3_epoch' if self.algo == 'TD3' else 'ddpg_epoch'
        self.best_fname = 'td3_best' if self.algo == 'TD3' else 'ddpg_best'

        if QUICK_TEST:
            self.num_action_rollouts = 3

class Memory():
    """Memory Object loads and stores experience tuples from the drive. Similar to Buffer except that memory refers to old buffers (generated by a different learner and stored in disk)

        Parameters:
            capacity (int): Maximum number of experiences to load

        """

    def __init__(self, capacity, args):
        self.capacity = capacity; self.args = args
        self.s = []; self.ns = []; self.a = []; self.r = []; self.done =[]
        self.num_entries = 0
        self.loaded_files = []

    def __len__(self): return self.num_entries

    def sample(self, batch_size):
        """Sample a batch of experiences from memory with uniform probability

            Parameters:
                batch_size (int): Size of the batch to sample
                args (object): Parameter class

            Returns:
                Experience (tuple): A tuple of (state, next_state, action, reward, done) each as a numpy array with shape (batch_size, :)
        """
        ind = random.sample(range(self.__len__()), batch_size)
        return self.s[ind], self.ns[ind], self.a[ind], self.r[ind], self.done[ind]

    def load(self, data_folder):
        """Load experiences from drive

            Parameters:
                data_folder (str): Folder to load data from

            Returns:
                None
        """

        ######## READ DATA #########
        list_files = os.listdir(data_folder)
        while len(list_files) < 1:
            time.sleep(10)
            continue #Wait for Data indefinitely

        #Randomize data buffers
        list_files = random.sample(list_files, len(list_files))

        num_loaded = 0
        for index, file in enumerate(list_files):

            #Limit max_uploads to 20
            if num_loaded > 20: break

            if file not in self.loaded_files:
                try: data = np.load(data_folder + file)
                except: continue
                num_loaded +=1
                self.loaded_files.append(file)
                s = data['state']; ns = data['next_state']; a = data['action']; r = data['reward']; done_dist = data['done_dist']


                #Round 2 Reward Scalarization
                if DIFFICULTY != 0:
                    r[:] = r[:] - (8.5*5) #Translate (get rid of the survival bonus)
                    pos_filter = np.where(r > 0)
                    r[pos_filter] = r[pos_filter] * 10
                    #r[:] = r[:] * 5 #Scale to highlight the differences

                # Reward Shaping for premature falling
                if self.args.rs_proportional_shape:
                    rs_flag = np.where(done_dist != -1) #All tuples which lead to premature convergence
                    #r[rs_flag] = r[rs_flag] - ((DONE_GAMMA ** done_dist[rs_flag]) * abs(r[rs_flag]))
                    r[rs_flag] = r[rs_flag] + (self.args.done_gamma ** done_dist[rs_flag]) * self.args.rs_done_w  # abs(r[rs_flag]))

                else:
                    rs_flag = np.where(done_dist == np.min(done_dist)) #All tuple which was the last experience in premature convergence
                    r[rs_flag] = self.args.rs_done_w



                ############## BEHAVIORAL REWARD SHAPE #########
                if self.args.use_behavior_rs:

                    #R1
                    if DIFFICULTY == 0:
                        r = rs.shaped_data(s,r,self.args.footz_w, self.args.kneefoot_w, self.args.prlv_w, self.args.footy_w, self.args.head_w)


                done = (done_dist == 1).astype(float)
                if isinstance(self.s, list):
                    self.s = torch.Tensor(s); self.ns = torch.Tensor(ns); self.a = torch.Tensor(a); self.r = torch.Tensor(r); self.done = torch.Tensor(done)

                else:
                    self.s = torch.cat((self.s, torch.Tensor(s)), 0)
                    self.ns = torch.cat((self.ns, torch.Tensor(ns)), 0)
                    self.a = torch.cat((self.a, torch.Tensor(a)), 0)
                    self.r = torch.cat((self.r, torch.Tensor(r)), 0)
                    self.done = torch.cat((self.done, torch.Tensor(done)), 0)

                self.num_entries = len(self.s)
                if QUICK_TEST and self.__len__() > 1000:
                    print('######## DEBUG MODE ########')
                    break


        print('BUFFER LOADED WITH', self.num_entries, 'SAMPLES')

        # self.s = self.s.pin_memory()
        # self.ns = self.ns.pin_memory()
        # self.a = self.a.pin_memory()
        # self.r = self.r.pin_memory()
        # self.done = self.done.pin_memory()

        if MEM_CUDA:
            self.s = self.s.cuda()
            self.ns = self.ns.cuda()
            self.a = self.a.cuda()
            self.r = self.r.cuda()
            self.done = self.done.cuda()

class Buffer():
    """Cyclic Buffer stores experience tuples from the rollouts

        Parameters:
            save_freq (int): Period for saving data to drive
            save_folder (str): Folder to save data to
            capacity (int): Maximum number of experiences to hold in cyclic buffer

        """

    def __init__(self, save_freq, save_folder, capacity = 1000000):
        self.save_freq = save_freq; self.folder = save_folder; self.capacity = 1000000
        self.s = []; self.ns = []; self.a = []; self.r = []; self.done_dist = []; self.done = []; self.shaped_r = []
        self.counter = 0

    def push(self, s, ns, a, r, done_dist, done, shaped_r): #f: FITNESS, t: TIMESTEP, done: DONE
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


        if self.__len__() < self.capacity:
            self.s.append(None); self.ns.append(None); self.a.append(None); self.r.append(None); self.done_dist.append(None); self.done.append(None); self.shaped_r.append(None)

        #Append new tuple
        ind = self.counter % self.capacity
        self.s[ind] = s; self.ns[ind] = ns; self.a[ind] = a; self.r[ind] = r; self.done_dist[ind] = done_dist; self.done[ind] = done; self.shaped_r[ind] = shaped_r
        self.counter += 1

    def __len__(self):
        return len(self.s)

    def sample(self, batch_size):
        """Sample a batch of experiences from memory with uniform probability

               Parameters:
                   batch_size (int): Size of the batch to sample

               Returns:
                   Experience (tuple): A tuple of (state, next_state, action, shaped_reward, done) each as a numpy array with shape (batch_size, :)
           """
        ind = random.sample(range(self.__len__()), batch_size)
        return np.vstack([self.s[i] for i in ind]), np.vstack([self.ns[i] for i in ind]), np.vstack([self.a[i] for i in ind]), np.vstack([self.shaped_r[i] for i in ind]), np.vstack([self.done_dist[i] for i in ind])


    def save(self):
        """Method to save experiences to drive

               Parameters:
                   None

               Returns:
                   None
           """

        tag = str(int(self.counter / self.save_freq))
        list_files = os.listdir(self.folder)
        while True:
            if (self.folder + 'pgdata_' + tag) in list_files:
                tag += 1
                continue
            break


        end_ind = self.counter % self.capacity
        start_ind = end_ind - self.save_freq

        try:
            np.savez_compressed(self.folder + 'pgdata_' + tag,
                            state=np.vstack(self.s[start_ind:end_ind]),
                            next_state=np.vstack(self.ns[start_ind:end_ind]),
                            action = np.vstack(self.a[start_ind:end_ind]),
                            reward = np.vstack(self.r[start_ind:end_ind]),
                            done_dist = np.vstack(self.done_dist[start_ind:end_ind]),
                            done=np.vstack(self.done[start_ind:end_ind]))
            print ('MEMORY BUFFER WITH INDEXES', str(start_ind), str(end_ind),  'SAVED WITH TAG', tag)
        except:
            print()
            print()
            print()
            print()
            print('############ WARNING! FAILED TO SAVE FROM INDEX ', str(start_ind), 'to', str(end_ind), '################')
            print()
            print()
            print()
            print()

class PG_ALGO:
    """Policy Gradient Algorithm main object which carries out off-policy learning using policy gradient
       Encodes all functionalities for 1. TD3 2. DDPG 3.Trust-region TD3/DDPG 4. Advantage TD3/DDPG

            Parameters:
                args (int): Parameter class with all the parameters

            """

    def __init__(self, args):
        self.args = args
        self.rl_agent = pg.TD3_DDPG(args)
        self.new_rlagent = pg.TD3_DDPG(args)


        #Use seed to bootstrap learning
        if SEED != 'none':
            try:

                #RLAGENT 2 always loads from erl_best
                self.new_rlagent.actor.load_state_dict(torch.load(args.model_save + 'erl_best'))
                self.new_rlagent.actor_target.load_state_dict(torch.load(args.model_save + 'erl_best'))
                self.rl_agent.actor.load_state_dict(torch.load(SEED))
                self.rl_agent.actor_target.load_state_dict(torch.load(SEED))

                print('Actor successfully loaded from ', SEED)
            except: print('Loading Actors failed')

            try:
                self.rl_agent.critic_target.load_state_dict(torch.load(args.model_save + self.args.critic_fname))
                self.rl_agent.critic.load_state_dict(torch.load(args.model_save + self.args.critic_fname))
                self.new_rlagent.critic_target.load_state_dict(torch.load(args.model_save + self.args.critic_fname))
                self.new_rlagent.critic.load_state_dict(torch.load(args.model_save + self.args.critic_fname))
                print('Critic successfully loaded from',args.model_save +  self.args.critic_fname)
            except: print ('Loading Critics failed')

        #Load to GPU
        self.rl_agent.actor_target.cuda(); self.rl_agent.critic_target.cuda(); self.rl_agent.critic.cuda(); self.rl_agent.actor.cuda()
        self.new_rlagent.actor_target.cuda(); self.new_rlagent.critic_target.cuda(); self.new_rlagent.critic.cuda(); self.new_rlagent.actor.cuda()


        ############ LOAD DATA AND CONSTRUCT GENERATORS ########
        self.memory = Memory(10000000, args)
        self.memory.load(self.args.data_folder)

        ###### Buffer is agent's own data self generated via its rollouts #########
        self.replay_buffer = Buffer(save_freq=100000, save_folder=self.args.data_folder, capacity=1000000)
        self.noise_gen = OU_handle.get_list_generators(args.num_action_rollouts, args.action_dim)

        ######### Multiprocessing TOOLS #########
        self.manager = Manager()
        self.exp_list = self.manager.list() #Experience list stores experiences from all processes

        ######## TEST ROLLOUT POLICY ############
        self.test_policy = self.manager.list(); self.test_policy.append(models.Actor(args)); self.test_policy.append(models.Actor(args))
        self.test_task_pipes = [Pipe() for _ in range(2)]
        self.test_result_pipes = [Pipe() for _ in range(2)]
        self.test_worker = [Process(target=rollout_worker, args=(i, self.test_task_pipes[i][1], self.test_result_pipes[i][0], None, self.exp_list, self.test_policy, DIFFICULTY, SAVE_RS, True, USE_SYNTHETIC_TARGET, XBIAS, ZBIAS, PHASE_LEN)) for i in range(2)]
        for worker in self.test_worker: worker.start()

        ######### TRAIN ROLLOUTS WITH ACTION NOISE ############
        self.rollout_pop = self.manager.list()
        for _ in range(self.args.num_action_rollouts): self.rollout_pop.append(models.Actor(args))
        self.task_pipes = [Pipe() for _ in range(args.num_action_rollouts)]
        self.result_pipes = [Pipe() for _ in range(args.num_action_rollouts)]
        self.train_workers = [Process(target=rollout_worker, args=(i, self.task_pipes[i][1], self.result_pipes[i][0], self.noise_gen[i], self.exp_list, self.rollout_pop,  DIFFICULTY, SAVE_RS, True,USE_SYNTHETIC_TARGET, XBIAS, ZBIAS, PHASE_LEN)) for i in range(args.num_action_rollouts)]
        for worker in self.train_workers: worker.start()



        #### STATS AND TRACKING WHICH ROLLOUT IS DONE ######
        self.best_policy = models.Actor(args) #Best policy found by PF yet
        self.best_score = 0.0; self.buffer_added = 0; self.test_len = [None, None]; self.test_score = [None, None]; self.best_action_noise_score = 0.0; self.best_agent_scores = [0.0, 0.0]
        self.action_noise_scores = [0.0 for _ in range(args.num_action_rollouts)]
        self.test_eval_flag = [True, True]
        self.train_eval_flag = [True for _ in range(args.num_action_rollouts)]
        self.burn_in_period = True


    def add_experience(self, state, action, next_state, reward, done_dist, done):
        """Process and send experiences to be added to the buffer

              Parameters:
                  state (ndarray): Current State
                  next_state (ndarray): Next State
                  action (ndarray): Action
                  reward (ndarray): Reward
                  done_dist (ndarray): Temporal distance to done (#action steps after which the skselton fell over)
                  done (ndarray): Done
                  shaped_r (ndarray): Shaped Reward (includes both temporal and behavioral shaping)

              Returns:
                  None
          """

        self.buffer_added += 1

        shaped_r = np.zeros((1,1))
        #Reward Scalarization

        shaped_r[0,0] = (float(reward) - (8.5*5))
        if shaped_r[0,0] > 0: shaped_r[0,0] = shaped_r[0,0] * 10


        #RS to GENERATE SHAPED_REWARD

        if self.args.rs_proportional_shape:
            if done_dist[0,0] != -1:
                shaped_r[0,0] = float(reward) + (self.args.done_gamma ** done_dist[0,0]) * self.args.rs_done_w
        else:
            if done_dist[0,0] == 1:
                shaped_r[0,0] = self.args.rs_done_w



        if self.args.use_behavior_rs:
            if DIFFICULTY == 0:
                shaped_r = rs.shaped_data(state, shaped_r, self.args.footz_w, self.args.kneefoot_w, self.args.pelv_w, self.args.footy_w, self.args.head_w)

        self.replay_buffer.push(state, next_state, action, reward, done_dist, done, shaped_r)
        if self.buffer_added % self.replay_buffer.save_freq == 0: self.replay_buffer.save()


    def train(self, gen):
        """Main training loop to do rollouts and run policy gradients

            Parameters:
                gen (int): Current epoch of training

            Returns:
                None
        """

        if gen % 100 == 0:
            self.memory.s = [];
            self.memory.ns = [];
            self.memory.a = [];
            self.memory.r = [];
            self.memory.done = []
            self.loaded_files = []
            self.memory.load(self.args.data_folder) #Reload memory
        #if gen % 2000 == 0: self.best_policy.load_state_dict(torch.load(self.args.model_save + 'erl_best')) #Referesh best policy

        ########### START TEST ROLLOUT ##########
        if self.test_eval_flag[0]: #ALL DATA TEST
            self.test_eval_flag[0] = False
            self.rl_agent.actor.cpu()
            self.rl_agent.hard_update(self.test_policy[0], self.rl_agent.actor)
            self.rl_agent.actor.cuda()
            self.test_task_pipes[0][0].send(True)

        if self.test_eval_flag[1] and not self.burn_in_period and self.args.num_action_rollouts != 0:  # Iterative RL Learner (New RL Agent)
            self.test_eval_flag[1] = False
            self.new_rlagent.actor.cpu()
            self.new_rlagent.hard_update(self.test_policy[1], self.new_rlagent.actor)
            self.new_rlagent.actor.cuda()
            self.test_task_pipes[1][0].send(True)


        ########## START TRAIN (ACTION NOISE) ROLLOUTS ##########
        for i in range(self.args.num_action_rollouts):
            if self.train_eval_flag[i]:
                self.train_eval_flag[i] = False

                #1/3 ACTION ROLLOUT ON BEST POLICY 1/3 ON CURRENT POLICY FOR L1 and L2
                if i % 3 == 0:
                    self.rl_agent.hard_update(self.rollout_pop[i], self.best_policy)
                elif i % 3 == 1:
                    self.rl_agent.actor.cpu()
                    self.rl_agent.hard_update(self.rollout_pop[i], self.rl_agent.actor)
                    self.rl_agent.actor.cuda()
                else:
                    if not self.burn_in_period: #If BURN IN PERIOD HAS COMPLETED, RUN ROLLOUT USING POLICY
                        self.new_rlagent.actor.cpu()
                        self.new_rlagent.hard_update(self.rollout_pop[i], self.new_rlagent.actor)
                        self.new_rlagent.actor.cuda()
                    else: #IF IN BURN IN PERIOD: JUST RUN MORE ROLLOUTS FOR BEST POLICY
                        self.rl_agent.hard_update(self.rollout_pop[i], self.best_policy)

                self.task_pipes[i][0].send(True)



        ############################## RL LEANRING DURING POPULATION EVALUATION ##################
        #TRAIN FROM MEMORY (PREVIOUS DATA) for RL AGENT
        if self.memory.__len__() > 10000: #MEMORY WAIT
            for _ in range(self.args.iter_per_epoch):
                s, ns, a, r, done = self.memory.sample(self.args.batch_size)
                s=s.cuda(); ns=ns.cuda(); a=a.cuda(); r=r.cuda(); done = done.cuda()
                self.rl_agent.update_parameters(s, ns, a, r, done, num_epoch=1)

        ##TRAIN FROM SELF_GENERATED DATA FOR NEW_RL_AGENT
        if self.replay_buffer.__len__() > 10000: #BURN IN PERIOD
            if self.burn_in_period:
                self.burn_in_period = False
                self.rl_agent.hard_update(self.new_rlagent.critic, self.rl_agent.critic)
                self.rl_agent.hard_update(self.new_rlagent.actor, self.rl_agent.actor)


        if not self.burn_in_period:
            for _ in range(int(self.args.iter_per_epoch/20)):
                s, ns, a, shaped_r, done_dist = self.replay_buffer.sample(self.args.batch_size)

                done = (done_dist == 1).astype(float)
                s = torch.Tensor(s); ns = torch.Tensor(ns); a = torch.Tensor(a);
                shaped_r = torch.Tensor(shaped_r); done = torch.Tensor(done)

                s=s.cuda(); ns=ns.cuda(); a=a.cuda(); shaped_r=shaped_r.cuda(); done = done.cuda()
                self.new_rlagent.update_parameters(s, ns, a, shaped_r, done, num_epoch=1)

        ################################ EO POLICY GRSDIENT ########################

        #Save critic periodically
        if gen % 200 == 0 and QUICK_TEST != True and not QUICK_TEST:
            #torch.save(self.rl_agent.actor.state_dict(), parameters.rl_models + actor_fname)
            torch.save(self.rl_agent.critic.state_dict(), parameters.model_save + self.args.critic_fname)
            print("Critic Saved periodically")


        ##### PROCESS TEST ROLLOUT ##########
        for i in range(2):
            if self.test_result_pipes[i][1].poll():
                entry = self.test_result_pipes[i][1].recv()
                self.test_eval_flag[i] = True
                self.test_score[i] = entry[1]
                self.test_len[i] = entry[2]

                if self.test_score[i] > self.best_score:
                    self.best_score = self.test_score[i]
                    self.best_agent_scores[i] = self.test_score[i]
                    if self.best_score > SAVE_THRESHOLD:
                        self.rl_agent.hard_update(self.best_policy, self.test_policy[i])
                        if not QUICK_TEST:
                            torch.save(self.best_policy.state_dict(), parameters.rl_models + self.args.best_fname)
                            print("Best policy saved with score ", self.best_score, 'originated from RL_Agent Index ', str(i))


        ####### PROCESS TRAIN ROLLOUTS ########
        for i in range(self.args.num_action_rollouts):
            if self.result_pipes[i][1].poll():
                entry = self.result_pipes[i][1].recv()
                self.train_eval_flag[i] = True
                self.action_noise_scores[i] = entry[1]
                if entry[1] > self.best_action_noise_score:
                    self.best_action_noise_score = entry[1]

        # Add ALL EXPERIENCE COLLECTED TO MEMORY concurrently
        for _ in range(len(self.exp_list)):
            exp = self.exp_list.pop()
            self.add_experience(exp[0], exp[1], exp[2], exp[3], exp[4], exp[5])

def shape_filename(fname, args):
    """Helper function to manipulate strings for setting save filenames that reflect the parameters

          Parameters:
            fname (str): Filename
            args (object): Parameter class

          Returns:
              fname (str): New filename
      """

    fname = fname + '_' + str(parameters.gamma)
    #if args.rs_proportional_shape: fname = fname + 'RS_PROP' + str(args.done_gamma) + '_'
    #fname + str(args.rs_done_w)
    #if args.use_advantage: fname = fname + '_ADV'
    #if args.use_behavior_rs:
        #if DIFFICULTY == 0:
            #fname = fname + '_' + str(args.footz_w)+ '_' + str(args.kneefoot_w)+ '_' + str(args.pelv_w)+ '_' + str(args.footy_w)

    return fname


if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class

    #################### PRCOESS FILENAMES TO SAVE PROGRESS  ################################
    parameters.critic_fname = shape_filename(parameters.critic_fname, parameters)
    parameters.actor_fname = shape_filename(parameters.actor_fname, parameters)
    parameters.log_fname = shape_filename(parameters.log_fname, parameters)
    parameters.best_fname = shape_filename(parameters.best_fname, parameters)
    ####################################################

    #
    frame_tracker = utils.Tracker(parameters.metric_save, [parameters.log_fname+'_1', parameters.log_fname+'_2'], '.csv')  # Initiate tracker
    ml_tracker = utils.Tracker(parameters.aux_save, [parameters.log_fname+'critic_loss', parameters.log_fname+'policy_loss'], '.csv')  # Initiate tracker
    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)    #Seeds

    # INITIALIZE THE MAIN AGENT CLASS
    agent = PG_ALGO(parameters)
    print('Running', parameters.algo,  ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim, 'for', 'Round 1' if DIFFICULTY == 0 else 'Round 2')
    time_start = time.time(); num_frames = 0.0

    ###### TRAINING LOOP ########
    for epoch in range(1, 1000000000): #RUN VIRTUALLY FOREVER
        gen_time = time.time()

        #ONE EPOCH OF TRAINING
        agent.train(epoch)

        #PRINT PROGRESS
        print('Ep:', epoch, 'Score cur/best:', [pprint(score) for score in agent.test_score], pprint(agent.best_score),
              'Time:',pprint(time.time()-gen_time), 'Len', pprint(agent.test_len), 'Best_action_noise_score', pprint(agent.best_action_noise_score), 'Best_Agent_scores', [pprint(score) for score in agent.best_agent_scores])

        #PRINT MORE DETAILED STATS PERIODICALLY
        if epoch % 5 == 0: #Special Stats
            print()
            print('#Data_Created', agent.buffer_added, 'Q_Val Stats', pprint(list_mean(agent.rl_agent.q['min'])), pprint(list_mean(agent.rl_agent.q['max'])), pprint(list_mean(agent.rl_agent.q['mean'])),
              'Val Stats', pprint(list_mean(agent.rl_agent.val['min'])), pprint(list_mean(agent.rl_agent.val['max'])), pprint(list_mean(agent.rl_agent.val['mean'])))
            print()
            print ('Memory_size/mil', pprint(agent.memory.num_entries/1000000.0), 'Algo:', parameters.best_fname,
                   'Gamma', parameters.gamma,
                   'RS_PROP', parameters.rs_proportional_shape,
                   'ADVANTAGE', parameters.use_advantage)
            print('Action Noise Rollouts: ', [pprint(score) for score in agent.action_noise_scores])
            print()
            print('Critic_loss mean:', pprint(list_mean(agent.rl_agent.critic_loss['mean'])),
              'Policy_loss Stats:', pprint(list_mean(agent.rl_agent.policy_loss['min'])),
              pprint(list_mean(agent.rl_agent.policy_loss['max'])),
              pprint(list_mean(agent.rl_agent.policy_loss['mean'])),
              )
            print('Critic_loss mean_new:', pprint(list_mean(agent.new_rlagent.critic_loss['mean'])),
              'Policy_loss_new Stats:', pprint(list_mean(agent.new_rlagent.policy_loss['min'])),
              pprint(list_mean(agent.new_rlagent.policy_loss['max'])),
              pprint(list_mean(agent.new_rlagent.policy_loss['mean'])),
              )

           # if agent.rl_agent.action_loss['min'] != None:
           #     print('Action_loss Stats', pprint(list_mean(agent.rl_agent.action_loss['min'])),
           #          pprint(list_mean(agent.rl_agent.action_loss['max'])),
           #          pprint(list_mean(agent.rl_agent.action_loss['mean'])),
           #          pprint(list_mean(agent.rl_agent.action_loss['std'])))
            print()

        #Update score to trackers
        frame_tracker.update([agent.test_score[0], agent.test_score[1]], epoch)
        try: ml_tracker.update([agent.rl_agent.critic_loss['mean'][-1], agent.rl_agent.policy_loss['mean'][-1]], epoch)
        except: None










