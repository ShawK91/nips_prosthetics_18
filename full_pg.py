import os
from core import off_policy_gradient as pg
from core import models
os.environ["CUDA_VISIBLE_DEVICES"]='3'
from core.mod_utils import list_mean, pprint
import core.reward_shaping as rs
import numpy as np, os, time, random, torch, sys
from core import mod_utils as utils
from core.runner import rollout_worker
import core.ounoise as OU_handle
from torch.multiprocessing import Process, Pipe, Manager



SEED = True
SEED_CHAMP = True
algo = 'TD3'    #1. TD3
                 #2. DDPG
                 #3. TD3_max   - TD3 but using max rather than min
DATA_LIMIT = 50000000
RS_DONE_W = -50.0; RS_PROPORTIONAL_SHAPE = True
DONE_GAMMA = 0.9
#FAIRLY STATIC
SAVE = True
QUICK_TEST = False
ITER_PER_EPOCH = 200
SAVE_THRESHOLD = 100

if QUICK_TEST:
    ITER_PER_EPOCH = 1
    SEED = False
    DATA_LIMIT = 1000

SAVE_RS = False #Best policy save rs or true score
USE_BEHAVIOR_RS = False
if USE_BEHAVIOR_RS:
    FOOTZ_W = -5.0; KNEEFOOT_W = -7.5; PELV_W = -5.0; FOOTY_W = 0.0; HEAD_W = -5.0

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

        #NUmber of rollouts
        self.num_action_rollouts = 27


        #MetaParams
        self.is_cuda= True
        self.algo = algo

        #RL params
        self.seed = 1991
        self.batch_size = 128
        self.action_loss = False
        self.action_loss_w = 1.0


        #TD3
        self.gamma = 0.95; self.tau = 0.001
        self.init_w = False
        self.policy_ups_freq = 2
        self.policy_noise = 0.03
        self.policy_noise_clip = 0.1

        self.use_advantage = True
        self.use_done_mask = True

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

class Memory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.s = []; self.ns = []; self.a = []; self.r = []; self.done =[]
        self.num_entries = 0
        self.loaded_files = []

    def __len__(self): return self.num_entries

    def sample(self, batch_size):
        ind = random.sample(range(self.__len__()), batch_size)
        return self.s[ind], self.ns[ind], self.a[ind], self.r[ind], self.done[ind]

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
                    r[rs_flag] = r[rs_flag] + (DONE_GAMMA ** done_dist[rs_flag]) * RS_DONE_W  # abs(r[rs_flag]))

                else:
                    rs_flag = np.where(done_dist == np.min(done_dist)) #All tuple which was the last experience in premature convergence
                    r[rs_flag] = RS_DONE_W

                ############## BEHAVIORAL REWARD SHAPE #########
                if USE_BEHAVIOR_RS: r = rs.shaped_data(s,r,FOOTZ_W, KNEEFOOT_W, PELV_W, FOOTY_W, HEAD_W)



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
                if self.num_entries > DATA_LIMIT: break


        print('BUFFER LOADED WITH', self.num_entries, 'SAMPLES')

        # self.s = self.s.pin_memory()
        # self.ns = self.ns.pin_memory()
        # self.a = self.a.pin_memory()
        # self.r = self.r.pin_memory()
        # self.done = self.done.pin_memory()

class Buffer():

    def __init__(self, save_freq, save_folder, capacity = 1000000):
        self.save_freq = save_freq; self.folder = save_folder; self.capacity = 1000000
        self.s = []; self.ns = []; self.a = []; self.r = []; self.done_dist = []; self.done = []; self.shaped_r = []
        self.counter = 0

    def push(self, s, ns, a, r, done_dist, done, shaped_r): #f: FITNESS, t: TIMESTEP, done: DONE

        if self.__len__() < self.capacity:
            self.s.append(None); self.ns.append(None); self.a.append(None); self.r.append(None); self.done_dist.append(None); self.done.append(None); self.shaped_r.append(None)

        #Append new tuple
        ind = self.counter % self.capacity
        self.s[ind] = s; self.ns[ind] = ns; self.a[ind] = a; self.r[ind] = r; self.done_dist[ind] = done_dist; self.done[ind] = done; self.shaped_r[ind] = shaped_r
        self.counter += 1

    def __len__(self):
        return len(self.s)

    def sample(self, batch_size):
        ind = random.sample(range(self.__len__()), batch_size)
        return np.vstack([self.s[i] for i in ind]), np.vstack([self.ns[i] for i in ind]), np.vstack([self.a[i] for i in ind]), np.vstack([self.shaped_r[i] for i in ind]), np.vstack([self.done_dist[i] for i in ind])

    def save(self):
        tag = str(int(self.counter / self.save_freq))

        end_ind = self.counter % self.capacity
        start_ind = end_ind - self.save_freq

        np.savez_compressed(self.folder + 'pg_data_' + tag,
                            state=np.vstack(self.s[start_ind:end_ind]),
                            next_state=np.vstack(self.ns[start_ind:end_ind]),
                            action = np.vstack(self.a[start_ind:end_ind]),
                            reward = np.vstack(self.r[start_ind:end_ind]),
                            done_dist = np.vstack(self.done_dist[start_ind:end_ind]),
                            done=np.vstack(self.done[start_ind:end_ind]))
        print ('MEMORY BUFFER WITH INDEXES', str(start_ind), str(end_ind),  'SAVED WITH TAG', tag)



class PG_ALGO:
    def __init__(self, args):
        self.args = args
        self.rl_agent = pg.TD3_DDPG(args)
        self.new_rlagent = pg.TD3_DDPG(args)

        if SEED:
            try:
                if SEED_CHAMP:
                    self.rl_agent.actor.load_state_dict(torch.load('R_Skeleton/models/erl_best'))
                    self.rl_agent.actor_target.load_state_dict(torch.load('R_Skeleton/models/erl_best'))
                    self.new_rlagent.actor.load_state_dict(torch.load('R_Skeleton/models/erl_best'))
                    self.new_rlagent.actor_target.load_state_dict(torch.load('R_Skeleton/models/erl_best'))
                else:
                    self.rl_agent.actor.load_state_dict(torch.load(args.rl_models + self.args.best_fname))
                    self.rl_agent.actor_target.load_state_dict(torch.load(args.rl_models + self.args.best_fname))
                    self.new_rlagent.actor.load_state_dict(torch.load(args.rl_models + self.args.best_fname))
                    self.new_rlagent.actor_target.load_state_dict(torch.load(args.rl_models + self.args.best_fname))

                print('Actor successfully loaded from the R_Skeleton folder for', self.args.best_fname)
            except: print('Loading Actors failed')
            try:
                self.rl_agent.critic_target.load_state_dict(torch.load(args.model_save + self.args.critic_fname))
                self.rl_agent.critic.load_state_dict(torch.load(args.model_save + self.args.critic_fname))
                self.new_rlagent.critic_target.load_state_dict(torch.load(args.model_save + self.args.critic_fname))
                self.new_rlagent.critic.load_state_dict(torch.load(args.model_save + self.args.critic_fname))
                print('Critic successfully loaded from the R_Skeleton folder for', self.args.critic_fname)
            except: print ('Loading Critics failed')

        self.rl_agent.actor_target.cuda(); self.rl_agent.critic_target.cuda(); self.rl_agent.critic.cuda(); self.rl_agent.actor.cuda()
        self.new_rlagent.actor_target.cuda(); self.new_rlagent.critic_target.cuda(); self.new_rlagent.critic.cuda(); self.new_rlagent.actor.cuda()


        ############ LOAD DATA AND CONSTRUCT GENERATORS ########
        self.memory = Memory(10000000)
        self.memory.load(self.args.data_folder)

        ###### Buffer is agent's own data self generated via its rollouts #########
        self.replay_buffer = Buffer(save_freq=100000, save_folder=self.args.data_folder, capacity=1000000)
        self.noise_gen = OU_handle.get_list_generators(args.num_action_rollouts, args.action_dim)

        ######### MP TOOLS #########
        self.manager = Manager()
        self.exp_list = self.manager.list()

        ######## TEST ROLLOUT POLICY ############
        self.test_policy = self.manager.list(); self.test_policy.append(models.Actor(args)); self.test_policy.append(models.Actor(args))
        self.test_task_pipes = [Pipe() for _ in range(args.num_action_rollouts)]
        self.test_result_pipes = [Pipe() for _ in range(args.num_action_rollouts)]
        self.test_worker = [Process(target=rollout_worker, args=(i, self.test_task_pipes[i][1], self.test_result_pipes[i][0], None, self.exp_list, self.test_policy, 0, SAVE_RS, True)) for i in range(2)]
        for worker in self.test_worker: worker.start()

        ######### TRAIN ROLLOUTS WITH ACTION NOISE ############
        self.rollout_pop = self.manager.list()
        for _ in range(self.args.num_action_rollouts): self.rollout_pop.append(models.Actor(args))
        self.task_pipes = [Pipe() for _ in range(args.num_action_rollouts)]
        self.result_pipes = [Pipe() for _ in range(args.num_action_rollouts)]
        self.train_workers = [Process(target=rollout_worker, args=(i, self.task_pipes[i][1], self.result_pipes[i][0], self.noise_gen[i], self.exp_list, self.rollout_pop,  0, SAVE_RS, True)) for i in range(args.num_action_rollouts)]
        for worker in self.train_workers: worker.start()


        self.best_policy = models.Actor(args)
        #### STATS AND TRACKING WHICH ROLLOUT IS DONE ######
        self.best_score = 0.0; self.buffer_added = 0; self.test_len = [None, None]; self.test_score = [None, None]; self.best_action_noise_score = 0.0; self.best_agent_scores = [0.0, 0.0]
        self.action_noise_scores = [0.0 for _ in range(args.num_action_rollouts)]
        self.test_eval_flag = [True, True]
        self.train_eval_flag = [True for _ in range(args.num_action_rollouts)]


    def add_experience(self, state, action, next_state, reward, done_probs, done):
        self.buffer_added += 1

        #RS to GENERATE SHAPED_REWARD
        shaped_r = np.zeros((1,1))
        if RS_PROPORTIONAL_SHAPE:
            if done_probs[0,0] != -1:
                shaped_r[0,0] = reward + (DONE_GAMMA ** done_probs[0,0]) * RS_DONE_W
        else:
            if done_probs[0,0] == 1:
                shaped_r[0,0] = RS_DONE_W

        if USE_BEHAVIOR_RS: shaped_r = rs.shaped_data(state, shaped_r, FOOTZ_W, KNEEFOOT_W, PELV_W, FOOTY_W, HEAD_W)

        self.replay_buffer.push(state, next_state, action, reward, done_probs, done, shaped_r)
        if self.buffer_added % self.replay_buffer.save_freq == 0: self.replay_buffer.save()


    def train(self, gen):

        if gen % 500 == 0: self.memory.load(self.args.data_folder) #Reload memoey
        if gen % 1000 == 0: self.best_policy.load_state_dict(torch.load('R_Skeleton/models/erl_best')) #Referesh best policy

        ########### START TEST ROLLOUT ##########
        if self.test_eval_flag[0]: #ALL DATA TEST
            self.test_eval_flag[0] = False
            self.rl_agent.actor.cpu()
            self.rl_agent.hard_update(self.test_policy[0], self.rl_agent.actor)
            self.rl_agent.actor.cuda()
            self.test_task_pipes[0][0].send(True)

        if self.test_eval_flag[1]:  # Iterative RL Learner (New RL Agent)
            self.test_eval_flag[1] = False
            self.new_rlagent.actor.cpu()
            self.new_rlagent.hard_update(self.test_policy[1], self.new_rlagent.actor)
            self.new_rlagent.actor.cuda()
            self.test_task_pipes[1][0].send(True)


        ########## START TRAIN (ACTION NOISE) ROLLOUTS ##########
        for i in range(self.args.num_action_rollouts):
            if self.train_eval_flag[i]:
                self.train_eval_flag[i] = False

                #HALF ACTION ROLLOUT ON BEST POLICY WHILE THE OTHER HALF ON CURRENT POLICY
                if i % 3 == 0:
                    self.rl_agent.hard_update(self.rollout_pop[i], self.best_policy)
                elif i % 3 == 1:
                    self.rl_agent.actor.cpu()
                    self.rl_agent.hard_update(self.rollout_pop[i], self.rl_agent.actor)
                    self.rl_agent.actor.cuda()
                else:
                    self.new_rlagent.actor.cpu()
                    self.new_rlagent.hard_update(self.rollout_pop[i], self.new_rlagent.actor)
                    self.new_rlagent.actor.cuda()

                self.task_pipes[i][0].send(True)




        ############################## RL LEANRING DURING POPULATION EVALUATION ##################
        #TRAIN FROM MEMORY (PREVIOUS DATA) for RL AGENT
        for _ in range(ITER_PER_EPOCH):
            s, ns, a, r, done = self.memory.sample(self.args.batch_size)
            s=s.cuda(); ns=ns.cuda(); a=a.cuda(); r=r.cuda(); done = done.cuda()
            self.rl_agent.update_parameters(s, ns, a, r, done, num_epoch=1)

        ##TRAIN FROM SELF_GENERATED DATA FOR NEW_RLAGENT
        if self.replay_buffer.__len__() > 100000:
            for _ in range(int(ITER_PER_EPOCH/2)):
                s, ns, a, shaped_r, done_dist = self.replay_buffer.sample(self.args.batch_size)

                done = (done_dist == 1).astype(float)
                s = torch.Tensor(s); ns = torch.Tensor(ns); a = torch.Tensor(a);
                shaped_r = torch.Tensor(shaped_r); done = torch.Tensor(done)

                s=s.cuda(); ns=ns.cuda(); a=a.cuda(); shaped_r=shaped_r.cuda(); done = done.cuda()
                self.new_rlagent.update_parameters(s, ns, a, shaped_r, done, num_epoch=1)

        ################################ EO RL LEARNING ########################


        if gen % 200 == 0 and QUICK_TEST != True and SAVE:
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
    fname = fname + str(parameters.gamma) + '_'
    if RS_PROPORTIONAL_SHAPE: fname = fname + 'RS_PROP' + str(DONE_GAMMA) + '_'
    fname + str(RS_DONE_W)
    if args.use_advantage: fname = fname + '_ADV'
    if USE_BEHAVIOR_RS:
        fname = fname + '_' + str(FOOTZ_W)+ '_' + str(KNEEFOOT_W)+ '_' + str(PELV_W)+ '_' + str(FOOTY_W)

    return fname


if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    #################################################### FILENAMES
    parameters.critic_fname = shape_filename(critic_fname, parameters)
    parameters.actor_fname = shape_filename(actor_fname, parameters)
    parameters.log_fname = shape_filename(log_fname, parameters)
    parameters.best_fname = shape_filename(best_fname, parameters)
    ####################################################


    frame_tracker = utils.Tracker(parameters.metric_save, [parameters.log_fname+'_1', parameters.log_fname+'_2'], '.csv')  # Initiate tracker
    ml_tracker = utils.Tracker(parameters.aux_save, [parameters.log_fname+'critic_loss', parameters.log_fname+'policy_loss'], '.csv')  # Initiate tracker

    #Initialize the environment

    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)
    agent = PG_ALGO(parameters) #Initialize the agent
    print('Running', algo,  ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim, 'using Data')

    time_start = time.time(); num_frames = 0.0

    for epoch in range(1, 1000000000): #Infinite generations
        gen_time = time.time()
        agent.train(epoch)
        print('Ep:', epoch, 'Score cur/best:', [pprint(score) for score in agent.test_score], pprint(agent.best_score),
              'Time:',pprint(time.time()-gen_time), 'Len', pprint(agent.test_len), 'Best_action_noise_score', pprint(agent.best_action_noise_score), 'Best_Agent_scores', [pprint(score) for score in agent.best_agent_scores])


        if epoch % 5 == 0: #Special Stats
            print()
            print('#Data_Created', agent.buffer_added, 'Q_Val Stats', pprint(list_mean(agent.rl_agent.q['min'])), pprint(list_mean(agent.rl_agent.q['max'])), pprint(list_mean(agent.rl_agent.q['mean'])),
              'Val Stats', pprint(list_mean(agent.rl_agent.val['min'])), pprint(list_mean(agent.rl_agent.val['max'])), pprint(list_mean(agent.rl_agent.val['mean'])))
            print()
            print ('Memory_size/mil', pprint(agent.memory.num_entries/1000000.0), 'Algo:', parameters.best_fname,
                   'Gamma', parameters.gamma,
                   'RS_PROP', RS_PROPORTIONAL_SHAPE,
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

        frame_tracker.update([agent.test_score[0], agent.test_score[1]], epoch)
        ml_tracker.update([agent.rl_agent.critic_loss['mean'][-1], agent.rl_agent.policy_loss['mean'][-1]], epoch)











