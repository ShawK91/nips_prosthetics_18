import torch
import numpy as np, os, time, random, sys
from core import mod_utils as utils
from osim.env import ProstheticsEnv
from multiprocessing import Process, Queue
from core import off_policy_gradient as pg
from core import models
os.environ["CUDA_VISIBLE_DEVICES"]='3'


SEED = True
RS_FINAL_STATE = False
RS_FITNESS = False
FIT_PRIORITAZION = True
algo = 'TD3'     #1. TD3
                 #2. DDPG
                 #3. TD3_max   - TD3 but using max rather than min

#FAIRLY STATIC
QUICK_TEST = False
IEPOCHS_PER_BATCH = 5
ITER_PER_EPOCH = 200
SAVE_THRESHOLD = 500
NORMALIZE = False
if QUICK_TEST:
    IEPOCHS_PER_BATCH = 1
    ITER_PER_EPOCH = 10
    SEED = False

#Algo Choice process
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

if RS_FINAL_STATE:
    critic_fname = 'RSFS_' + critic_fname
    actor_fname = 'RSFS_' + actor_fname
    log_fname = 'RSFS_' + log_fname
    best_fname = 'RSFS_' + best_fname

if RS_FITNESS:
    critic_fname = 'RSFIT_' + critic_fname
    actor_fname = 'RSFIT_' + actor_fname
    log_fname = 'RSFIT_' + log_fname
    best_fname = 'RSFIT_' + best_fname

if FIT_PRIORITAZION:
    critic_fname = 'FITPR_' + critic_fname
    actor_fname = 'FITPR_' + actor_fname
    log_fname = 'FITPR_' + log_fname
    best_fname = 'FITPR_' + best_fname


class Parameters:
    def __init__(self):

        #MetaParams
        self.is_cuda= True
        self.algo = algo

        #RL params
        self.seed = 4
        self.batch_size = 512
        self.action_loss = True
        self.action_loss_w = 0.1

        #TD3
        self.gamma = 0.9999; self.tau = 0.001
        self.init_w = False
        self.policy_ups_freq = 2
        self.policy_noise = 0.05
        self.policy_noise_clip = 0.1

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


def evaluate(task_q, res_q, env, noise, skip_step=1):

    while True:
        task = task_q.get()
        net = task[1]

        state = env.reset(); fitness = 0.0
        state.append(0)
        state = utils.to_tensor(np.array(state)).unsqueeze(0);
        last_action = None
        for frame in range(1, 1000000000): #Infinite
            if frame % skip_step == 0 or frame == 1:
                action = net.forward(state)
                last_action = action
            else: action = last_action

            action = utils.to_numpy(action.cpu())
            if noise != None: action += noise.noise()

            next_state, reward, done, info = env.step(action.flatten())  # Simulate one step in environment
            next_state.append(frame)
            next_state = utils.to_tensor(np.array(next_state)).unsqueeze(0)
            fitness += reward

            state = next_state
            if done: break


        res_q.put([fitness, frame])


class Buffer():

    def __init__(self, capacity):
        self.capacity = capacity
        self.s = []; self.ns = []; self.a = []; self.r = []; self.fit =[]; self.done = []
        self.position = 0; self.fit_pr = None

    def sample(self, batch_size):
        if FIT_PRIORITAZION:
            fit_batch_size = int(0.2 * batch_size); uniform_batch_size = batch_size-fit_batch_size
            fit_ind = random.sample(self.fit_pr, fit_batch_size)
            uniform_ind = random.sample(range(self.__len__()), uniform_batch_size)
            ind = fit_ind + uniform_ind

        else: #Unbiased Sample
            ind = random.sample(range(self.__len__()), batch_size)

        return self.s[ind], self.ns[ind], self.a[ind], self.r[ind]

    def __len__(self):
        return self.position

    def load(self, data_folder):

        ######## READ DATA #########
        list_files = os.listdir(data_folder)
        if len(list_files) != 0: #Reset everything
            self.s = []; self.ns = []; self.a = []; self.r = []; self.fit =[]; self.done = []; self.position = 0
            self.fit_pr = []
        for file in list_files:
            data = np.load(data_folder + file)

            for s, ns, a, r, fit, done in zip(data['state'], data['next_state'], data['action'], data['reward'], data['fitness'], data['done']):

                #REWARDS SHAPING
                if RS_FINAL_STATE:
                    if s[-1] < 299 and done:
                        r[0] = r[0] - 100
                if RS_FITNESS:
                    r[0] = r[0] + fit[0]/1000

                self.s.append(torch.Tensor(s).unsqueeze(0))
                self.ns.append(torch.Tensor(ns).unsqueeze(0))
                self.a.append(torch.Tensor(a).unsqueeze(0))
                self.r.append(torch.Tensor(r).unsqueeze(0))
                self.fit.append(torch.Tensor(fit).unsqueeze(0))
                self.done.append(torch.Tensor(done).unsqueeze(0))

                self.position+=1
                if QUICK_TEST and self.position > 1000: break #QUICK TEST

            print('BUFFER LOADED WITH', self.position, 'SAMPLES FROM WITH TAG', file)

        self.s = torch.cat(self.s).pin_memory()
        self.ns = torch.cat(self.ns).pin_memory()
        self.a = torch.cat(self.a).pin_memory()
        self.r = torch.cat(self.r).pin_memory()
        self.fit = torch.cat(self.fit).pin_memory()
        self.done = torch.cat(self.done).pin_memory()

        if FIT_PRIORITAZION:
            pr_index = (torch.sort(self.fit, 0, descending=True)[1])
            self.fit_pr = pr_index[0:int(0.2*self.position)].data.numpy().flatten().tolist()
            #print (self.fit_pr)
        # #Normalize states and next_states
        # if NORMALIZE:
        #     min_T = torch.min(self.s, 0)[0]
        #     self.s = self.s-min_T
        #     self.ns = self.ns-min_T
        #     max_T = torch.max(self.s, 0)[0] + 0.123456
        #     self.s = self.s/max_T; self.ns = self.ns/max_T
        # else:
        #     min_T = torch.zeros(158)
        #     max_T = torch.ones(158)



class PG_ALGO:
    def __init__(self, args, env):
        self.args = args; self.env = env
        self.rl_agent = pg.TD3_DDPG(args)

        if SEED != None:
            try:
                self.rl_agent.actor.load_state_dict(torch.load(args.rl_models + best_fname))
                self.rl_agent.actor_target.load_state_dict(torch.load(args.rl_models + best_fname))
                print('Actor successfully loaded from the R_Skeleton folder for ', algo)
            except: print('Loading Actors failed')
            try:
                self.rl_agent.critic_target.load_state_dict(torch.load(args.model_save + critic_fname))
                self.rl_agent.critic.load_state_dict(torch.load(args.model_save + critic_fname))
                print('Critic successfully loaded from the R_Skeleton folder for ', algo)
            except: print ('Loading Critics failed')



        self.rl_agent.actor_target.cuda(); self.rl_agent.critic_target.cuda(); self.rl_agent.critic.cuda(); self.rl_agent.actor.cuda()

        ############ LOAD DATA AND CONSTRUCT GENERATORS ########
        self.buffer = Buffer(10000000)
        self.buffer.load(self.args.data_folder)


        #RL ROLLOUT PROCESSOR
        #self.res_list = Manager().list(); self.job_count = 0
        self.rl_task_q = Queue(); self.rl_res_q = Queue()
        self.rl_worker = [Process(target=evaluate, args=(self.rl_task_q, self.rl_res_q, self.env, None))]
        self.rl_worker[0].start()
        self.rollout_policy = models.Actor(args)
        self.best_policy = models.Actor(args); self.best_score = 0.0
        self.job_count = 0; self.job_done = 0
        self.test_lens = []

    def train(self, gen, frame_tracker):

        #Start RL test Rollout
        if self.job_count <= self.job_done:
            self.rollout_policy.cuda()
            self.rl_agent.hard_update(self.rollout_policy, self.rl_agent.actor)
            self.rollout_policy.cpu()
            self.rl_task_q.put([-1, self.rollout_policy])
            self.job_count += 1

        ############################## RL LEANRING DURING POPULATION EVALUATION ##################
        #RL TRAIN
        for _ in range(ITER_PER_EPOCH):
            s, ns, a, r = self.buffer.sample(self.args.batch_size)
            s=s.cuda(); ns=ns.cuda(); a=a.cuda(); r=r.cuda()
            self.rl_agent.update_parameters(s, ns, a, r, num_epoch=IEPOCHS_PER_BATCH)


        ################################ EO RL LEARNING ########################

        if gen % 200 == 0:
            #torch.save(self.rl_agent.actor.state_dict(), parameters.rl_models + actor_fname)
            torch.save(self.rl_agent.critic.state_dict(), parameters.model_save + critic_fname)
            print("Critic Saved periodically")

        #Periodically reload buffer
        if gen % 10000 == 0: self.buffer.load(self.args.data_folder)

        # Join RL Test
        if self.rl_res_q.empty():
            test_score = None
        else:
            entry = self.rl_res_q.get()
            self.job_done += 1
            self.test_lens.append(entry[1])
            test_score = entry[0]
            frame_tracker.update([test_score], gen)
            if test_score > self.best_score:
                self.best_score = test_score
                self.rl_agent.hard_update(self.best_policy, self.rollout_policy)
                if self.best_score > SAVE_THRESHOLD:
                    torch.save(self.best_policy.state_dict(), parameters.rl_models + best_fname)
                    print("Best policy saved with score ", self.best_score)


        return test_score

if __name__ == "__main__":
    parameters = Parameters()  # Create the Parameters class
    frame_tracker = utils.Tracker(parameters.metric_save, [log_fname], '.csv')  # Initiate tracker

    #Initialize the environment
    env = ProstheticsEnv(visualize=False)
    torch.manual_seed(parameters.seed); np.random.seed(parameters.seed); random.seed(parameters.seed)
    agent = PG_ALGO(parameters, env) #Initialize the agent
    print('Running', algo,  ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim, 'using Data')

    time_start = time.time(); num_frames = 0.0
    for epoch in range(1, 1000000000): #Infinite generations
        gen_time = time.time()
        test_score = agent.train(epoch, frame_tracker)
        print('Ep:', epoch,  'Avg:','%.2f'%frame_tracker.all_tracker[0][1], 'Score', '%.2f'%frame_tracker.all_tracker[0][0][-1] if len(frame_tracker.all_tracker[0][0]) != 0 else None,
              'Time:','%.1f'%(time.time()-gen_time), 'Test_len', '%.1f'%agent.test_lens[-1] if len(agent.test_lens) != 0 else None,  'Best_Score', '%.1f'%agent.best_score)
        if epoch % 5 == 0: #Special Stats
            print()
            try: print ('RL_Inspector:', 'Critic_loss:', '%.2f'%utils.list_mean(agent.rl_agent.critic_loss), 'Policy_loss:', '%.2f'%utils.list_mean(agent.rl_agent.policy_loss), 'Action_loss:', '%.2f'%utils.list_mean(agent.rl_agent.action_loss))
            except: None
            print ('Buffer_size/mil', '%.1f'%(agent.buffer.position/1000000.0), 'Algo:', best_fname)
            print()












