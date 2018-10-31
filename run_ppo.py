from core import ppo as ppo
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
parser.add_argument('-seed_policy', help='Where to seed from if any: none--> No seeding; no_entry --> R_PPO/models/erl_best', default='none')
parser.add_argument('-save_folder', help='Primary save folder to save logs, data and policies',  default='R_PPO')
parser.add_argument('-num_workers', type=int,  help='#Rollout workers',  default=8)
parser.add_argument('-shorts', type=str2bool,  help='#Short run',  default=False)


SEED = vars(parser.parse_args())['seed_policy']
SAVE_FOLDER = vars(parser.parse_args())['save_folder'] + '/'
NUM_WORKERS = vars(parser.parse_args())['num_workers']
USE_SYNTHETIC_TARGET = vars(parser.parse_args())['shorts']
XBIAS = False; ZBIAS = False; PHASE_LEN = 100


#MACROS
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

        self.seed = 7
        self.batch_size = 256 #Batch size for learning
        self.gamma = 0.99 #Discount rate
        self.init_w = False #Whether to initialize model weights using Kaimiling Ne

        self.state_dim = 415; self.action_dim = 19 #HARDCODED FOR THIS PROBLEM
        #Save Results
        if DIFFICULTY == 0: self.save_foldername = 'R_PPO/'
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

        self.critic_fname = 'ppo_vfunc'
        self.actor_fname = 'ppo'
        self.log_fname =  'ppo_log'
        self.best_fname = 'ppo_best'


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

class PPO_ALGO:
    """Policy Gradient Algorithm main object which carries out off-policy learning using policy gradient
       Encodes all functionalities for 1. TD3 2. DDPG 3.Trust-region TD3/DDPG 4. Advantage TD3/DDPG

            Parameters:
                args (int): Parameter class with all the parameters

            """

    def __init__(self, args):
        self.args = args
        self.rl_agent = ppo.PPO(args)


        #Use seed to bootstrap learning
        if SEED != 'none':
            try:
                self.rl_agent.actor.load_state_dict(torch.load(SEED))
                self.rl_agent.actor_target.load_state_dict(torch.load(SEED))

                print('Actor successfully loaded from ', SEED)
            except: print('Loading Actors failed')

            try:
                self.rl_agent.vfunc.load_state_dict(torch.load(args.model_save + self.args.critic_fname))
                print('VFunc successfully loaded from',args.model_save +  self.args.critic_fname)
            except: print ('Loading Vfunc failed')

        #Load to GPU
        #self.rl_agent.vfunc.cuda(); self.rl_agent.actor.cuda()

        self.replay_buffer = Buffer(save_freq=100000, save_folder=self.args.data_folder, capacity=1000000)
        self.noise_gen = OU_handle.get_list_generators(args.num_action_rollouts, args.action_dim)

        ######### Multiprocessing TOOLS #########
        self.manager = Manager()
        self.exp_list = self.manager.list() #Experience list stores experiences from all processes
        self.traj_list = self.manager.list() #Experience list stores experiences from all processes

        ######## TEST ROLLOUT POLICY ############
        self.test_policy = self.manager.list(); self.test_policy.append(models.Actor(args))
        self.test_task_pipes = [Pipe() for _ in range(1)]
        self.test_result_pipes = [Pipe() for _ in range(1)]
        self.test_worker = [Process(target=rollout_worker, args=(i, self.test_task_pipes[i][1], self.test_result_pipes[i][0], None, self.exp_list, self.test_policy, DIFFICULTY, True, True, USE_SYNTHETIC_TARGET, XBIAS, ZBIAS, PHASE_LEN)) for i in range(1)]
        for worker in self.test_worker: worker.start()

        ######### TRAIN ROLLOUTS WITH ACTION NOISE ############
        self.rollout_pop = self.manager.list()
        for _ in range(self.args.num_action_rollouts): self.rollout_pop.append(models.Actor(args))
        self.task_pipes = [Pipe() for _ in range(args.num_action_rollouts)]
        self.result_pipes = [Pipe() for _ in range(args.num_action_rollouts)]
        self.train_workers = [Process(target=rollout_worker, args=(i, self.task_pipes[i][1], self.result_pipes[i][0], self.noise_gen[i], self.exp_list, self.rollout_pop,  DIFFICULTY, True, True,USE_SYNTHETIC_TARGET, XBIAS, ZBIAS, PHASE_LEN, self.traj_list)) for i in range(args.num_action_rollouts)]
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


        self.replay_buffer.push(state, next_state, action, reward, done_dist, done, shaped_r)
        if self.buffer_added % self.replay_buffer.save_freq == 0: self.replay_buffer.save()


    def train(self, gen):
        """Main training loop to do rollouts and run policy gradients

            Parameters:
                gen (int): Current epoch of training

            Returns:
                None
        """


        ########### START TEST ROLLOUT ##########
        if self.test_eval_flag[0]: #ALL DATA TEST
            self.test_eval_flag[0] = False
            #self.rl_agent.actor.cpu()
            self.rl_agent.hard_update(self.test_policy[0], self.rl_agent.actor)
            #self.rl_agent.actor.cuda()
            self.test_task_pipes[0][0].send(True)


        ########## START TRAIN (ACTION NOISE) ROLLOUTS ##########
        for i in range(self.args.num_action_rollouts):
            if self.train_eval_flag[i]:
                self.train_eval_flag[i] = False
                self.rl_agent.hard_update(self.rollout_pop[i], self.rl_agent.actor)
                self.task_pipes[i][0].send(True)



        ####################### HARD-JOIN ROLLOUTS AND COMPUTE GAE ############################
        num_finished_rollouts = 0
        all_values = []
        all_states = []
        all_actions = []
        all_advs = []
        all_returns = []

        while True:
            if len(self.traj_list) != 0: #Worker done
                num_finished_rollouts += 1
                trajectory = self.traj_list.pop()

                #Compute GAE
                states, actions, values, returns = self.rl_agent.compute_gae(trajectory)
                advantages = [r-v for r, v in zip(returns,values)]

                #Store all necessary info to temp buffer
                for s, a, v, r, adv in zip(states, actions, values, returns, advantages):
                    all_states.append(s)
                    all_values.append(v)
                    all_actions.append(a)
                    all_returns.append(r)
                    all_advs.append(adv)


            # Soft-join (50%)
            if num_finished_rollouts == self.args.num_action_rollouts: break

        #Update PP0
        mini_batch_size = int(len(all_states)/8) * 2
        self.rl_agent.update_parameters(torch.cat(all_states), torch.cat(all_actions), None, torch.cat(all_returns), torch.cat(all_advs), mini_batch_size=mini_batch_size)



        ################################ EO POLICY GRaDIENT ########################

        #Save critic periodically
        if gen % 200 == 0:
            #torch.save(self.rl_agent.actor.state_dict(), parameters.rl_models + actor_fname)
            torch.save(self.rl_agent.vfunc.state_dict(), parameters.model_save + self.args.critic_fname)
            print("Critic Saved periodically")


        ##### PROCESS TEST ROLLOUT ##########
        for i in range(1):
            if self.test_result_pipes[i][1].poll():
                entry = self.test_result_pipes[i][1].recv()
                self.test_eval_flag[i] = True
                self.test_score[i] = entry[1]
                self.test_len[i] = entry[2]

                if self.test_score[i] > self.best_score:
                    self.best_score = self.test_score[i]
                    self.best_agent_scores[i] = self.test_score[i]
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
    agent = PPO_ALGO(parameters)
    print('Running PPO',  ' State_dim:', parameters.state_dim, ' Action_dim:', parameters.action_dim, 'for', 'Round 1' if DIFFICULTY == 0 else 'Round 2')
    time_start = time.time(); num_frames = 0.0

    ###### TRAINING LOOP ########
    for epoch in range(1, 1000000000): #RUN VIRTUALLY FOREVER
        gen_time = time.time()

        #ONE EPOCH OF TRAINING
        agent.train(epoch)

        #PRINT PROGRESS
        print('Ep:', epoch, 'Score cur/best:', [pprint(score) for score in agent.test_score], pprint(agent.best_score),
              'Time:',pprint(time.time()-gen_time), 'Len', pprint(agent.test_len[0]), 'Noisy_best', pprint(agent.best_action_noise_score), 'Best_scores', [pprint(score) for score in agent.best_agent_scores])



        #Update score to trackers
        frame_tracker.update([agent.test_score[0], agent.test_score[1]], epoch)
        try: ml_tracker.update([agent.rl_agent.critic_loss['mean'][-1], agent.rl_agent.policy_loss['mean'][-1]], epoch)
        except: None










