import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import random
from core import mod_utils as utils
from core.models import Actor





class ValueFunc(nn.Module):

    """Critic model

        Parameters:
              args (object): Parameter class

    """

    def __init__(self, args):
        super(ValueFunc, self).__init__()
        self.args = args
        l1 = 400; l2 = 400; l3 = 300


        #Value Head
        self.val_f1 = nn.Linear(args.state_dim, l1)
        self.val_ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.val_f2 = nn.Linear(l1, l2)
        #self.f2_lin = nn.Linear(l1*3, l2)
        self.val_ln2 = nn.LayerNorm(l2)

        # Hidden Layer 2
        self.val_f3 = nn.Linear(l2, l3)
        # self.f2_lin = nn.Linear(l1*3, l2)
        self.val_ln3 = nn.LayerNorm(l3)

        # Hidden Layer 2
        self.val_out = nn.Linear(l3, 1)



    def forward(self, input):
        """Method to forward propagate through the critic's graph

             Parameters:
                   input (tensor): states

             Returns:
                   action (tensor): actions


         """

        ################################ VALUE HEAD ###################
        val = self.val_ln1(F.elu(self.val_f1(input)))
        val = self.val_ln2(F.elu(self.val_f2(val)))
        val = self.val_ln3(F.elu(self.val_f3(val)))
        val = self.val_out(val)


        # Output interface
        return val


class PPO(object):
    """Classes implementing TD3 and DDPG off-policy learners

         Parameters:
               args (object): Parameter class


     """

    def __init__(self, args):

        self.args = args

        self.actor = Actor(args)
        if args.init_w: self.actor.apply(utils.init_weights)
        self.actor_target = Actor(args)
        self.optim = Adam(self.actor.parameters(), lr=5e-4)


        self.vfunc = ValueFunc(args)
        if args.init_w: self.vfunc.apply(utils.init_weights)

        self.gamma = args.gamma
        self.loss = nn.SmoothL1Loss()#nn.MSELoss()

        #self.actor.cuda(); self.vfunc.cuda()
        self.num_critic_updates = 0

        #Statistics Tracker
        self.action_loss = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.policy_loss = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.critic_loss = {'mean':[]}
        self.q = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.val = {'min':[], 'max': [], 'mean':[], 'std':[]}


    def compute_gae(self, trajectory, gamma=0.99, tau=0.95):
        with torch.no_grad():
            values = [];
            next_values = [];
            rewards = [];
            masks = [];
            states = []
            actions = []

            for entry in trajectory:
                states.append(torch.tensor(entry[0]))
                actions.append(torch.tensor(entry[1]))
                values.append(self.vfunc(torch.Tensor(entry[0])))
                rewards.append(torch.Tensor(entry[3]))
                masks.append(torch.Tensor(entry[5]))
            values.append(self.vfunc(torch.Tensor(entry[2])))

            gae = 0.0
            returns = []
            for step in reversed(range(len(rewards))):
                delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
                gae = delta + gamma * tau * masks[step] * gae
                returns.insert(0, gae + values[step])

        return states, actions, values, returns



    def compute_stats(self, tensor, tracker):
        """Computes stats from intermediate tensors

             Parameters:
                   tensor (tensor): tensor
                   tracker (object): logger

             Returns:
                   None


         """
        tracker['min'].append(torch.min(tensor).item())
        tracker['max'].append(torch.max(tensor).item())
        tracker['mean'].append(torch.mean(tensor).item())
        tracker['mean'].append(torch.mean(tensor).item())

    def update_parameters(self, states, actions, log_probs, returns, advantages, ppo_epochs=4, mini_batch_size=128, clip_param=0.2):
        """Runs a step of Bellman upodate and policy gradient using a batch of experiences

             Parameters:
                  state_batch (tensor): Current States
                  next_state_batch (tensor): Next States
                  action_batch (tensor): Actions
                  reward_batch (tensor): Rewards
                  done_batch (tensor): Done batch
                  num_epoch (int): Number of learning iteration to run with the same data

             Returns:
                   None

         """


        for _ in range(ppo_epochs):
            ind = random.sample(range(len(states)), mini_batch_size)
            mini_s = states[ind]; mini_a = actions[ind]; mini_ret = returns[ind]; mini_adv = advantages[ind]


            #PPO Update
            new_action, value = self.actor(mini_s), self.vfunc(mini_s)


            ratio = mini_a - new_action
            surr1 = ratio * mini_adv
            surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * mini_adv

            actor_loss = - torch.min(surr1, surr2).mean()
            critic_loss = (mini_ret - value).pow(2).mean()

            loss = 0.5 * critic_loss + actor_loss

            self.optim.zero_grad()
            loss.backward()
            self.optim.step()
















    def soft_update(self, target, source, tau):
        """Soft update from target network to source

            Parameters:
                  target (object): A pytorch model
                  source (object): A pytorch model
                  tau (float): Tau parameter

            Returns:
                None

        """

        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        """Hard update (clone) from target network to source

            Parameters:
                  target (object): A pytorch model
                  source (object): A pytorch model

            Returns:
                None
        """

        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)








