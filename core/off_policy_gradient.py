import torch
import torch.nn as nn
from torch.optim import Adam
import torch.nn.functional as F
import numpy as np
from core import mod_utils as utils
from core.models import Actor


class Critic(nn.Module):

    def __init__(self, args):
        super(Critic, self).__init__()
        self.args = args
        l1 = 256; l2 = 256; l3 = 128

        # Construct Hidden Layer 1 with state
        self.f1_state = nn.Linear(args.state_dim, l1)
        #self.f1_lin_state = nn.Linear(args.state_dim, l1)

        # Construct Hidden Layer 1 with action
        self.f1_action = nn.Linear(args.action_dim, int(l1/2))
        #self.f1_lin_action = nn.Linear(args.action_dim, int(l1/2))

        self.ln1 = nn.LayerNorm(384)

        #Hidden Layer 2
        self.f2 = nn.Linear(384, l2)
        #self.f2_lin = nn.Linear(l1*3, l2)
        self.ln2 = nn.LayerNorm(l2)

        ############### Q HEAD SPLITS FROM HERE ON ##################
        #Hidden Layer 3
        self.f3 = nn.Linear(l2, l3)
        #self.f3_lin = nn.Linear(l2*2, l3)
        self.ln3 = nn.LayerNorm(l3)

        self.f3_2 = nn.Linear(l2, l3)
        #self.f3_lin_2 = nn.Linear(l2*2, l3)
        self.ln3_2 = nn.LayerNorm(l3)

        #Out
        self.w_out = nn.Linear(l3, 1)
        self.w_out_2 = nn.Linear(l3, 1)



    def forward(self, input, action):

        #Hidden Layer 1 (Input Interfaces)
        #State
        out_state = F.elu(self.f1_state(input))
        #lin_out_state = self.f1_lin_state(input)
        #out_state = torch.cat([nl_out_state, lin_out_state], 1)


        #Action
        out_action = F.elu(self.f1_action(action))
        #lin_out_action = self.f1_lin_action(action)
        #out_action = torch.cat([nl_out_action, lin_out_action], 1)

        #Combined
        out = torch.cat([out_state, out_action], 1)
        out = self.ln1(out)

        #Hidden Layer 2
        out = F.elu(self.f2(out))
        #lin_out = self.f2_lin(out)
        #out = torch.cat([nl_out, lin_out], 1)
        out = self.ln2(out)

        ############# Q HEADS SPLIT ##############

        #Hidden Layer 3
        out1 = F.elu(self.f3(out))
        #lin_out1 = self.f3_lin(out)
        #out1 = torch.cat([nl_out1, lin_out1], 1)
        out1 = self.ln3(out1)

        out2 = F.elu(self.f3(out))
        #lin_out2 = self.f3_lin(out)
        #out2 = torch.cat([nl_out2, lin_out2], 1)
        out2 = self.ln3_2(out2)

        # Output interface
        return self.w_out(out1), self.w_out_2(out2)

class TD3_DDPG(object):
    def __init__(self, args):

        self.args = args
        self.algo = args.algo

        self.actor = Actor(args)
        if args.init_w: self.actor.apply(utils.init_weights)
        self.actor_target = Actor(args)
        self.actor_optim = Adam(self.actor.parameters(), lr=5e-4)


        self.critic = Critic(args)
        if args.init_w: self.critic.apply(utils.init_weights)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=5e-3)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.MSELoss()

        self.hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        self.hard_update(self.critic_target, self.critic)
        self.actor_target.cuda(); self.critic_target.cuda(); self.actor.cuda(); self.critic.cuda()
        self.num_critic_updates = 0

        #Statistics Tracker
        self.action_loss = []
        self.policy_loss = []
        self.critic_loss = []
        self.critic_w_mag = []
        self.actor_w_mag = []


    def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, num_epoch=5):
        if isinstance(state_batch, list): state_batch = torch.cat(state_batch); next_state_batch = torch.cat(next_state_batch); action_batch = torch.cat(action_batch); reward_batch = torch.cat(reward_batch)
        for _ in range(num_epoch):
            #Critic Update
            with torch.no_grad():
                policy_noise = np.random.normal(0, self.args.policy_noise, (action_batch.size()[0], action_batch.size()[1]))
                policy_noise = torch.clamp(torch.Tensor(policy_noise), -self.args.policy_noise_clip, self.args.policy_noise_clip)
                next_action_batch = self.actor_target.forward(next_state_batch) + policy_noise.cuda()
                next_action_batch = torch.clamp(next_action_batch, 0, 1)


                q1, q2 = self.critic_target.forward(next_state_batch, next_action_batch)
                if self.algo == 'TD3' or self.algo == 'TD3_actor_min': next_q = torch.min(q1, q2)
                elif self.algo == 'DDPG': next_q = q1
                elif self.algo == 'TD3_max': next_q = torch.max(q1, q2)

                target_q = reward_batch + (self.gamma * next_q)

            self.critic_optim.zero_grad()
            current_q1, current_q2 = self.critic.forward((state_batch), (action_batch))
            dt = self.loss(current_q1, target_q)
            if self.algo == 'TD3' or self.algo == 'TD3_max': dt = dt + self.loss(current_q2, target_q)
            dt.backward()
            self.critic_loss.append(dt.item())
            #nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
            self.critic_optim.step()
            self.num_critic_updates += 1


            #Delayed Actor Update
            if self.num_critic_updates % self.args.policy_ups_freq == 0:
                actor_actions = self.actor.forward(state_batch)
                Q1, Q2 = self.critic.forward(state_batch, actor_actions)
                policy_loss = -Q1.mean()
                if self.algo == 'TD3_actor_min': policy_loss = -torch.min(Q1, Q2).mean()


                self.actor_optim.zero_grad()
                policy_loss.backward(retain_graph=True)
                self.policy_loss.append(policy_loss.item())
                #nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
                if self.args.action_loss:
                    action_loss = torch.abs(actor_actions-0.5).mean()
                    action_loss = action_loss * self.args.action_loss_w
                    action_loss.backward()
                    self.action_loss.append(action_loss.item())
                    if self.action_loss[-1] > self.policy_loss[-1]: self.args.action_loss_w *= 0.9 #Decay action_w loss if action loss is larger than policy gradient loss
                self.actor_optim.step()

                self.soft_update(self.actor_target, self.actor, self.tau)
                self.soft_update(self.critic_target, self.critic, self.tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)








