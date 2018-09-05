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
        l1 = 400; l2 = 400; l3 = 300

        # Construct Hidden Layer 1 with state
        self.f1_state = nn.Linear(args.state_dim, l1)
        #self.f1_lin_state = nn.Linear(args.state_dim, l1)

        # Construct Hidden Layer 1 with action
        self.f1_action = nn.Linear(args.action_dim, int(l1/4))
        #self.f1_lin_action = nn.Linear(args.action_dim, int(l1/2))

        self.ln1 = nn.LayerNorm(500)

        #Hidden Layer 2
        self.f2 = nn.Linear(500, l2)
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
        out1 = self.w_out(out1)

        out2 = F.elu(self.f3(out))
        #lin_out2 = self.f3_lin(out)
        #out2 = torch.cat([nl_out2, lin_out2], 1)
        out2 = self.ln3_2(out2)
        out2 = self.w_out_2(out2)

        ################################ VALUE HEAD ###################
        val = self.val_ln1(F.elu(self.val_f1(input)))
        val = self.val_ln2(F.elu(self.val_f2(val)))
        val = self.val_ln3(F.elu(self.val_f3(val)))
        val = self.val_out(val)


        # Output interface
        return out1, out2, val



class TD3_DDPG(object):
    def __init__(self, args):

        self.args = args
        self.algo = args.algo

        self.actor = Actor(args)
        if args.init_w: self.actor.apply(utils.init_weights)
        self.actor_target = Actor(args)
        self.actor_optim = Adam(self.actor.parameters(), lr=1e-4)


        self.critic = Critic(args)
        if args.init_w: self.critic.apply(utils.init_weights)
        self.critic_target = Critic(args)
        self.critic_optim = Adam(self.critic.parameters(), lr=1e-3)

        self.gamma = args.gamma; self.tau = self.args.tau
        self.loss = nn.SmoothL1Loss()#nn.MSELoss()

        self.hard_update(self.actor_target, self.actor)  # Make sure target is with the same weight
        self.hard_update(self.critic_target, self.critic)
        self.actor_target.cuda(); self.critic_target.cuda(); self.actor.cuda(); self.critic.cuda()
        self.num_critic_updates = 0

        #Statistics Tracker
        self.action_loss = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.policy_loss = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.critic_loss = {'mean':[]}
        self.q = {'min':[], 'max': [], 'mean':[], 'std':[]}
        self.val = {'min':[], 'max': [], 'mean':[], 'std':[]}

    def compute_stats(self, tensor, tracker):
        tracker['min'].append(torch.min(tensor).item())
        tracker['max'].append(torch.max(tensor).item())
        tracker['mean'].append(torch.mean(tensor).item())
        tracker['mean'].append(torch.mean(tensor).item())


    def update_parameters(self, state_batch, next_state_batch, action_batch, reward_batch, done_batch, num_epoch=1):
        if isinstance(state_batch, list): state_batch = torch.cat(state_batch); next_state_batch = torch.cat(next_state_batch); action_batch = torch.cat(action_batch); reward_batch = torch.cat(reward_batch). done_batch = torch.cat(done_batch)

        for _ in range(num_epoch):
            ########### CRITIC UPDATE ####################

            #Compute next q-val, next_v and target
            with torch.no_grad():
                #Policy Noise
                policy_noise = np.random.normal(0, self.args.policy_noise, (action_batch.size()[0], action_batch.size()[1]))
                policy_noise = torch.clamp(torch.Tensor(policy_noise), -self.args.policy_noise_clip, self.args.policy_noise_clip)

                #Compute next action_bacth
                next_action_batch = self.actor_target.forward(next_state_batch) + policy_noise.cuda()
                next_action_batch = torch.clamp(next_action_batch, 0, 1)

                #Compute Q-val and value of next state masking by done
                q1, q2, next_val = self.critic_target.forward(next_state_batch, next_action_batch)
                if self.args.use_done_mask:
                    q1 = (1 - done_batch) * q1
                    q2 = (1 - done_batch) * q2
                    next_val = (1 - done_batch) * next_val

                #Clamp Q-vals
                if self.args.q_clamp != None:
                    q1 = torch.clamp(q1, -self.args.q_clamp, self.args.q_clamp)
                    q1 = torch.clamp(q2, -self.args.q_clamp, self.args.q_clamp)

                #Select which q to use as next-q (depends on algo)
                if self.algo == 'TD3' or self.algo == 'TD3_actor_min': next_q = torch.min(q1, q2)
                elif self.algo == 'DDPG': next_q = q1
                elif self.algo == 'TD3_max': next_q = torch.max(q1, q2)

                #Compute target q and target val
                target_q = reward_batch + (self.gamma * next_q)
                if self.args.use_advantage: target_val = reward_batch + (self.gamma * next_val)


            self.critic_optim.zero_grad()
            current_q1, current_q2, current_val = self.critic.forward((state_batch), (action_batch))
            self.compute_stats(current_q1, self.q)

            dt = self.loss(current_q1, target_q)
            if self.args.use_advantage:
                dt = dt + self.loss(current_val, target_val)
                self.compute_stats(current_val, self.val)

            if self.algo == 'TD3' or self.algo == 'TD3_max': dt = dt + self.loss(current_q2, target_q)
            self.critic_loss['mean'].append(dt.item())

            if self.args.critic_constraint: dt = dt * (abs(self.args.critic_constraint_w / dt.item()))
            dt.backward()

            self.critic_optim.step()
            self.num_critic_updates += 1


            #Delayed Actor Update
            if self.num_critic_updates % self.args.policy_ups_freq == 0:
                actor_actions = self.actor.forward(state_batch)
                Q1, Q2, val = self.critic.forward(state_batch, actor_actions)

                if self.args.use_advantage: policy_loss = -(Q1 - val)
                else: policy_loss = -Q1
                self.compute_stats(policy_loss,self.policy_loss)
                policy_loss = policy_loss.mean()

                self.actor_optim.zero_grad()
                if self.args.policy_constraint: policy_loss = policy_loss * (abs(self.args.policy_constraint_w / policy_loss.item()))
                policy_loss.backward(retain_graph=True)
                #nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
                if self.args.action_loss:
                    action_loss = torch.abs(actor_actions-0.5)
                    self.compute_stats(action_loss, self.action_loss)
                    action_loss = action_loss.mean() * self.args.action_loss_w
                    action_loss.backward()
                    #if self.action_loss[-1] > self.policy_loss[-1]: self.args.action_loss_w *= 0.9 #Decay action_w loss if action loss is larger than policy gradient loss
                self.actor_optim.step()


            if self.args.hard_update:
                if self.num_critic_updates % self.args.hard_update_freq == 0:
                    if self.num_critic_updates % self.args.policy_ups_freq == 0: self.hard_update(self.actor_target, self.actor)
                    self.hard_update(self.critic_target, self.critic)

            else:
                if self.num_critic_updates % self.args.policy_ups_freq == 0: self.soft_update(self.actor_target, self.actor, self.tau)
                self.soft_update(self.critic_target, self.critic, self.tau)

    def soft_update(self, target, source, tau):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)

    def hard_update(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)








