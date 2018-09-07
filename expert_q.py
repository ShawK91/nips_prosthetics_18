import numpy as np
import torch, time
from core.models import Actor
from core.off_policy_gradient import Critic
from core import mod_utils as utils
from core.env_wrapper import EnvironmentWrapper



CRITIC_DIR = 'models_repo/critics/'
ACTOR_DIR = 'models_repo/actors/'

#GIVEN
DIFFICULTY = 0
FRAMESKIP = 5
XNORM = True

class Parameters:
    def __init__(self):
        self.state_dim = 415; self.action_dim = 19

class Expert_Q:
    def __init__(self, sample_budget):
        self.sample_budget = sample_budget
        dummy_args = Parameters()
        #Load all Critics
        critic_template = Critic(dummy_args)
        self.critic_ensemble = utils.load_all_models_dir(CRITIC_DIR, critic_template)

        #Load all Actors
        actor_template = Actor(dummy_args)
        self.actor_ensemble = utils.load_all_models_dir(ACTOR_DIR, actor_template)


    def take_action(self, state):
        state = utils.to_tensor(np.array(state)).unsqueeze(0)

        #Compute the actions suggested by the actors
        seed_actions = [actor.forward(state) for actor in self.actor_ensemble]

        #Perturb the seed actions to get the action samples
        action_samples = self.perturb_action(seed_actions)

        #Score and pick action using critic ensemble
        action = self.pick_actions(state, action_samples)
        return action


    def perturb_action(self, seed_action):
        pertubations_each = int(self.sample_budget / len(seed_action))

        action_samples = []
        for a in seed_action:
            a = a.repeat(pertubations_each, 1)
            noise = torch.Tensor(np.random.normal(0, 0.5, (pertubations_each, 19)))
            action_samples.append(torch.clamp(a+noise, 0, 1))
        return torch.cat(action_samples)


    def pick_actions(self, state, action_samples):
        state = state.repeat(len(action_samples), 1)

        all_q = []
        for critic in self.critic_ensemble:
            q1, q2, _ = critic.forward(state, action_samples)
            all_q.append(q1)

        avg_q = torch.cat(all_q, 1)
        avg_q = avg_q.mean(1)

        action = action_samples[torch.argmax(avg_q)]

        return action.detach().numpy()





env = EnvironmentWrapper(difficulty=DIFFICULTY, frameskip=FRAMESKIP, x_norm=XNORM)
observation = env.reset()

sim_start = time.time()
total_rew = 0.0; step  = 0; exit = False; total_steps = 0; total_score = 0.0; all_fit = []; all_len = []
expert = Expert_Q(10000)

while True:
    action = expert.take_action(observation)

    [observation, reward, done, info] = env.step(action)
    total_rew += reward; step+=1; total_steps+=1; total_score+=reward

    print('Steps', step, 'Rew', '%.2f'%reward, 'Total_Reward', '%.2f'%total_rew, 'Final_Score', '%.2f'%total_score,'Pelvis_pos', '%.2f'%env.pelvis_pos, 'Pelvis_vel', '%.2f'%env.pelvis_vel,
          'FITNESSES', ['%.2f'%f for f in all_fit], 'LENS', all_len, 'File', FRAMESKIP, 'X_NORM', XNORM)
    next_obs_dict = env.env.get_state_desc()
    knee_pos = [next_obs_dict["joint_pos"]["knee_l"][0], next_obs_dict["joint_pos"]["knee_r"][0]]
    foot_pos = [next_obs_dict["body_pos"]["toes_l"][0], next_obs_dict["body_pos"]["pros_foot_r"][0]]
    print (knee_pos)
    print (foot_pos)
    print(knee_pos[0] - foot_pos[0], knee_pos[1] - foot_pos[1])
    print()
    #joints = np.degrees(np.array([next_obs_dict['joint_pos']['ground_pelvis'][0], next_obs_dict['joint_pos']['ground_pelvis'][1], next_obs_dict['joint_pos']['ground_pelvis'][2],
                  #next_obs_dict['joint_pos']['hip_r'][0], next_obs_dict['joint_pos']['knee_r'][0], next_obs_dict['joint_pos']['ankle_l'][0]]))
    #joints = [next_obs_dict['body_pos']]

    #print(joints)
    #print(next_obs_dict['body_pos'])
    #print()


    if done:
        all_fit.append(total_rew); all_len.append(step)
        if exit: break
        else:
            observation = env.reset()
            step = 0
            total_rew = 0
            #exit = True


print('FITNESSES', ['%.2f'%f for f in all_fit], 'LENS', all_len)

