import numpy as np
import torch, time
from core.models import Actor
from core import mod_utils as utils
from core.env_wrapper import EnvironmentWrapper
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-policy', help='Where to find the test policy', default='docker_sub/nips2018-ai-for-prosthetics-round2-starter-kit/models/m4')
parser.add_argument('-seed', type=int, help='seed', default=1001183)
parser.add_argument('-scheme', type=int, help='scheme', default=0)


POLICY_FILE = vars(parser.parse_args())['policy']
SEED = vars(parser.parse_args())['seed']
SCHEME = vars(parser.parse_args())['scheme']



DIFFICULTY = 1
FRAMESKIP = 5
XNORM = True

class Parameters:
    def __init__(self):
        self.state_dim = 415; self.action_dim = 19

def take_action(model, state):
    state = utils.to_tensor(np.array(state)).unsqueeze(0)
    action = model.forward(state)
    action = utils.to_numpy(action)
    return action.flatten()

args = Parameters()
model = Actor(args)
model.load_state_dict(torch.load(POLICY_FILE))

env = EnvironmentWrapper(difficulty=DIFFICULTY, frameskip=FRAMESKIP)
observation = env.reset()

sim_start = time.time()
total_rew = 0.0; step  = 0; total_steps = 0; total_score = 0.0; all_fit = []; all_len = []


last_z = 0;
while True:
    if SCHEME == 1:
        if observation[-1] < last_z: observation[-1] = last_z
    elif SCHEME == 2:
        observation[-1] = 0.0

    action = take_action(model, observation)

    [observation, reward, done, info] = env.step(action)
    total_rew += reward; step+=1; total_steps+=1; total_score+=reward




    print('Steps', env.istep, 'Rew', '%.2f'%reward, 'Total_Reward', '%.2f'%total_rew,
          'FITNESSES', ['%.2f'%f for f in all_fit], 'LENS', all_len, 'File', POLICY_FILE, 'Frameskip', FRAMESKIP)
    next_obs_dict = env.env.get_state_desc()




    if done:
        all_fit.append(total_rew); all_len.append(env.istep)
        observation = env.reset()
        step = 0
        total_rew = 0
        last_z = 0.0



print('FITNESSES', ['%.2f'%f for f in all_fit], 'LENS', all_len)
