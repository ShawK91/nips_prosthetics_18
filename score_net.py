import numpy as np
import torch, time
from core.models import Actor
from core import mod_utils as utils
from core.env_wrapper import EnvironmentWrapper
import core.reward_shaping as rs
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('-policy', help='Where to find the test policy', required=True)

POLICY_FILE = vars(parser.parse_args())['policy']



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

env = EnvironmentWrapper(difficulty=DIFFICULTY, frameskip=FRAMESKIP, x_norm=XNORM)
observation = env.reset()

sim_start = time.time()
total_rew = 0.0; step  = 0; exit = False; total_steps = 0; total_score = 0.0; all_fit = []; all_len = []


while True:
    action = take_action(model, observation)

    [observation, reward, done, info] = env.step(action)
    total_rew += reward; step+=1; total_steps+=1; total_score+=reward




    print('Steps', env.istep, 'Rew', '%.2f'%reward, 'Total_Reward', '%.2f'%total_rew,'Pelvis_pos', '%.2f'%env.pelvis_y,
          'FITNESSES', ['%.2f'%f for f in all_fit], 'LENS', all_len, 'File', POLICY_FILE, 'Frameskip', FRAMESKIP)
    next_obs_dict = env.env.get_state_desc()


    print('Action_pen', '%.4f'%env.action_pen, 'X_pen', '%.4f'%env.x_pen, 'Z_pen', '%.4f'%env.z_pen, 'Z_pen Minus/Plus', '%.4f'%env.zminus_pen, '%.4f'%env.zplus_pen)
    print ('Target', ['%.2f'%v for v in env.last_real_target])
    print('Vel', ['%.2f'%v for v in env.vel_traj[-1]])
    print()


    if done:
        all_fit.append(total_rew); all_len.append(env.istep)
        if exit: break
        else:
            observation = env.reset()
            step = 0
            total_rew = 0
            #exit = True


print('FITNESSES', ['%.2f'%f for f in all_fit], 'LENS', all_len)
