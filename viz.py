import numpy as np
import torch, time
from core.models import Actor
from core import mod_utils as utils
from core.env_wrapper import EnvironmentWrapper

#POLICY_FILE = 'R_Skeleton/rl_models/td3_best0.95_RS_PROP_ADV_DMASK'
#POLICY_FILE = 'models_repo/actors/erl_best'
POLICY_FILE = '0_EXP/R_Skeleton/models/shaped_erl_best'         #
DIFFICULTY = 0
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
#model.eval()

env = EnvironmentWrapper(difficulty=DIFFICULTY, frameskip=FRAMESKIP, x_norm=XNORM, visualize=True)
observation = env.reset()

sim_start = time.time()
total_rew = 0.0; step  = 0; exit = False; total_steps = 0; total_score = 0.0; all_fit = []; all_len = []

while True:
    action = take_action(model, observation)

    [observation, reward, done, info] = env.step(action)
    total_rew += reward; step+=1; total_steps+=1; total_score+=reward

    print('Steps', step*FRAMESKIP, 'Rew', '%.2f'%reward, 'Score', '%.2f'%total_rew, 'Pel_x', '%.2f'%env.pelvis_pos, 'Pel_vel', '%.2f'%env.pelvis_vel,
          'FITNESSES', ['%.2f'%f for f in all_fit], 'LENS', all_len, 'File', POLICY_FILE, 'Frameskip', FRAMESKIP)
    next_obs_dict = env.env.get_state_desc()
    print('Pelvis', next_obs_dict['body_pos']['pelvis'][0])
    print('Foot L/R', next_obs_dict["body_pos"]["toes_l"][0], next_obs_dict["body_pos"]["pros_foot_r"][0])
    print('Tibia L/R', next_obs_dict["body_pos"]["tibia_l"][0], next_obs_dict["body_pos"]["pros_tibia_r"][0])
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
