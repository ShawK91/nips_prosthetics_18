import numpy as np
import torch, time
from core.models import Actor
from core import mod_utils as utils
from core.env_wrapper import EnvironmentWrapper

POLICY_FILE = 'R2_Skeleton/models/erl_best'

#        #
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
#model.eval()

env = EnvironmentWrapper(difficulty=DIFFICULTY, frameskip=FRAMESKIP, x_norm=XNORM, visualize=True)
observation = env.reset()

sim_start = time.time()
total_rew = 0.0; step  = 0; exit = False; total_steps = 0; total_score = 0.0; all_fit = []; all_len = []

while True:
    action = take_action(model, observation)
    #action = np.array([0 for _ in range(19)])
    #action[2] = 1.0; action[4] = 1.0; action[10] = 1.0; action[12] = 1.0

    [observation, reward, done, info] = env.step(action)
    total_rew += reward; step+=1; total_steps+=1; total_score+=reward

    print('Steps', step*FRAMESKIP, 'Rew', '%.2f'%reward, 'Score', '%.2f'%total_rew, 'Pel_y',
          'FITNESSES', ['%.2f'%f for f in all_fit], 'LENS', all_len, 'File', POLICY_FILE, 'Frameskip', FRAMESKIP)
    next_obs_dict = env.env.get_state_desc()
    # lfoot = next_obs_dict["body_pos"]["toes_l"][0]; rfoot = next_obs_dict["body_pos"]["pros_foot_r"][0]
    # lknee = next_obs_dict["body_pos"]["tibia_l"][0]; rknee = next_obs_dict["body_pos"]["pros_tibia_r"][0]
    # #print('Foot L/R', lfoot ,rfoot )
    # #print('Tibia L/R', lknee, rknee)
    # print ('LEFT', 'KNEE_PELVIS', lknee>0, 'KNEE_FOOT', lknee>lfoot, 'FOOT_PELVIS', lfoot>0)
    # print('RIGHT', 'KNEE_PELVIS', rknee > 0, 'KNEE_FOOT', rknee > rfoot, 'FOOT_PELVIS', rfoot > 0)
    # print('PELVIS_X', '%.2f'%next_obs_dict['body_pos']['pelvis'][0], 'KNEE_RL', rknee>lknee, 'FOOT_RL', rfoot>lfoot)
    print('Target:', next_obs_dict["target_vel"])
    print('Current:', next_obs_dict["body_vel"]['pelvis'])





    print()



    # print (np.degrees(np.array([next_obs_dict['body_pos_rot']['tibia_l'][2], next_obs_dict['body_pos_rot']['femur_l'][2]])))
    #
    # print (np.degrees(np.array([next_obs_dict['body_pos_rot']['pros_tibia_r'][2], next_obs_dict['body_pos_rot']['femur_r'][2]])))




    #input('ENTER')


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
