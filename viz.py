import numpy as np
import torch, time
from core.models import Actor
from core import mod_utils as utils
from core.env_wrapper import EnvironmentWrapper
import opensim
import argparse
from core.mod_utils import str2bool

parser = argparse.ArgumentParser()
parser.add_argument('-policy', help='Where to find the test policy', required=True)
parser.add_argument('-render', type=str2bool, help='Render', default=False)
parser.add_argument('-jgs', type=str2bool, help='JGS?', default=False)

POLICY_FILE = vars(parser.parse_args())['policy']
RENDER = vars(parser.parse_args())['render']
JGS = vars(parser.parse_args())['jgs']


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

env = EnvironmentWrapper(difficulty=DIFFICULTY, frameskip=FRAMESKIP, x_norm=XNORM, visualize=RENDER)
observation = env.reset()

sim_start = time.time()
total_rew = 0.0; step  = 0; exit = False; total_steps = 0; total_score = 0.0; all_fit = []; all_len = []


#SETUP_VIZ
if RENDER:
    vis = env.env.osim_model.model.updVisualizer().updSimbodyVisualizer()
    #vis.setBackgroundType(vis.GroundAndSky)
    vis.setShowFrameNumber(True)
    vis.zoomCameraToShowAllGeometry()
    vis.setCameraFieldOfView(1)

while True:
    if JGS:
        observation[-3] = 1.25; observation[-1] = 0.0
    action = take_action(model, observation)


    [observation, reward, done, info] = env.step(action)

    # if RENDER:
    #     vis.pointCameraAt(opensim.Vec3(env.env.get_state_desc()["body_pos"]["pelvis"][0], 0, 0), opensim.Vec3(0, 1, 0))

    total_rew += reward; step+=1; total_steps+=1; total_score+=reward

    print('Steps', step*FRAMESKIP, 'Rew', '%.2f'%reward, 'Score', '%.2f'%total_rew, 'Pel_y',
          'FITNESSES', ['%.2f'%f for f in all_fit], 'LENS', all_len, 'File', POLICY_FILE, 'Frameskip', FRAMESKIP)
    next_obs_dict = env.env.get_state_desc()
    print('Target:', next_obs_dict["target_vel"])
    print('Current:', next_obs_dict["body_vel"]['pelvis'])





    print()




    if done:
        all_fit.append(total_rew); all_len.append(step)
        if exit: break
        else:
            observation = env.reset()
            step = 0
            total_rew = 0
            #exit = True


print('FITNESSES', ['%.2f'%f for f in all_fit], 'LENS', all_len)
