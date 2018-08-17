import opensim as osim
from osim.http.client import Client
import torch
import numpy as np
from core.models import Actor
from core import mod_utils as utils

class Parameters:
    def __init__(self):
        self.state_dim = 159; self.action_dim = 19

def process_dict(state_desc):

    # Augmented environment from the L2R challenge
    res = []
    pelvis = None

    for body_part in ["pelvis", "head", "torso", "toes_l", "toes_r", "talus_l", "talus_r"]:
        if body_part in ["toes_r", "talus_r"]:
            res += [0] * 9
            continue
        cur = []
        cur += state_desc["body_pos"][body_part][0:2]
        cur += state_desc["body_vel"][body_part][0:2]
        cur += state_desc["body_acc"][body_part][0:2]
        cur += state_desc["body_pos_rot"][body_part][2:]
        cur += state_desc["body_vel_rot"][body_part][2:]
        cur += state_desc["body_acc_rot"][body_part][2:]
        if body_part == "pelvis":
            pelvis = cur
            res += cur[1:]
        else:
            cur_upd = cur
            cur_upd[:2] = [cur[i] - pelvis[i] for i in range(2)]
            cur_upd[6:7] = [cur[i] - pelvis[i] for i in range(6, 7)]
            res += cur

    for joint in ["ankle_l", "ankle_r", "back", "hip_l", "hip_r", "knee_l", "knee_r"]:
        res += state_desc["joint_pos"][joint]
        res += state_desc["joint_vel"][joint]
        res += state_desc["joint_acc"][joint]

    for muscle in sorted(state_desc["muscles"].keys()):
        res += [state_desc["muscles"][muscle]["activation"]]
        res += [state_desc["muscles"][muscle]["fiber_length"]]
        res += [state_desc["muscles"][muscle]["fiber_velocity"]]

    cm_pos = [state_desc["misc"]["mass_center_pos"][i] - pelvis[i] for i in range(2)]
    res = res + cm_pos + state_desc["misc"]["mass_center_vel"] + state_desc["misc"]["mass_center_acc"]

    return res

def take_action(model, state, step):
    state = process_dict(state)
    state.append(step)
    state = utils.to_tensor(np.array(state)).unsqueeze(0)
    action = model.forward(state)
    action = utils.to_numpy(action.cpu())
    return action

args = Parameters()
my_controller = Actor(args)
my_controller.load_state_dict(torch.load('R_Skeleton/models/champ.pt'))


intl = 'f0b289455f165a84ff0765b389e346e2'
shawk = "fe00987f44fd6ede87854050e9922f14"
# Settings
remote_base = "http://grader.crowdai.org:1729"
crowdai_token = shawk

client = Client(remote_base)
#client = ClientToEnv(remote_base, crowdai_token)

# Create environment
observation = client.env_create(crowdai_token, env_id="ProstheticsEnv")


# IMPLEMENTATION OF YOUR CONTROLLER
# my_controller = ... (for example the one trained in keras_rl)
total_rew = 0.0; steps = 0
while True:
    action = take_action(my_controller, observation, steps).flatten().tolist()
    [observation, reward, done, info] = client.env_step(action, True)

    total_rew += reward; steps += 1
    print('Steps', steps, 'Rew', reward, 'Total_Reward', total_rew)
    if done:
        steps = 0
        observation = client.env_reset()
        if not observation:
            break

client.submit()