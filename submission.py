import numpy as np
import torch, time
from core.models import Actor
from core import mod_utils as utils
from osim.http.client import Client


#POLICY_FILE = 'R_Skeleton/rl_models/td3_best0.95_RS_PROP_ADV_DMASK' #
POLICY_FILE = 'R2_Skeleton/models/erl_best'
#POLICY_FILE = 'models_repo/shaped_erl_best'
DIFFICULTY = 1
FRAMESKIP = 5
USER = 'shawk'

def flatten(d):
    res = []  # Result list
    if isinstance(d, dict):
        for key, val in sorted(d.items()):
            res.extend(flatten(val))
    elif isinstance(d, list):
        res = d
    else:
        res = [d]
    return res

def normalize_xpos(d):
    pelvis_x = d["body_pos"]["pelvis"][0]

    d["body_pos"]["femur_r"][0] -= pelvis_x
    d["body_pos"]["pros_tibia_r"][0] -= pelvis_x
    d["body_pos"]["pros_foot_r"][0] -= pelvis_x
    d["body_pos"]["femur_l"][0] -= pelvis_x
    d["body_pos"]["tibia_l"][0] -= pelvis_x
    d["body_pos"]["talus_l"][0] -= pelvis_x
    d["body_pos"]["calcn_l"][0] -= pelvis_x
    d["body_pos"]["toes_l"][0] -= pelvis_x
    d["body_pos"]["torso"][0] -= pelvis_x
    d["body_pos"]["head"][0] -= pelvis_x


    # d["body_pos"]["pelvis"][0] = 0
    #
    # pelvis_z = d["body_pos"]["pelvis"][2]
    #
    # d["body_pos"]["femur_r"][2] -= pelvis_z
    # d["body_pos"]["pros_tibia_r"][2] -= pelvis_z
    # d["body_pos"]["pros_foot_r"][2] -= pelvis_z
    # d["body_pos"]["femur_l"][2] -= pelvis_z
    # d["body_pos"]["tibia_l"][2] -= pelvis_z
    # d["body_pos"]["talus_l"][2] -= pelvis_z
    # d["body_pos"]["calcn_l"][2] -= pelvis_z
    # d["body_pos"]["toes_l"][2] -= pelvis_z
    # d["body_pos"]["torso"][2] -= pelvis_z
    # d["body_pos"]["head"][2] -= pelvis_z
    # d["body_pos"]["pelvis"][2] = 0

    return d

class ClientWrapper:
    def __init__(self, difficulty, args):
        """
        A base template for all environment wrappers.
        """
        # Create environment
        self.client = Client(args.remote_base)
        observation = self.client.env_create(args.crowdai_token, env_id="ProstheticsEnv")

        self.difficulty = difficulty
        self.pelvis_pos = None; self.pelvis_vel = None; self.target_vel = []
        if difficulty == 0: self.target_vel = [3.0, 0.0, 0.0]
        self.update_vars(observation)
        obs = flatten(observation)
        if self.difficulty == 0:  obs = obs + self.target_vel
        self.start_obs = obs



    def reset(self):
        obs_dict = self.client.env_reset()
        if obs_dict == None: return obs_dict
        normalize_xpos(obs_dict)
        self.update_vars(obs_dict)
        obs = flatten(obs_dict)

        if self.difficulty == 0:  obs = obs + self.target_vel
        return obs


    def step(self, action): #Expects a numpy action
        reward = 0.0
        for _ in range(FRAMESKIP):
            next_obs_dict, rew, done, info = self.client.env_step(action.tolist())
            reward += rew
            if done: break

        if not next_obs_dict: return next_obs_dict, reward, done, info
        normalize_xpos(next_obs_dict)
        self.update_vars(next_obs_dict)
        next_obs = flatten(next_obs_dict)
        if self.difficulty == 0:  next_obs = next_obs + self.target_vel
        return next_obs, reward, done, info



    def update_vars(self, obs_dict):
        self.pelvis_vel = obs_dict["body_vel"]["pelvis"][0]
        self.pelvis_pos = obs_dict["body_pos"]["pelvis"][1]
        if self.difficulty != 0: self.target_vel = [obs_dict["target_vel"][0], obs_dict["target_vel"][1], obs_dict["target_vel"][2]]

class Parameters:
    def __init__(self):
        self.state_dim = 415; self.action_dim = 19
        self.remote_base = "http://grader.crowdai.org:1730"
        self.crowdai_token = 'f0b289455f165a84ff0765b389e346e2' if USER == 'intl' else "fe00987f44fd6ede87854050e9922f14"

def take_action(model, state):
    state = utils.to_tensor(np.array(state)).unsqueeze(0)
    action = model.forward(state)
    action = utils.to_numpy(action)
    return action.flatten()

args = Parameters()
model = Actor(args)
model.load_state_dict(torch.load(POLICY_FILE))
client = ClientWrapper(DIFFICULTY, args)
observation = client.start_obs

total_rew = 0.0; step = 0;
while True:

    action = take_action(model, observation)
    #if step < 10: action[(action == 1.0)] = 1.2


    [observation, reward, done, info] = client.step(action)

    total_rew += reward; step += 1
    print('Steps', step*FRAMESKIP, 'Rew', '%.2f'%reward, 'Total_Reward', '%.2f'%total_rew)
    if done:
        observation = client.reset()
        if not observation: break
        step = 0


client.client.submit()
