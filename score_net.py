import opensim as osim
import numpy as np
import torch, time
from core.models import Actor
from core import mod_utils as utils
from osim.env import ProstheticsEnv

policy_file = 'R_Skeleton/rl_models/td3_best0.997_-10.0'
integrator_accuracy = 5e-5

class Parameters:
    def __init__(self):
        self.state_dim = 158; self.action_dim = 19

def take_action(model, state, step):
    if isinstance(state, dict): state = utils.process_dict(state)
    #state.append(step)
    state = utils.to_tensor(np.array(state)).unsqueeze(0)
    action = model.forward(state)
    action = utils.to_numpy(action.cpu())
    return action

args = Parameters()
my_controller = Actor(args)
my_controller.load_state_dict(torch.load(policy_file))
my_controller.eval()

env = ProstheticsEnv(visualize=False, integrator_accuracy=integrator_accuracy)
observation = env.reset(project=False)
all_actions = []
all_fitness = []; all_len = []
total_steps = 0
sim_start = time.time()
for i in range(1):
    start = time.time()
    total_rew = 0.0; step  = 0.0; exit = False; total_steps = 0.0; total_score = 0.0
    while True:
        action = take_action(my_controller, observation, step).flatten()
        all_actions.append(action)

        [observation, reward, done, info] = env.step(action, False)

        total_rew += reward; step+=1; total_steps+=1; total_score+=reward
        print('Steps', step, 'Rew', '%.2f'%reward, 'Total_Reward', '%.2f'%total_rew, 'Final_Score', '%.2f'%total_score,'Pelvis_pos', '%.2f'%observation['body_pos']['pelvis'][1], 'Pelvis_pos',
              'FITNESSES', ['%.2f'%f for f in all_fitness], 'LENS', all_len, 'Int_Acc', integrator_accuracy, 'Time_per_frame', '%.2f'%((time.time()-sim_start)/float(total_steps)))

        if done:
            if exit: break
            else:
                observation = env.reset()
                step = 0;
                total_rew = 0
                exit = True

#     all_fitness.append(total_rew); all_len.append(step)
#     print('Trial:', i + 1, total_rew, step, 'Time:', time.time()-start)
#
# print (all_fitness, all_len, policy_file)
# k = 0

