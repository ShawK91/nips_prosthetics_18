import numpy as np
import torch, time
from core.models import Actor
from core import mod_utils as utils
from osim.env import ProstheticsEnv

policy_file = 'Models_repo/policy_best'
integrator_accuracy = 1e-3 /20

class Parameters:
    def __init__(self):
        self.state_dim = 159; self.action_dim = 19



def take_action(model, states, step, all_states, all_actions):
    for state in states:
        if isinstance(state, dict): state = utils.process_dict(state)
        state.append(step)

        state = utils.to_tensor(np.array(state)).unsqueeze(0)
        action = model.forward(state)
        all_states.append(state)
        all_actions.append(action)
        action = utils.to_numpy(action.cpu())
    return all_actions

args = Parameters()
my_controller = Actor(args)
my_controller.load_state_dict(torch.load(policy_file))
my_controller.eval()

envs = [ProstheticsEnv(visualize=False, integrator_accuracy=integrator_accuracy) for _ in range (10)]

all_states = [[] for i in range(5)]
all_actions = [[] for i in range(5)]
all_fitness = []; all_len = []
total_steps = 0
sim_start = time.time()
for i in range(5):
    start = time.time()
    observation = []
    for env in envs: observation.append(env.reset(project=False))
    total_rew = 0.0
    for step in range(1, 1000000):
        actions = take_action(my_controller, observation, step, all_states[i], all_actions[i])

        new_states = []
        for i, action in enumerate(all_actions[0]):
            [observation, reward, done, info] = envs[i].step(utils.to_numpy(action).flatten(), False)
            new_states.append(utils.process_dict(observation))


        #all_actions.append(action)

    #for
       # print (9.0 - (observation["body_vel"]["pelvis"][0] - 3.0)**2, reward,  reward - (9.0 - (observation["body_vel"]["pelvis"][0] - 3.0)**2))

        total_rew += reward; total_steps+=1
        print('Steps', step, 'Rew', '%.2f'%reward, 'Total_Reward', '%.2f'%total_rew, 'Pelvis_pos', '%.2f'%observation['body_pos']['pelvis'][1], 'Pelvis_pos',
              'FITNESSES', ['%.2f'%f for f in all_fitness], 'LENS', all_len, 'Int_Acc', integrator_accuracy, 'Time_per_frame', '%.2f'%((time.time()-sim_start)/float(total_steps)))

        if done: break
    all_fitness.append(total_rew); all_len.append(step)
    print('Trial:', i + 1, total_rew, step, 'Time:', time.time()-start)

print (all_fitness, all_len, policy_file)
k = 0

