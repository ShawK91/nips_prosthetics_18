import opensim as osim
from osim.env import ProstheticsEnv

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

class EnvironmentWrapper:
    def __init__(self, difficulty):
        """
        A base template for all environment wrappers.
        """
        self.env = ProstheticsEnv(visualize=False, difficulty=difficulty)
        self.difficulty = difficulty
        self.pelvis_pos = None; self.pelvis_vel = None; self.target_vel = []
        if difficulty == 0: self.target_vel = [3.0, 0.0, 0.0]


        # Attributes
        self.observation_space = self.env.observation_space if hasattr(self.env, 'observation_space') else None
        self.action_space = self.env.action_space if hasattr(self.env, 'action_space') else None
        self.time_limit = self.env.time_limit if hasattr(self.env, 'time_limit') else None
        self.submit = self.env.submit if hasattr(self.env, 'submit') else None
        self.difficulty = self.env.difficulty if hasattr(self.env, 'difficulty') else None

    def reset(self):
        obs_dict = self.env.reset(project=False)
        #obs_dict = self.env.get_state_desc()
        self.update_vars(obs_dict)
        obs = flatten(obs_dict)

        if self.difficulty == 0:  obs = obs + self.target_vel
        return obs


    def step(self, action): #Expects a numpy action
        next_obs_dict, reward, done, info = self.env.step(action.tolist(), project=False)
        #next_obs_dict = self.env.get_state_desc()
        self.update_vars(next_obs_dict)
        next_obs = flatten(next_obs_dict)

        if self.difficulty == 0:  next_obs = next_obs + self.target_vel
        return next_obs, reward, done, info


    def respawn(self):
        #self.env.reset()
        self.env = ProstheticsEnv(visualize=False, difficulty=self.difficulty)

    def update_vars(self, obs_dict):
        self.pelvis_vel = obs_dict["body_vel"]["pelvis"][0]
        self.pelvis_pos = obs_dict["body_pos"]["pelvis"][1]
        if self.difficulty != 0: self.target_vel = [obs_dict["target_vel"][0], obs_dict["target_vel"][1], obs_dict["target_vel"][2]]




