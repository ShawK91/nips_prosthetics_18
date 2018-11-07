import opensim as osim
from osim.env import ProstheticsEnv


class EnvironmentWrapper:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            difficulty (int): Env difficulty: 0 --> Round 1; 1 --> Round 2
            frameskip (int): Number of frames to skip (controller frequency)
            x_norm (bool): Use x normalization? Absolute to pelvis centered frame?
            rs (bool): Use reward shaping?
            visualize (bool): Render the env?

    """
    def __init__(self, difficulty, frameskip=5, visualize=False):
        """
        A base template for all environment wrappers.
        rs --> Reward shaping
        """
        self.env = ProstheticsEnv(visualize=visualize, difficulty=difficulty)
        self.istep = 0
        self.difficulty = difficulty; self.frameskip = frameskip;

        self.real_z = None


    def reset(self):
        """Method to reset state variables for a rollout
            Parameters:
                None

            Returns:
                None
        """

        self.istep = 0
        obs_dict = self.env.reset(project=False)
        self.real_z = obs_dict["target_vel"][2]
        obs_dict["target_vel"][2] = 0.0
        obs_dict = normalize_pos(obs_dict)
        obs = flatten(obs_dict)

        return obs


    def step(self, action): #Expects a numpy action
        """Take an action to forward the simulation

            Parameters:
                action (ndarray): action to take in the env

            Returns:
                next_obs (list): Next state
                reward (float): Reward for this step
                done (bool): Simulation done?
                info (None): Template from OpenAi gym (doesnt have anything)
        """

        reward = 0

        for _ in range(self.frameskip):
            self.istep += 1
            next_obs_dict, rew, done, info = self.env.step(action.tolist(), project=False)

            #Compensate for z
            rew += (next_obs_dict["body_vel"]["pelvis"][2] - self.real_z) ** 2
            rew -= (next_obs_dict["body_vel"]["pelvis"][2]) ** 2
            self.real_z = next_obs_dict["target_vel"][2]

            reward += rew
            if done: break


        next_obs_dict = normalize_pos(next_obs_dict)
        next_obs_dict["target_vel"][2] = 0.0
        next_obs = flatten(next_obs_dict)

        return next_obs, reward, done, info


    def respawn(self):
        """Method to respawn the env (hard reset)

            Parameters:
                None

            Returns:
                None
        """

        self.env = ProstheticsEnv(visualize=False, difficulty=self.difficulty)





def flatten(d):
    """Recursive method to flatten a dict -->list
        Parameters:
            d (dict): dict
        Returns:
            l (list)
    """

    res = []  # Result list
    if isinstance(d, dict):
        for key, val in sorted(d.items()):
            res.extend(flatten(val))
    elif isinstance(d, list):
        res = d
    else:
        res = [d]
    return res



def normalize_pos(d):
    """Put positions from absolute --> relative frame of the pelvis
        Parameters:
            d (dict): dict
        Returns:
            d (dict)
    """

    #X position for the pelvis
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
    d["body_pos"]["pelvis"][0] = 0

    #Z position for the pelvis
    pelvis_z = d["body_pos"]["pelvis"][2]
    d["body_pos"]["femur_r"][2] -= pelvis_z
    d["body_pos"]["pros_tibia_r"][2] -= pelvis_z
    d["body_pos"]["pros_foot_r"][2] -= pelvis_z
    d["body_pos"]["femur_l"][2] -= pelvis_z
    d["body_pos"]["tibia_l"][2] -= pelvis_z
    d["body_pos"]["talus_l"][2] -= pelvis_z
    d["body_pos"]["calcn_l"][2] -= pelvis_z
    d["body_pos"]["toes_l"][2] -= pelvis_z
    d["body_pos"]["torso"][2] -= pelvis_z
    d["body_pos"]["head"][2] -= pelvis_z
    d["body_pos"]["pelvis"][2] = 0

    return d

