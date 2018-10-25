import opensim as osim
from osim.env import ProstheticsEnv
import numpy as np, random, math


class EnvironmentWrapper:
    """Wrapper around the Environment to expose a cleaner interface for RL

        Parameters:
            difficulty (int): Env difficulty: 0 --> Round 1; 1 --> Round 2
            frameskip (int): Number of frames to skip (controller frequency)
            x_norm (bool): Use x normalization? Absolute to pelvis centered frame?
            rs (bool): Use reward shaping?
            visualize (bool): Render the env?

    """
    def __init__(self, difficulty, frameskip=5, x_norm=True, rs=False, visualize=False, use_synthetic_targets=False, xbias=None, zbias=None, phase_len = 100):
        """
        A base template for all environment wrappers.
        rs --> Reward shaping
        """
        self.env = ProstheticsEnv(visualize=visualize, difficulty=difficulty)
        self.difficulty = difficulty; self.frameskip = frameskip; self.x_norm = x_norm; self.rs = rs
        self.use_synth_targets = use_synthetic_targets; self.xbias = xbias; self.zbias=zbias; self.phase_len = phase_len

        #Some imporatant state trackers
        self.pelvis_y = None; self.pelvis_vel = None;  self.ltibia_xyz = []; self.rtibia_xyz = []; self.lfoot_xyz = []; self.rfoot_xyz = []; self.pelvis_x = None
        self.ltibia_angle = None; self.rtibia_angle = None; self.lfemur_angle = None; self.rfemur_angle = None
        self.head_x = None
        self.lfoot_y = None; self.rfoot_y = None

        #Round 2 Attributes
        self.target_vel_traj = []
        self.vel_traj = []
        self.z_pen = 0.0; self.zplus_pen = 0.0; self.zminus_pen = 0.0
        self.x_pen = 0.0
        self.action_pen = 0.0


        self.istep = 0

        # Attributes
        self.observation_space = self.env.observation_space if hasattr(self.env, 'observation_space') else None
        self.action_space = self.env.action_space if hasattr(self.env, 'action_space') else None
        self.time_limit = self.env.time_limit if hasattr(self.env, 'time_limit') else None
        self.submit = self.env.submit if hasattr(self.env, 'submit') else None
        self.difficulty = self.env.difficulty if hasattr(self.env, 'difficulty') else None

        self.last_real_target = None
        #Synthetic Target
        if self.use_synth_targets:
            self.synth_targets = condense_targets(phase_len+1, xbias, zbias)





    def reset(self):
        """Method to reset state variables for a rollout
            Parameters:
                None

            Returns:
                None
        """
        if self.difficulty != 0: #ROUND 2
        #Reset Round 2 Attributes
            self.target_vel_traj = []
            self.vel_traj = []
            self.z_pen = 0.0; self.zplus_pen = 0.0; self.zminus_pen = 0.0
            self.x_pen = 0.0
            self.action_pen = 0.0

        #Synthetic Target
        if self.use_synth_targets:
            self.synth_targets = condense_targets(self.phase_len+1, self.xbias, self.zbias)


        self.istep = 0
        obs_dict = self.env.reset(project=False)

        self.last_real_target = [obs_dict["target_vel"][0], 0, obs_dict["target_vel"][2]]
        if self.use_synth_targets:
            obs_dict["target_vel"][2] = self.synth_targets[self.istep][2]
            obs_dict["target_vel"][0] = self.synth_targets[self.istep][0]

        if self.x_norm:
            if self.difficulty == 0: obs_dict = normalize_xpos(obs_dict)
            else: obs_dict = normalize_pos(obs_dict)

        self.update_vars(obs_dict)
        obs = flatten(obs_dict)

        if self.difficulty == 0:  obs = obs + [3.0, 0, 0]
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

            if self.use_synth_targets:
                #Cancel last movement penalty
                rew += (next_obs_dict["body_vel"]["pelvis"][2] - self.last_real_target[2]) ** 2
                rew += (next_obs_dict["body_vel"]["pelvis"][0] - self.last_real_target[0]) ** 2

                #Update last real target
                self.last_real_target = [next_obs_dict["target_vel"][0], 0, next_obs_dict["target_vel"][2]]

                #Replace targets in dictionary of observation
                next_obs_dict["target_vel"][0] = self.synth_targets[self.istep][0]
                next_obs_dict["target_vel"][2] = self.synth_targets[self.istep][2]

                #Compute new penalties based on synthetic ones given last timestep
                rew -= (next_obs_dict["body_vel"]["pelvis"][2] - self.synth_targets[self.istep-1][2]) ** 2
                rew -= (next_obs_dict["body_vel"]["pelvis"][0] - self.synth_targets[self.istep-1][0]) ** 2


            reward += rew

            # ROUND 2 Attributes
            if self.difficulty != 0:
                self.vel_traj.append(next_obs_dict["body_vel"]["pelvis"])

                if self.use_synth_targets:
                    self.z_pen += (next_obs_dict["body_vel"]["pelvis"][2] - self.synth_targets[self.istep-1][2]) ** 2
                    self.x_pen += (next_obs_dict["body_vel"]["pelvis"][0] - self.synth_targets[self.istep-1][0]) ** 2
                    # self.action_pen += np.sum(np.array(self.env.osim_model.get_activations()) ** 2) * 0.001
                    if next_obs_dict["target_vel"][2] > 0:  # Z matching in the positive axis
                        self.zplus_pen += (next_obs_dict["body_vel"]["pelvis"][2] - self.synth_targets[self.istep-1][2]) ** 2
                    else:  # Z matching in the negative axis
                        self.zminus_pen += (next_obs_dict["body_vel"]["pelvis"][2] - self.synth_targets[self.istep-1][2]) ** 2
                else:
                    self.z_pen += (next_obs_dict["body_vel"]["pelvis"][2] - self.last_real_target[2]) ** 2
                    self.x_pen += (next_obs_dict["body_vel"]["pelvis"][0] - self.last_real_target[0]) ** 2
                    # self.action_pen += np.sum(np.array(self.env.osim_model.get_activations()) ** 2) * 0.001
                    if next_obs_dict["target_vel"][2] > 0:  # Z matching in the positive axis
                        self.zplus_pen += (next_obs_dict["body_vel"]["pelvis"][2] - self.last_real_target[2]) ** 2
                    else:  # Z matching in the negative axis
                        self.zminus_pen += (next_obs_dict["body_vel"]["pelvis"][2] - self.last_real_target[2]) ** 2



            if done: break

        if self.x_norm:
            if self.difficulty == 0: next_obs_dict = normalize_xpos(next_obs_dict)
            else: next_obs_dict = normalize_pos(next_obs_dict)

        self.update_vars(next_obs_dict)
        next_obs = flatten(next_obs_dict)


        if self.difficulty == 0:  next_obs = next_obs + [3.0, 0, 0]
        return next_obs, reward, done, info


    def respawn(self):
        """Method to respawn the env (hard reset)

            Parameters:
                None

            Returns:
                None
        """

        self.env = ProstheticsEnv(visualize=False, difficulty=self.difficulty)

    def update_vars(self, obs_dict):
        """Updates the variables that are being tracked from observations

            Parameters:
                obs_dict (dict): state dict

            Returns:
                None
        """

        #Target Velocity
        self.target_vel_traj.append(obs_dict["target_vel"])
        self.pelvis_vel = obs_dict["body_vel"]["pelvis"][0]
        self.pelvis_y = obs_dict["body_pos"]["pelvis"][1]



        #RS Variables
        if self.rs:

            self.ltibia_xyz = obs_dict["body_pos"]["tibia_l"]; self.rtibia_xyz = obs_dict["body_pos"]["pros_tibia_r"]
            self.lfoot_xyz = obs_dict["body_pos"]["toes_l"]; self.rfoot_xyz = obs_dict["body_pos"]["pros_foot_r"]


            #Angles
            self.ltibia_angle = obs_dict['body_pos_rot']['tibia_l'][2]
            self.rtibia_angle = obs_dict['body_pos_rot']['pros_tibia_r'][2]
            self.lfemur_angle = obs_dict['body_pos_rot']['femur_l'][2]
            self.rfemur_angle = obs_dict['body_pos_rot']['femur_r'][2]

            self.head_x = obs_dict['body_pos']['head'][0]
            self.pelvis_x = obs_dict["body_pos"]["pelvis"][0]





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

def normalize_xpos(d):
    """Put x positions from absolute --> relative frame of the pelvis
        Parameters:
            d (dict): dict
        Returns:
            d (dict)
    """
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

    return d

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

def rect(row):
    r = row[0]
    theta = row[1]
    x = r * math.cos(theta)
    y = 0
    z = r * math.sin(theta)
    return np.array([x,y,z])

def generate_new_targets(poisson_lambda=300, xbias=False, zbias=False):
    nsteps = 1001
    rg = np.array(range(nsteps))
    velocity = np.zeros(nsteps)
    heading = np.zeros(nsteps)

    velocity[0] = 1.25
    heading[0] = 0

    change = np.cumsum(np.random.poisson(poisson_lambda, 10))

    for i in range(1, nsteps):
        velocity[i] = velocity[i - 1]
        heading[i] = heading[i - 1]

        if i in change:
            dx = random.choice([-1, 1]) * random.uniform(-0.5, 0.5)
            dz = random.choice([-1, 1]) * random.uniform(-math.pi / 8, math.pi / 8)

            nextx = velocity[i] + dx
            nextz = heading[i] + dz

            #Synthetic Targets
            if xbias == 'positive' and nextx < 1.25: nextx =  velocity[i] - dx
            if xbias == 'negative' and nextx > 1.25: nextx = velocity[i] - dx
            if zbias == 'positive' and nextz < 0.0: nextz = heading[i] - dz
            if zbias == 'negative' and nextz > 0.0: nextz = heading[i] - dz



            velocity[i] = nextx
            heading[i] = nextz

    trajectory_polar = np.vstack((velocity, heading)).transpose()
    targets = np.apply_along_axis(rect, 1, trajectory_polar)
    return targets



def condense_targets(phase_len, x_bias = None, z_bias = None):
    targets = np.array(generate_new_targets(xbias=x_bias, zbias=z_bias)[1::300])
    targets = np.repeat(targets, phase_len, 0)
    return targets
