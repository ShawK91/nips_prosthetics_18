import numpy as np


def create_her_experiences(s, ns, a, r, done, k=4):
    """Generate synthetic experiences by substituting goals using Hindsight Experience Replay

        Parameters:
              s (ndarray): Current State
              ns (ndarray): Next State
              a (ndarray): Action
              r (ndarray): Reward
              done (ndarray): Done
              k (int): Ratio of hindsight experience per unit real experiences

        Returns:
              her_s (ndarray): Hindsight-augmented states
              her_ns (ndarray): Hindsight-augmented Next State
              her_a (ndarray): Hindsight-augmented Action
              her_r (ndarray): Hindsight-augmented Reward
              her_done (ndarray): Hindsight-augmented Done

    """


    her_s = s.repeat(4, 0); her_ns = ns.repeat(4, 0); her_a = a.repeat(4, 0); her_r = r.repeat(4, 0); her_done = done.repeat(4, 0)

    #Compute noise perturbation to target velocity
    x_noise = np.random.normal(0, 0.3, (len(her_s), ))
    z_noise = np.random.normal(0, 0.3, (len(her_s),))

    #Log original velocity_penalty
    prev_xpenalty = (her_ns[:,-3]-her_ns[:,144])**2
    prev_zpenalty = (her_ns[:,-1]-her_ns[:,146])**2


    #Add noise to state and next state target velocity vectors
    her_s[:,-3] += x_noise; her_ns[:,-3] += x_noise
    her_s[:, -1] += z_noise; her_ns[:, -1] += z_noise

    #Recompute velocity penalties
    x_penalty = (her_ns[:,-3]-her_ns[:,144])**2
    z_penalty = (her_ns[:,-1]-her_ns[:,146])**2

    #Perturb reward to reflect the hindsight experiences and new reward
    correction = np.reshape(x_penalty - prev_xpenalty + z_penalty - prev_zpenalty, (len(z_penalty), 1))
    her_r = her_r - correction

    return her_s, her_ns, her_a, her_r, her_done



#TEST
# s = np.ones((10,415))
# ns = np.ones((10,415))
# a = np.ones((10,19))
# r = np.ones((10,1))
# done = np.ones((10,1))
#
# hs, hns, ha, hr, hdone = create_her_experiences(s,ns,a,r,done)
#
# k = None