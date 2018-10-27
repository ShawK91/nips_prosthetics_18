import numpy as np



################# ROUND 2 BRS #######################


def pelvis_swing(pel_v, use_synthetic_targets, phase_len):


    if use_synthetic_targets:
        mask = np.ones(phase_len*4+1)
        mask[0:30] = 0
        mask[phase_len-5:phase_len+30] = 0
        mask[2*phase_len-5:2*phase_len+30] = 0
        mask[3*phase_len-5:3*phase_len+30] = 0


    else:
        mask = np.ones(1001)
        mask[0:40] = 0
        mask[295:340] = 0
        mask[595:640] = 0
        mask[895:940] = 0

    """Penalizes pelvis swing (a dynamic shaping function)

        Parameters:
            pel_v (ndarray): pelvis velocity trajectory


        Returns:
            r (float): continous reward based on degree that the constraint was satisfied
    """

    pelvis_swing = np.square(np.ediff1d(pel_v[:,0]))
    mask = mask[0:len(pelvis_swing)]
    r = -np.sum(np.multiply(mask, pelvis_swing)) * 50


    return r









####################### ROUND 1 BRS #################



def foot_z_rs(lfoot, rfoot):
    """Foot do not criss-cross over each other in z-axis

        Parameters:
            lfoot (ndarray): left foot positions in z
            rfoot (ndarray): right foot positions in z

        Returns:
            r (float): reward within the range (0.0,1.0) based on percent of timeseteps that the constraint was satisfied
    """
    r = np.mean(lfoot[:,2] < rfoot[:,2])
    return r

def pelvis_height_rs(pelvis_y):
    """pelvis remains below 0.8m (crouched position)

        Parameters:
            pelvis_y (ndarray): pelvis positions in y

        Returns:
            r (float): reward within the range (0.0,1.0) based on percent of timeseteps that the constraint was satisfied
    """

    r = np.mean(pelvis_y < 0.8)
    return r

def knee_bend(ltibia_angle, lfemur_angle, rtibia_angle, rfemur_angle):
    """knee remains bend

        Parameters:
            ltibia_angle (ndarray): angle for left tibia in degrees
            lfemur_angle (ndarray): angle for left femur in degrees
            rtibia_angle (ndarray): angle for right tibia in degrees
            rfemur_angle (ndarray): angle for right femur in degrees

        Returns:
            r (float): reward within the range (0.0,1.0) based on percent of timeseteps that the constraint was satisfied
    """

    tibia_bent_back = np.bitwise_and((ltibia_angle<0), (rtibia_angle<0))
    knee_bend = np.bitwise_and((ltibia_angle < lfemur_angle), (rtibia_angle < rfemur_angle))
    r = np.bitwise_and(tibia_bent_back, knee_bend)
    r = np.mean(r)
    return r


def knee_bend_regression(ltibia_angle, lfemur_angle, rtibia_angle, rfemur_angle):
    """knee remains bend (soft-constraint)

        Parameters:
            ltibia_angle (ndarray): angle for left tibia in degrees
            lfemur_angle (ndarray): angle for left femur in degrees
            rtibia_angle (ndarray): angle for right tibia in degrees
            rfemur_angle (ndarray): angle for right femur in degrees

        Returns:
            r (float): continous reward based on the degree that the constraint was satisfied
    """

    tibia_bent_back = -np.sum(ltibia_angle) - np.sum(rtibia_angle)
    knee_bend = np.sum(lfemur_angle-ltibia_angle) + np.sum(rfemur_angle-rtibia_angle)
    r  = tibia_bent_back + knee_bend
    return r


def thighs_swing(lfemur_angle, rfemur_angle):
    """rewards thighs swing (a dynamic shaping function)

        Parameters:
            lfemur_angle (ndarray): angle for left femur in degrees
            rfemur_angle (ndarray): angle for right femur in degrees

        Returns:
            r (float): continous reward based on degree that the constraint was satisfied
    """

    lswing = np.sum(np.abs(np.ediff1d(lfemur_angle)))
    rswing = np.sum(np.abs(np.ediff1d(rfemur_angle)))

    r = lswing + rswing
    return r

def head_behind_pelvis(head_x):
    """head remains behind pelvis

        Parameters:
            head_x (ndarray): head position in x relative to pelvis x

        Returns:
            r (float): reward within the range (0.0,1.0) based on percent of timeseteps that the constraint was satisfied
    """

    r = np.mean(head_x < 0)
    return r




def pelvis_slack(pelvis_y):
    """slack continous measurement for pelvis remains below 0.8m (crouched position)

        Parameters:
            pelvis_y (ndarray): pelvis positions in y

        Returns:
            r (float): continous reward based on degree that the constraint was satisfied
    """
    r = len(pelvis_y) - np.sum(np.abs(pelvis_y-0.75))
    return r

def foot_y(lfoot, rfoot):
    """foot is not raised too high

        Parameters:
            lfoot (ndarray): left foot positions in y
            rfoot (ndarray): right foot positions in y

        Returns:
            r (float): continous reward based on degree that the constraint was satisfied
    """

    r = lfoot[:,1] + rfoot[:,1]
    r = len(r) - 2.0 *np.sum(r)
    return r



def final_footx(pelv_x, lfoot, rfoot):
    """slack continous measurement for final foot position without raising it

        Parameters:
            pelv_x (ndarray): pelvis positions in x
            lfoot (ndarray): left foot positions in y
            rfoot (ndarray): right foot positions in y

        Returns:
            r (float): continous reward based on degree that the constraint was satisfied
    """

    best_lfoot = (pelv_x + lfoot[:,0]) * (lfoot[:,1] < 0.1)
    best_rfoot = (pelv_x + rfoot[:,0]) * (rfoot[:,1] < 0.1)

    r = max(np.max(best_lfoot), np.max(best_rfoot))
    return r


def shaped_data(s, r, footz_w, kneefoot_w, pelv_w, footy_w, head_w):
    """method to shape a reward based on the above constraints (behavioral reward shaping unlike the temporal one)

        Parameters:
                s (ndarray): Current State
                r (ndarray): Reward
                footz_w (float): weight for computing shaped reward
                kneefoot_w (float): weight for computing shaped reward
                pelv_w (float): weight for computing shaped reward
                footy_w (float): weight for computing shaped reward
                head_w (float): weight for computing shaped reward


        Returns:
            r (ndarray): shaped reward with behavioral shaping
    """
    ####### FOOT Z AXIS ######
    footz_flag = np.where(s[:,95] > s[:,83]) #Left foot z greater than right foot z

    ####### KNEE BEFORE FOOT #######
    kneebend_flag = np.where(np.bitwise_or((s[:,125] > s[:,104]), (s[:,119]>s[:,107])))
    tibia_bent_back_flag = np.where(np.bitwise_or( (s[:,125]>0),  (s[:,119]>0)))

    ######## PELVIS BELOW 0.8 #######
    pelv_flag = np.where(s[:,79] > 0.8)

    ######### FOOT HEIGHT ######
    footy_flag = np.where(np.bitwise_or(s[:,94]>0.15, s[:,82]>0.15))

    ######## HEAD BEHIND PELVIS #######
    head_flag = np.where((s[:,75]>0))

    ##### INCUR PENALTIES #####
    r[footz_flag] = r[footz_flag] + footz_w

    r[kneebend_flag] = r[kneebend_flag] + kneefoot_w
    r[tibia_bent_back_flag] = r[tibia_bent_back_flag] + kneefoot_w

    r[pelv_flag] = r[pelv_flag] + pelv_w
    r[footy_flag] = r[footy_flag] + footy_w
    r[head_flag] = r[head_flag] + head_w

    return r









################### INDICES ############

#["body_pos"]["tibia_l"][0] = 90
#["body_pos"]["pros_tibia_r"][0] = 84


#["body_pos"]["toes_l"][0] = 93
#["body_pos"]["pros_foot_r"][0] = 81

#["body_pos"]["toes_l"][1] = 94
#["body_pos"]["toes_l"][2] = 95

#["body_pos"]["pros_tibia_r"][1] = 85
#["body_pos"]["pros_tibia_r"][2] = 86

#obs_dict["body_pos"]["pelvis"][0] = 78
#obs_dict["body_pos"]["pelvis"][1] = 79


#['body_pos_rot']['tibia_l'][2] = 125
#['body_pos_rot']['pros_tibia_r'][2] = 119
#['body_pos_rot']['femur_l'][2] = 104
#['body_pos_rot']['femur_r'][2] = 107

#['body_pos']['head'][0] - 75

#['body_vel']['pelvis'][0] = 144
#['body_vel']['pelvis'][2] = 146