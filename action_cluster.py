import os
from core import off_policy_gradient as pg
from core import models
os.environ["CUDA_VISIBLE_DEVICES"]='3'
from core.mod_utils import list_mean, pprint
from core.models import Actor
import core.reward_shaping as rs
import numpy as np, os, time, random, torch, sys
from core import mod_utils as utils
from core.runner import rollout_worker
import core.ounoise as OU_handle
from torch.multiprocessing import Process, Pipe, Manager




DATA_LIMIT = 50000000





class Parameters:
    def __init__(self):

        self.state_dim = 415
        self.action_dim = 19





class Memory():

    def __init__(self, capacity):
        self.capacity = capacity
        self.s = []; self.ns = []; self.a = []; self.r = []; self.done =[]
        self.num_entries = 0
        self.loaded_files = []

    def __len__(self): return self.num_entries

    def sample(self, batch_size):
        ind = random.sample(range(self.__len__()), batch_size)
        return self.s[ind], self.ns[ind], self.a[ind], self.r[ind], self.done[ind]

    def load(self, data_folder):

        ######## READ DATA #########
        list_files = os.listdir(data_folder)
        while len(list_files) < 1: continue #Wait for Data indefinitely
        print (list_files)

        for index, file in enumerate(list_files):
            if file not in self.loaded_files:
                data = np.load(data_folder + file)
                self.loaded_files.append(file)
                s = data['state']; ns = data['next_state']; a = data['action']; r = data['reward']; done_dist = data['done_dist']

                # Reward Shaping for premature falling
                if RS_PROPORTIONAL_SHAPE:
                    rs_flag = np.where(done_dist != -1) #All tuples which lead to premature convergence
                    #r[rs_flag] = r[rs_flag] - ((DONE_GAMMA ** done_dist[rs_flag]) * abs(r[rs_flag]))
                    r[rs_flag] = r[rs_flag] + (DONE_GAMMA ** done_dist[rs_flag]) * RS_DONE_W  # abs(r[rs_flag]))

                else:
                    rs_flag = np.where(done_dist == np.min(done_dist)) #All tuple which was the last experience in premature convergence
                    r[rs_flag] = RS_DONE_W

                ############## BEHAVIORAL REWARD SHAPE #########
                if USE_BEHAVIOR_RS: r = rs.shaped_data(s,r,FOOTZ_W, KNEEFOOT_W, PELV_W, FOOTY_W, HEAD_W)



                done = (done_dist == 1).astype(float)
                if isinstance(self.s, list):
                    self.s = torch.Tensor(s); self.ns = torch.Tensor(ns); self.a = torch.Tensor(a); self.r = torch.Tensor(r); self.done = torch.Tensor(done)

                else:
                    self.s = torch.cat((self.s, torch.Tensor(s)), 0)
                    self.ns = torch.cat((self.ns, torch.Tensor(ns)), 0)
                    self.a = torch.cat((self.a, torch.Tensor(a)), 0)
                    self.r = torch.cat((self.r, torch.Tensor(r)), 0)
                    self.done = torch.cat((self.done, torch.Tensor(done)), 0)

                self.num_entries = len(self.s)
                if self.num_entries > DATA_LIMIT: break


        print('BUFFER LOADED WITH', self.num_entries, 'SAMPLES')

        # self.s = self.s.pin_memory()
        # self.ns = self.ns.pin_memory()
        # self.a = self.a.pin_memory()
        # self.r = self.r.pin_memory()
        # self.done = self.done.pin_memory()




args = Parameters()

pg_model = Actor(args)
pg_model.load_state_dict(torch.load('R_Skeleton/models/champ'))


evo_model = Actor(args)
evo_model.load_state_dict(torch.load('R_Skeleton/rl_models/td3_best0.95_RS_PROP0.9__ADV_-5.0_-7.5_-5.0_0.0'))


k = None