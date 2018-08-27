import numpy as np, os, random
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import torch
from core import vae_fm
from core import mod_utils as utils

model_fname = 'vae_model'
best_valid = 'best_valid_fm'

# Hyperparameters for ConvVAE
z_dim = 256
USE_VAE = False
BATCH_SIZE = 1024
learning_rate = 1e-3
SEED = True
ITER_PER_EPOCH = 1000

#LOSS WEIGHTS
STATE_W = 1.0
POS_W = 100.0
VEL_W = 100.0
ACTION_W = 1.0


#FAIRLY STATIC
QUICK_TEST = False
if QUICK_TEST:
    ITER_PER_EPOCH = 50
    BATCH_SIZE = 32


class Buffer():

    def __init__(self, capacity):
        self.capacity = capacity
        self.s = []; self.ns = []; self.a = []; self.r = []; self.fit =[]; self.done = []
        self.position = 0
        #Validation set
        self.valid_inds = []; self.train_inds = []

    def __len__(self):
        return self.position

    def sample(self, batch_size, is_train):
        if is_train: ind = random.sample(self.train_inds, batch_size)
        else: ind = random.sample(self.valid_inds, batch_size)
        return self.s[ind], self.ns[ind], self.a[ind], self.r[ind], self.done[ind]

    def load(self, data_folder):

        ######## READ DATA #########
        list_files = os.listdir(data_folder)
        print (list_files)
        if len(list_files) != 0: #Reset everything
            self.s = []; self.ns = []; self.a = []; self.r = []; self.fit =[]; self.done = []; self.position = 0
            self.fit_pr = []

        for file_id, file in enumerate(list_files):
            data = np.load(data_folder + file)

            for s, ns, a, r, fit, done in zip(data['state'], data['next_state'], data['action'], data['reward'], data['fitness'], data['done']):

                self.s.append(torch.Tensor(s).unsqueeze(0))
                self.ns.append(torch.Tensor(ns).unsqueeze(0))
                self.a.append(torch.Tensor(a).unsqueeze(0))
                self.r.append(torch.Tensor(r).unsqueeze(0))
                self.fit.append(torch.Tensor(fit).unsqueeze(0))
                self.done.append(torch.Tensor(done).unsqueeze(0))

                self.position+=1
                #if QUICK_TEST and self.position >= 10000: break
                if QUICK_TEST and self.position >= 10000*(file_id+1): break #QUICK TEST

            print('BUFFER LOADED WITH', self.position, 'SAMPLES FROM WITH TAG', file)
            del data
            if QUICK_TEST and self.position > 100000: break  # QUICK TEST

        self.s = torch.cat(self.s)[:,:-1].pin_memory()
        self.ns = torch.cat(self.ns)[:,:-1].pin_memory()
        self.a = torch.cat(self.a).pin_memory()
        self.r = torch.cat(self.r).pin_memory()
        self.fit = torch.cat(self.fit).pin_memory()
        self.done = torch.cat(self.done).pin_memory()

        #Validation indices
        self.valid_inds = random.sample(range(self.__len__()), int(0.1*self.__len__()))
        self.train_inds = list(set(range(self.__len__())) - set(self.valid_inds))


############ LOAD DATA AND CONSTRUCT DATA BUFFER ########
buffer= Buffer(capacity=10000000)
buffer.load('R_Skeleton/data/')
tracker = utils.Tracker('R_VAE/', ['total', 'fm', 'state', 'pos', 'vel', 'action', 'total_val', 'fm_val', 'state_val', 'pos_val', 'vel_val', 'action_val' ], '_loss.csv')  # Initiate tracker

######################TRAIN VAE########################
vae_model = vae_fm.VAE(158, 19, z_dim).cuda()
if SEED:
    try:
        vae_model.load_state_dict(torch.load('R_VAE/' + model_fname))
        print ('MODEL LOADED FROM R_VAE')
    except: print ('NO SEED - STARTING FROM SCRATCH')

optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate, weight_decay=0.001)
total_params = sum(p.numel() for p in vae_model.parameters())
print ('Running VAE with', total_params, 'params')

################################# TRAIN LOOP ########################
val_scores = []; train_scores =[]
for epoch in range(1, 1000000000):

    #REPLOAD DATA PERIODICALLY
    #if epoch % 1000 == 0: training_generator, test_generator = load_data()

    #SAVE MODEL PERIODICALLY
    if epoch % 50 == 0: torch.save(vae_model.state_dict(), 'R_VAE/'+model_fname)

    tr_total_loss, tr_fm_loss, tr_state_loss, tr_pos_loss, tr_vel_loss, tr_a_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for _ in range(ITER_PER_EPOCH):
        s, ns, a, r, d = buffer.sample(BATCH_SIZE, is_train=True)
        s = s.cuda(); ns = ns.cuda(); a = a.cuda(); r = r.cuda(); d = d.cuda()

        z, next_z, dec_state, dec_next_state, mu, logvar = vae_model.forward(s, a, deterministic=USE_VAE) #FM FORWARD PASS
        r_pred, a_pred = vae_model.action_reward_predictor(z, next_z, dec_next_state) #REWARD_ACTION FORWARD PASS

        #### COMPUTE LOSS #####
        total_loss_T, fm_loss, state_loss, pos_loss, vel_loss = vae_model.fm_recon_loss(s, dec_state, ns, dec_next_state, mu, logvar, state_w=STATE_W, pos_w=POS_W, vel_w=VEL_W) #FM LOSS
        a_loss_T, a_loss = vae_model.ar_pred_loss(a, a_pred, a_w=ACTION_W)

        tr_total_loss += total_loss_T.item()/(ITER_PER_EPOCH); tr_fm_loss+=fm_loss/(ITER_PER_EPOCH);
        tr_state_loss+=state_loss/(ITER_PER_EPOCH); tr_pos_loss+=pos_loss/(ITER_PER_EPOCH)
        tr_vel_loss+= vel_loss / (ITER_PER_EPOCH ); tr_a_loss += a_loss / (ITER_PER_EPOCH)

        optimizer.zero_grad()
        total_loss_T.backward(retain_graph=True)
        a_loss_T.backward()
        #torch.nn.utils.clip_grad_norm_(vae_model.parameters(), 10)
        optimizer.step()

    # #Validation loss from test set
    vae_model.eval()
    val_total_loss, val_fm_loss, val_state_loss, val_pos_loss, val_vel_loss, val_a_loss = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
    for _ in range(int(ITER_PER_EPOCH*0.1)):
        s, ns, a, r, d = buffer.sample(BATCH_SIZE, is_train=False)
        s = s.cuda(); ns = ns.cuda(); a = a.cuda(); r = r.cuda(); d = d.cuda()

        z, next_z, dec_state, dec_next_state, mu, logvar = vae_model.forward(s, a, deterministic=USE_VAE) #FM FORWARD PASS
        r_pred, a_pred = vae_model.action_reward_predictor(z, next_z, dec_next_state) #REWARD_ACTION FORWARD PASS

        #### COMPUTE LOSS #####
        total_loss_T, fm_loss, state_loss, pos_loss, vel_loss = vae_model.fm_recon_loss(s, dec_state, ns, dec_next_state, mu, logvar, state_w=STATE_W, pos_w=POS_W, vel_w=VEL_W) #FM LOSS
        _, a_loss = vae_model.ar_pred_loss(a, a_pred, a_w=ACTION_W)

        val_total_loss += total_loss_T.item()/(ITER_PER_EPOCH*0.1); val_fm_loss+=fm_loss/(ITER_PER_EPOCH*0.1);
        val_state_loss+=state_loss/(ITER_PER_EPOCH*0.1); val_pos_loss+=pos_loss/(ITER_PER_EPOCH*0.1)
        val_vel_loss+= vel_loss / (ITER_PER_EPOCH*0.1); val_a_loss+= a_loss / (ITER_PER_EPOCH*0.1)
    vae_model.train()

    print ('Epoch:', epoch, ' Total_loss:', '%.2f'%tr_total_loss, '/', '%.2f' % val_total_loss, ' FM:', '%.2f'%tr_fm_loss,'/', '%.2f' % val_fm_loss, ' State:', '%.2f'%tr_state_loss,'/', '%.2f' % val_state_loss,
           'Action', '%.2f'%tr_a_loss,'/', '%.2f' % val_a_loss, 'Pos:','%.2f'%tr_pos_loss,'/', '%.2f' % val_pos_loss, ' Vel:','%.2f'%tr_vel_loss,'/', '%.2f' % val_vel_loss)

    val_scores.append(val_total_loss); train_scores.append(tr_total_loss)
    if epoch > 100 and val_total_loss <= min(val_scores): torch.save(vae_model.state_dict(), 'R_VAE/'+best_valid)


    tracker.update([tr_total_loss, tr_fm_loss, tr_state_loss, tr_pos_loss, tr_vel_loss, tr_a_loss, val_total_loss, val_fm_loss, val_state_loss, val_pos_loss, val_vel_loss, val_a_loss], epoch)