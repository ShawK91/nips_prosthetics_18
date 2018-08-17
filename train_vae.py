import numpy as np, os
os.environ["CUDA_VISIBLE_DEVICES"]='3'
import torch
from core import vae_fm
import torch.utils.data as pt_data


# Hyperparameters for ConvVAE
z_dim = 100
batch_size = 1024
learning_rate = 1e-4
SEED = True


# Parameters for training
NUM_EPOCH = 1000000000000
model_save_path = "R_Skeleton/"

class Tracker(): #Tracker
    def __init__(self, save_foldername, vars_string, project_string):
        self.vars_string = vars_string; self.project_string = project_string
        self.foldername = save_foldername
        self.all_tracker = [[[],0.0,[]] for _ in vars_string] #[Id of var tracked][fitnesses, avg_fitness, csv_fitnesses]
        self.counter = 0
        self.conv_size = 10
        if not os.path.exists(self.foldername):
            os.makedirs(self.foldername)


    def update(self, updates, generation):
        self.counter += 1
        for update, var in zip(updates, self.all_tracker):
            if update == None: continue
            var[0].append(update)

        #Constrain size of convolution
        for var in self.all_tracker:
            if len(var[0]) > self.conv_size: var[0].pop(0)

        #Update new average
        for var in self.all_tracker:
            if len(var[0]) == 0: continue
            var[1] = sum(var[0])/float(len(var[0]))

        if self.counter % 4 == 0:  # Save to csv file
            for i, var in enumerate(self.all_tracker):
                if len(var[0]) == 0: continue
                var[2].append(np.array([generation, var[1]]))
                filename = self.foldername + self.vars_string[i] + self.project_string
                np.savetxt(filename, np.array(var[2]), fmt='%.3f', delimiter=',')

class Dataset(pt_data.Dataset):
  'Characterizes a dataset for PyTorch'
  def __init__(self, states, next_states, actions, rewards):
      #Init
      self.state = states.pin_memory()
      self.next_state = next_states.pin_memory()
      self.action = actions.pin_memory()
      self.reward = rewards.pin_memory()


  def __len__(self):
        'Denotes the total number of samples'
        return len(self.state)

  def __getitem__(self, index):
        'Generates one sample of data'
        # Select sample
        return self.state[index], self.next_state[index], self.action[index], self.reward[index]

def load_data():
    all_s = []; all_ns = []; all_a = []; all_r = []

    ######## READ DATA #########
    while len(all_s) == 0:
        for tag in range(0, 100):
            try:
                data = np.load(model_save_path + 'numpy_buffer_' + str(tag) + '.npz')
                all_s.append(torch.Tensor(data['state']))
                all_ns.append(torch.Tensor(data['next_state']))
                all_a.append(torch.Tensor(data['action']))
                all_r.append(torch.Tensor(data['reward']))

            except: break


    all_s = torch.cat(all_s); all_ns = torch.cat(all_ns); all_a = torch.cat(all_a); all_r = torch.cat(all_r)

    ############ SHUFFLE DATA ##########
    shuffle_ind = torch.randperm(all_s.shape[0])
    all_s = all_s[shuffle_ind]; all_ns = all_ns[shuffle_ind]; all_a = all_a[shuffle_ind]; all_r = all_r[shuffle_ind]

    ########### CREATE DATASETS ############
    split = int(all_s.shape[0]*0.9)
    train_dataset = Dataset(all_s[0:split], all_ns[0:split], all_a[0:split], all_r[0:split])
    test_dataset = Dataset(all_s[split:], all_ns[split:], all_a[split:], all_r[split:])

    train_generator = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                                     pin_memory=True)
    test_generator = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
                                                       pin_memory=True)
    print ('LOADED DATASET', tag, 'WITH', len(train_dataset.reward), 'TRAIN AND', len(test_dataset.reward), 'TEST EXAMPLES')
    return train_generator, test_generator

############ LOAD DATA AND CONSTRUCT GENERATORS ########
training_generator, test_generator = load_data()
tracker = Tracker('R_VAE/', ['recon', 'kl', 'reward_recon', 'action_recon', 'valid_recon', 'valid_kl', 'valid_reward_recon', 'valid_action_recon'], '_score.csv')  # Initiate tracker


######################TRAIN VAE########################

vae_model = vae_fm.VAE(158, 19, z_dim).cuda()
if SEED:
    vae_model.load_state_dict(torch.load('R_VAE/vae_model'))
    print ('MODEL LOADED FROM R_VAE')
else: print ('NO SEED - STARTING FROM SCRATCH')

optimizer = torch.optim.Adam(vae_model.parameters(), lr=learning_rate, weight_decay=0.001)
total_params = sum(p.numel() for p in vae_model.parameters())
print ('Running VAE with', total_params, 'params')

############ TRAIN LOOP #################
for epoch in range(1, NUM_EPOCH):

    #REPLOAD DATA PERIODICALLY
    #if epoch % 1000 == 0: training_generator, test_generator = load_data()

    #SAVE MODEL PERIODICALLY
    if epoch % 500 == 0: torch.save(vae_model.state_dict(), 'R_VAE/vae_model')

    ep_rec_loss, ep_kl_loss, ep_r_loss, ep_a_loss = 0.0, 0.0, 0.0, 0.0
    for s, ns, a, r in training_generator:
        s = s.cuda(); ns = ns.cuda(); a = a.cuda(); r = r.cuda()

        z, next_z, dec_state, dec_next_state, mu, logvar = vae_model.forward(s, a) #FM FORWARD PASS
        r_pred, a_pred = vae_model.action_reward_predictor(z, next_z) #REWARD_ACTION FORWARD PASS

        #### COMPUTE LOSS #####
        fm_loss_T, rec_loss, kl_loss = vae_model.fm_recon_loss(s, dec_state, ns, dec_next_state, mu, logvar) #FM LOSS
        ar_loss_T, r_loss, a_loss = vae_model.ar_pred_loss(r, r_pred, a, a_pred)

        ep_rec_loss += rec_loss/training_generator.dataset.__len__(); ep_kl_loss+=kl_loss/training_generator.dataset.__len__(); ep_a_loss+=a_loss/training_generator.dataset.__len__(); ep_r_loss+=r_loss/training_generator.dataset.__len__()

        optimizer.zero_grad()
        fm_loss_T.backward(retain_graph=True)
        ar_loss_T.backward()
        #torch.nn.utils.clip_grad_norm_(vae_model.parameters(), 10)
        optimizer.step()

    # #Validation loss from test set
    val_recloss, val_kl_loss, val_r_loss, val_a_loss = 0.0, 0.0, 0.0, 0.0
    for s, ns, a, r in test_generator:
        s = s.cuda(); ns = ns.cuda(); a = a.cuda(); r = r.cuda()

        z, next_z, dec_state, dec_next_state, mu, logvar = vae_model.forward(s, a) #FM FORWARD PASS
        r_pred, a_pred = vae_model.action_reward_predictor(z, next_z) #REWARD_ACTION FORWARD PASS

        #### COMPUTE LOSS #####
        _, rec_loss, kl_loss = vae_model.fm_recon_loss(s, dec_state, ns, dec_next_state, mu, logvar) #FM LOSS
        _, r_loss, a_loss = vae_model.ar_pred_loss(r, r_pred, a, a_pred)

        val_recloss += rec_loss/test_generator.dataset.__len__(); val_kl_loss+=kl_loss/test_generator.dataset.__len__(); val_a_loss+=a_loss/test_generator.dataset.__len__(); val_r_loss+=r_loss/test_generator.dataset.__len__() #*9 is to normalize between the number of train/test examples being summed over



    print ('Epoch:', epoch, ' Rec_loss:', '%.2f'%ep_rec_loss, ' KL_loss', '%.2f'%ep_kl_loss, 'R_loss', '%.2f'%ep_r_loss, 'A_loss','%.2f'%ep_a_loss, ' V_Rec_loss:', '%.2f'%val_recloss, ' V_KL_loss', '%.2f'%val_kl_loss, 'V_R_loss', '%.2f'%val_r_loss, 'V_A_loss','%.2f'%val_a_loss )
    tracker.update([ep_rec_loss, ep_kl_loss, ep_r_loss,ep_a_loss, val_recloss, val_kl_loss, val_r_loss, val_a_loss], epoch)