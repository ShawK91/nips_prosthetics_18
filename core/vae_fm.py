import torch
from torch import nn
import torch.nn.functional as F

#PYTORCH
class VAE(nn.Module):
    def __init__(self, state_dim, action_dim, z_dim):
        super(VAE, self).__init__()

        #State Encoder
        self.enc1 = nn.Linear(state_dim, 256)
        self.enc2 = nn.Linear(256, 128)
        self.enc_mu = nn.Linear(128, z_dim)
        self.enc_var = nn.Linear(128, z_dim)
        self.enc_var.weight.data.mul_(0.001) #Force small weights so the KL doesn;t blow up

        #Decoder
        self.dec1 = nn.Linear(z_dim, 256)
        self.dec2 = nn.Linear(256, 200)
        self.dec3 = nn.Linear(200, state_dim)

        ################### FORWARD MODEL ################
        self.action_embedder = nn.Linear(action_dim, 128)
        self.action_embedder2 = nn.Linear(128, 64)
        self.fm1 = nn.Linear(64+z_dim, 256)
        self.fm2 = nn.Linear(256, 200)
        self.fm3 = nn.Linear(200, z_dim)


        ################# REWARD AND ACTION REGULARIZER ################
        self.action_head1 = nn.Linear(z_dim*3, 256)
        self.action_head2 = nn.Linear(256, 128)
        self.action_head3 = nn.Linear(128, action_dim)

        self.reward_head1 = nn.Linear(z_dim*3, 256)
        self.reward_head2 = nn.Linear(256, 128)
        self.reward_head3 = nn.Linear(128, 1)

    def encode_state(self, x):
        h = F.elu(self.enc1(x))
        h = F.elu(self.enc2(h))
        return self.enc_mu(h), torch.sigmoid(self.enc_var(h))


    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std  + mu

    def decode_state(self, z):
        h = F.elu(self.dec1(z))
        h = F.elu(self.dec2(h))
        h = self.dec3(h)
        return h

    def forward(self, state, action):
        #State Encoder and decoder
        mu, logvar = self.encode_state(state)
        z = self.reparameterize(mu, logvar)
        dec_state = self.decode_state(z)

        ######### FORWARD MODEL #######
        #Action embedder
        action_emb = F.elu(self.action_embedder(action))
        action_emb = F.elu(self.action_embedder2(action_emb))

        h = torch.cat([action_emb, z], 1)
        h = F.elu(self.fm1(h))
        h = F.elu(self.fm2(h))
        next_z = (self.fm3(h))

        dec_next_state = self.decode_state(next_z)

        return z, next_z, dec_state, dec_next_state, mu, logvar

    def action_reward_predictor(self, z, next_z):
        diff = z - next_z
        h = torch.cat([z, next_z, diff], 1)

        #Reward HEAD
        r = F.elu(self.reward_head1(h))
        r = F.elu(self.reward_head2(r))
        r = self.reward_head3(r)

        #ACTION HEAD
        a = F.elu(self.action_head1(h))
        a = F.elu(self.action_head2(a))
        a = torch.tanh(self.action_head3(a))

        return r, a


    def fm_recon_loss(self, state, dec_state, next_state, dec_next_state, mu, logvar, beta=1):
        #RECONSTRUCTION LOSSES
        rec_loss = (torch.mean(torch.nn.functional.mse_loss(state, dec_state)) + torch.mean(torch.nn.functional.mse_loss(next_state, dec_next_state)))
        kld = 0.001 * (-0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp()))
        return rec_loss + beta * kld, rec_loss.item(), kld.item()

    def ar_pred_loss(self, r, r_pred, a, a_pred):
        r_recon = torch.mean(torch.nn.functional.mse_loss(r, r_pred))
        a_recon = (1.0/19) * torch.mean(torch.nn.functional.mse_loss(a, a_pred))
        return r_recon + a_recon, r_recon.item(), a_recon.item()









