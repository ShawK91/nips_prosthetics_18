import torch.nn as nn
import torch.nn.functional as F
import torch

class Actor(nn.Module):

    def __init__(self, args):
        super(Actor, self).__init__()
        self.args = args
        l1 = 256; l2 = 256

        # Construct Hidden Layer 1
        self.f1 = nn.Linear(args.state_dim, l1)
        self.ln1 = nn.LayerNorm(l1)

        #Hidden Layer 2
        self.f2 = nn.Linear(l1, l2)
        self.ln2 = nn.LayerNorm(l2)

        #Out
        self.w_out = nn.Linear(l2, args.action_dim)

    def forward(self, input):

        #Hidden Layer 1
        out = F.elu(self.f1(input))
        out = self.ln1(out)

        #Hidden Layer 2
        out = F.elu(self.f2(out))
        out = self.ln2(out)

        #Out
        return torch.sigmoid(self.w_out(out))