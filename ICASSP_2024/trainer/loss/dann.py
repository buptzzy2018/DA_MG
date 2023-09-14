import torch
import torch.nn as nn
import torch.nn.functional as F

from   torch.autograd import Variable
import torch.nn.init as init
from .reverse import GradReverse

## DOMAIN CLASSIFIER ##
class LossFunction(nn.Module):
    def __init__(self,embedding_dim, **kwargs):
        super(LossFunction, self).__init__()
        self.in_feats = embedding_dim
        self.constant = 1.0
        self.ce = nn.CrossEntropyLoss()

        self.critic = nn.Sequential(
            nn.Linear(embedding_dim,embedding_dim),
            nn.LeakyReLU(),
            nn.Linear(embedding_dim,11)
            )
        for m in self.critic.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_uniform_(m.weight, a=0.01)
                init.constant_(m.bias, 0)

    def forward(self, x ,label,genre_label):

        genre_label = genre_label.reshape(-1)        
        x = x.reshape(-1, self.in_feats)
        x = GradReverse.grad_reverse(x, self.constant)
        x = self.critic(x)
        loss_d = self.ce(x,genre_label)

        return loss_d

