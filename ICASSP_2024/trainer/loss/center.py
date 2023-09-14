#! /usr/bin/python
# -*- encoding: utf-8 -*-
# Adapted from https://github.com/CoinCheung/pytorch-loss (MIT License)

import torch
import torch.nn as nn
import torch.nn.functional as F
try:
    from .utils import accuracy
except:
    from utils import accuracy

class LossFunction(nn.Module):
    def __init__(self, matric_type=1,**kwargs):
        super(LossFunction, self).__init__()

        self.mse = nn.MSELoss()

    def forward(self, x, label, genre_label):
        
        loss =  self.mse(x[:,0,:],x[:,1,:])

        return loss

