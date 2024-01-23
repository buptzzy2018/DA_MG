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
    def __init__(self, consistency_loss,**kwargs):
        super(LossFunction, self).__init__()

        self.matric_type = consistency_loss['matric_type']
        self.weight_in = consistency_loss['weight_in']
        self.weight_bet = consistency_loss['weight_bet']
        print('matric_type (0: corr 1:cov)', self.matric_type)
        print('weight_in', self.weight_in)
        print('weight_bet', self.weight_bet)


    def compute_various(self, x):

        tol_num = x.size(0)*x.size(1)
        x_mean = torch.mean(x, dim=1,keepdim=False)  
        mean = torch.mean(x.reshape(-1,x.size(-1)) ,dim=0,keepdim=False)
        within = torch.zeros((x.size(-1),x.size(-1)),device = mean.device)
        for i in range(x.size(0)):
            res_x =  x[i] - x_mean[i]
            sum1 = torch.mm(res_x.T, res_x)
            within = sum1 + within
        within = within / tol_num
        res_mean = x_mean - mean
        sum2 = x.size(1) * torch.mm(res_mean.T, res_mean)
        between = sum2 / tol_num

        return within, between


    def forward(self, x, label, genre_label):
        d = x.shape[-1]
        input_x = x[::2]
        input_u = x[1::2]

        x1_within, x1_between = self.compute_various(input_x)
        x2_within, x2_between = self.compute_various(input_u)
        

        if self.matric_type==0:

            x1_Diag =  torch.diag(x1_within).unsqueeze(-1)
            x1_Diag = torch.sqrt(torch.mm(x1_Diag.T, x1_Diag))
            x1_within = x1_within / x1_Diag
            x2_Diag =  torch.diag(x2_within).unsqueeze(-1)
            x2_Diag = torch.sqrt(torch.mm(x2_Diag.T, x2_Diag))
            x2_within = x2_within / x2_Diag
            x1_Diag_bet =  torch.diag(x1_between).unsqueeze(-1)
            x1_Diag_bet = torch.sqrt(torch.mm(x1_Diag_bet.T, x1_Diag_bet))
            x1_between = x1_between / x1_Diag_bet
            x2_Diag_bet =  torch.diag(x2_between).unsqueeze(-1)
            x2_Diag_bet = torch.sqrt(torch.mm(x2_Diag_bet.T, x2_Diag_bet))
            x2_between = x2_between / x2_Diag_bet


        within = torch.sum((x1_within - x2_within) ** 2)
        between = torch.sum((x1_between - x2_between) ** 2)
        loss  =  self.weight_in * within +  self.weight_bet * between

        
        return loss

