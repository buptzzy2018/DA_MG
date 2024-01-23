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

        self.matric_type = matric_type
        print('matric_type (0: corr, 1:cov)', matric_type)


    def forward(self, x, label, genre_label):

        input_x = x[::2]
        input_u = x[1::2]
        input_x = input_x.reshape(-1, x.shape[-1])
        input_u = input_u.reshape(-1, x.shape[-1])

        d = input_x.shape[1]
        xs_mean = torch.mean(input_x, dim=0, keepdim=True)
        xt_mean = torch.mean(input_u, dim=0, keepdim=True)
        
        # 计算中心化的特征
        source_centered = input_x - xs_mean
        target_centered = input_u - xt_mean
        
        # 计算协方差矩阵
        covariance_source =  torch.mm(source_centered.t(),source_centered)  / (input_x.shape[0] - 1)
        covariance_target = torch.mm(target_centered.t(),target_centered)  / (input_u.shape[0] - 1)        

        if self.matric_type==0:

            covariance_source_Diag =  torch.diag(covariance_source).unsqueeze(-1)
            covariance_source_Diag = torch.sqrt(torch.mm(covariance_source_Diag.T, covariance_source_Diag))
            covariance_source = covariance_source / covariance_source_Diag

            covariance_target_Diag =  torch.diag(covariance_target).unsqueeze(-1)
            covariance_target_Diag = torch.sqrt(torch.mm(covariance_target_Diag.T, covariance_target_Diag))
            covariance_target = covariance_target / covariance_target_Diag
       
        loss = torch.norm(covariance_source - covariance_target, p='fro') ** 2 / (4*d*d)
        # loss = torch.norm(covariance_source - covariance_target, p='fro') ** 2 
        # loss = torch.sum((covariance_source - covariance_target) ** 2)

        return loss

