import torch
import torch.nn as nn
class LossFunction(nn.Module):
    def __init__(self,embedding_dim,kernel_mul = 2.0, kernel_num = 5, **kwargs):
        super(LossFunction, self).__init__()
        self.kernel_num = kernel_num
        self.kernel_mul = kernel_mul
        self.fix_sigma = None
        self.embedding_dim = embedding_dim

    def forward(self, x, label,genre_label):

        source = x[::2].reshape(-1,self.embedding_dim)
        target = x[1::2].reshape(-1,self.embedding_dim)
        
        n_samples = int(source.size()[0])+int(target.size()[0])
        total = torch.cat([source, target], dim=0)
        total0 = total.unsqueeze(0).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        total1 = total.unsqueeze(1).expand(int(total.size(0)), int(total.size(0)), int(total.size(1)))
        L2_distance = ((total0-total1)**2).sum(2) 
        if self.fix_sigma:
            bandwidth = self.fix_sigma
        else:
            bandwidth = torch.sum(L2_distance.data) / (n_samples**2-n_samples)
        bandwidth /= self.kernel_mul ** (self.kernel_num // 2)
        bandwidth_list = [bandwidth * (self.kernel_mul**i) for i in range(self.kernel_num)]
        kernel_val = [torch.exp(-L2_distance / bandwidth_temp) for bandwidth_temp in bandwidth_list]
        kernels = sum(kernel_val)
        x_n = int(source.size()[0])
        y_n = int(target.size()[0])
        xx = kernels[:x_n, :x_n]
        yy = kernels[y_n:, y_n:]
        xy = kernels[:x_n, y_n:]
        yx = kernels[y_n:, :x_n]
        # print('xx.shape',xx.shape)
        # print('yy.shape',yy.shape)
        # print('xy.shape',xy.shape)
        # print('yx.shape',yx.shape)
        
        loss = torch.mean(xx + yy - xy -yx)

        return loss
