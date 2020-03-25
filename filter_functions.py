import torch
import torch.nn as nn


class AXA(nn.Module):
    def __init__(self,dim_feature = 256,dim_filter = 3, device='cuda:1'):
        super(AXA, self).__init__()
        self.dim_feature = dim_feature
        self.dim_filter = dim_filter
        #self.A = 
        self.A = nn.Parameter(0.1 * torch.randn(self.dim_filter, self.dim_feature)-0.05)
        #self.A = torch.zeros(self.dim_filter, self.dim_feature).requires_grad_(True)
        #self.A = self.A.to(device)
        
    def forward(self, X):
        #print(self.A.size())
        #print(X.size())
        #print(self.A)
        AX = self.A@X
        return AX@torch.transpose(self.A,0,1)

class BXtBXB(nn.Module):
    def __init__(self,dim_feature = 256,dim_filter = 3):
        super(BXtBXB, self).__init__()
        self.dim_feature = dim_feature
        self.dim_filter = dim_filter
        self.B = torch.randn(self.dim_filter, self.dim_feature).requires_grad_(True)
        
    def forward(self, X):
        return self.B@torch.transpose(X)
        AX = self.A@X
        return AX@torch.transpose(self.A,0,1)

        
class tanhAXA(nn.Module):
    def __init__(self,dim_feature = 256,dim_filter = 3, device='cuda:1'):
        super(tanhAXA, self).__init__()
        self.dim_feature = dim_feature
        self.dim_filter = dim_filter
        #self.A = 
        self.A = nn.Parameter(0.1 * torch.randn(self.dim_filter, self.dim_feature)-0.05)
        #self.A = nn.Parameter(self.A)
        #self.A = torch.zeros(self.dim_filter, self.dim_feature).requires_grad_(True)
        #self.A = self.A.to(device)
        self.tanh = nn.Tanh()
        
    def forward(self, X):
        #print(self.A.size())
        #print(X.size())
        #print(self.A)
        AX = self.A@X
        return self.tanh(AX@torch.transpose(self.A,0,1))
