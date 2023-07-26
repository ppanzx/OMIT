""" Optimal Transport module"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modules.utils import get_mask,get_fgsims,get_fgmask
INF = math.inf

# fine-grained sinkhorn distance
class Wasserstain(nn.Module):
    def __init__(self, iters=3, lamb=5e-2, split=1, yokoi_init=False):
        super(Wasserstain, self).__init__()
        self.eps = 1e-6
        self.iters = iters
        self.lamb = lamb
        self.split = split
        self.yokoi_init = yokoi_init

    def _ave_init(self, mask):
        r = mask.sum(dim=[-1])!=0        
        c = mask.sum(dim=[-2])!=0
        r = r*(1/r.sum(dim=-1, keepdim=True))
        c = c*(1/c.sum(dim=-1, keepdim=True))
        return r, c

    def _yokoi_init(self, features, mask):
        """
            <Word Rotator’s Distance>

        """
        # max_len = int(lengths.max())
        # features = features[:,:max_len,:]
        weight = torch.norm(features,p=2,dim=-1,keepdim=True)
        weight = weight.masked_fill(mask == 0, 0)
        weight = weight/weight.sum(dim=1,keepdim=True)
        return weight

    def Sinkhorn_Knopp(self, sims, r, c):
        """
        Computes the optimal transport matrix and Slinkhorn distance using the
        Sinkhorn-Knopp algorithm
        """
        P = torch.exp(-1 / self.lamb * sims)    
        # Avoiding poor math condition
        P = P / (P.sum(dim=[-2,-1], keepdim=True)+self.eps)

        # Normalize this matrix so that P.sum(-1) == r, P.sum(-2) == c
        for i in range(self.iters):
            # Shape (n, )
            u = P.sum(dim=[-1],) + self.eps # u(0)
            P = P * (r / u).unsqueeze(dim=-1) # u(0)*
            v = P.sum(dim=[-2]) + self.eps
            P = P * (c / v).unsqueeze(dim=-2)
            # if (u - P.sum(dim=[-1],)).max()<self.eps or \
            #     (v - P.sum(dim=[-2],)).max()<self.eps:
            #     break
        return P

    def forward(self, imgs, caps, img_lens, cap_lens, return_attn=False):
        bi, bt = imgs.size(0), caps.size(0)
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        fg_sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        if return_attn: attn = torch.zeros_like(fg_sims,device=fg_sims.device)

        sims = torch.zeros(bi, bt).to(device=caps.device)
        step = bi//self.split
        for i in range(self.split):
            beg = step*i
            ed = bi if i+1==self.split else step*(i+1) 
            if self.yokoi_init:
                img_mask = get_mask(img_lens)
                cap_mask = get_mask(cap_lens)
                r = self._yokoi_init(imgs[beg:ed,:int(img_lens.max())], img_mask[beg:ed])
                r = r.permute(0,2,1).repeat(1,caps.shape[0],1)
                c = self._yokoi_init(caps[:,:int(cap_lens.max())], cap_mask)
                c = c.permute(2,0,1).repeat(imgs[beg:ed].shape[0],1,1)
            else:
                r,c = self._ave_init(mask[beg:ed])
            tp = self.Sinkhorn_Knopp((1-fg_sims[beg:ed]).masked_fill(mask[beg:ed] == 0, INF), r, c)
            sims[beg:ed] = (fg_sims[beg:ed]*tp*mask[beg:ed]).sum(dim=[-2,-1])
            if return_attn: attn[beg:ed]=tp
        if return_attn: return sims, attn
        else: return sims

if __name__=="__main__":
    batch_size = 128
    imgs = torch.rand([batch_size,36,1024])
    caps = torch.rand([batch_size,51,1024])
    img_lens = torch.randint(20, 37, [batch_size])
    cap_lens = torch.randint(36, 52, [batch_size])
    model = Wasserstain(yokoi_init=True)
    sims = model(imgs, caps, img_lens, cap_lens)