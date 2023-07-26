""" Uncertainty modules
Reference code:
    PCME in
    https://github.com/naver-ai/pcme/blob/main/models/uncertainty_module.py
"""

import numpy as np

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from lib.modules.utils import l2norm, get_mask
from lib.aggr.pooling import AvePool

class MultiHeadSelfAttention(nn.Module):
    """Self-attention module by Lin, Zhouhan, et al. ICLR 2017"""

    def __init__(self, n_head, d_in, d_hidden):
        super(MultiHeadSelfAttention, self).__init__()

        self.n_head = n_head
        self.w_1 = nn.Linear(d_in, d_hidden, bias=False)
        self.w_2 = nn.Linear(d_hidden, n_head, bias=False)
        self.tanh = nn.Tanh()
        self.softmax = nn.Softmax(dim=1)
        self.init_weights()

    def init_weights(self):
        nn.init.xavier_uniform_(self.w_1.weight)
        nn.init.xavier_uniform_(self.w_2.weight)

    def forward(self, x, mask=None):
        # This expects input x to be of size (b x seqlen x d_feat)
        attn = self.w_2(self.tanh(self.w_1(x)))
        if mask is not None:
            mask = mask.repeat(self.n_head, 1, 1).permute(1, 2, 0)
            attn.masked_fill_(mask, -np.inf)
        attn = self.softmax(attn)

        output = torch.bmm(attn.transpose(1, 2), x)
        if output.shape[1] == 1:
            output = output.squeeze(1)
        return output, attn

class PEM(nn.Module):
    def __init__(self, d_in=1024, heads=1):
        super(PEM, self).__init__()
        self.ave_pool = AvePool()
        self.img_attn = MultiHeadSelfAttention(heads, d_in, d_in//2)
        self.cap_attn = MultiHeadSelfAttention(heads, d_in, d_in//2)

    def forward(self, imgs, caps, img_lens, cap_lens,):
        n_img = imgs.shape[0]
        n_cap = caps.shape[0]
        img_max_len = int(img_lens.max())
        imgs = imgs[:,:img_max_len,:]
        cap_max_len = int(cap_lens.max())
        caps = caps[:,:cap_max_len,:]

        img_mu,_ = self.ave_pool(imgs, img_lens)
        img_mu = l2norm(img_mu, dim=-1)
        # cap_mu = torch.gather(caps, 1, cap_lens).squeeze(1)
        cap_mu,_ = self.ave_pool(caps, cap_lens)
        cap_mu = l2norm(cap_mu, dim=-1)

        img_mask = get_mask(img_lens).squeeze()==0
        cap_mask = get_mask(cap_lens).squeeze()==0

        # img_log_sigma
        residual, attn = self.img_attn(imgs, img_mask)
        img_logsigma = img_mu + residual
        img_logsigma = l2norm(img_logsigma, dim=-1)
        img_sigma = img_logsigma.unsqueeze(dim=1).repeat(1,n_cap,1).exp()
        
        img_mu = img_mu.unsqueeze(dim=1).repeat(1,n_cap,1)

        # cap_sigma
        residual, attn = self.cap_attn(caps, cap_mask)
        cap_logsigma = cap_mu + residual
        cap_logsigma = l2norm(cap_logsigma, dim=-1)
        cap_sigma = cap_logsigma.unsqueeze(dim=0).repeat(n_img,1,1).exp()
        cap_mu = cap_mu.unsqueeze(dim=0).repeat(n_img,1,1)

        # wasserstain distance
        div = self.wasserstain(img_mu, img_sigma, cap_mu, cap_sigma)
        return div

    def wasserstain(self, mu_1, sigma_1, mu_2, sigma_2):
        mu_norm = torch.norm(mu_1-mu_2, p=2, dim=-1)
        sigma_norm = torch.norm(sigma_1-sigma_2, p=2, dim=-1)
        wd = torch.sqrt(mu_norm**2+sigma_norm**2)
        return -wd