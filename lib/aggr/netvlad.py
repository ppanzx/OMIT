# coding=utf-8
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modules.utils import get_mask, l2norm

class NetVLAD(nn.Module):
    """NetVLAD layer implementation"""

    def __init__(self, num_clusters=64, dim=1024, alpha=1.0, normalize=True):
        """
        Args:
            num_clusters : int
                The number of clusters
            dim : int
                Dimension of descriptors
            alpha : float
                Parameter of initialization. Larger value is harder assignment.
            normalize_input : bool
                If true, descriptor-wise L2 normalization is applied to input.
        """
        super(NetVLAD, self).__init__()
        self.num_clusters = num_clusters
        self.dim = dim
        self.alpha = alpha
        self.normalize = normalize
        self.assign = nn.Linear(dim, num_clusters, bias=True)
        self.centroids = nn.Parameter(torch.rand(num_clusters, dim))
        # self._init_params()

    def _init_params(self):
        self.assign.weight = nn.Parameter(
            (2.0 * self.alpha * self.centroids).unsqueeze(-1).unsqueeze(-1)
        )
        self.assign.bias = nn.Parameter(
            - self.alpha * self.centroids.norm(dim=1)
        )

    def forward(self, feats, lens):
        B, L, D = feats.shape
        assert D==self.dim
        max_l = int(lens.max())
        feats = feats[:,:max_l,:]
        mask = get_mask(lens).squeeze()

        # soft-assignment
        soft_assign = self.assign(feats) # B x R x C
        soft_assign[mask==0] = -torch.inf
        soft_assign = F.softmax(soft_assign, dim=1).unsqueeze(dim=-1)

        # calculate residuals to each clusters
        residual = feats.unsqueeze(dim=-2).repeat(1,1,self.num_clusters,1) - self.centroids
        vlad = (soft_assign*residual).sum(dim=1)

        vlad = vlad.min(dim=-2)[0]
        if self.normalize:
            vlad = F.normalize(vlad, p=2, dim=-1)  # L2 normalize

        return vlad

class vladEncoders(nn.Module):
    def __init__(self, embed_size):
        super(vladEncoders, self).__init__()
        self.net_vlad = NetVLAD(num_clusters=32, dim=embed_size)

    def forward(self, imgs_emb, caps_emb, img_lens, cap_lens):
        imgs_emb = self.net_vlad(imgs_emb, img_lens)
        imgs_emb = l2norm(imgs_emb, dim=-1)
        caps_emb = self.net_vlad(caps_emb, cap_lens)
        caps_emb = l2norm(caps_emb, dim=-1)
        sims = imgs_emb @ caps_emb.t()
        return sims