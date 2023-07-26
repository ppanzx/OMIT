import torch
import torch.nn as nn
from lib.modules.utils import l2norm, get_mask

class AvePool(nn.Module):
    def __init__(self):
        super(AvePool, self).__init__()
    def forward(self, features, lengths):
        """
            add None in return item for matching the ouput of GPO
        """
        max_len = int(lengths.max())
        features = features[:,:max_len,:]

        mask = get_mask(lengths)
        features = features.masked_fill(mask == 0, 0)
        features = features.sum(dim=-2)/lengths.unsqueeze(dim=-1)
        return features,None

class MaxPool(nn.Module):
    def __init__(self):
        super(MaxPool, self).__init__()

    def forward(self, features, lengths):
        max_len = int(lengths.max())
        features = features[:,:max_len,:]

        mask = get_mask(lengths)
        features = features.masked_fill(mask == 0, -1e2)
        features = features.max(dim=-2)[0]
        return features,None

class AdaptiveMaxPooling(nn.Module):
    def __init__(self):
        super(AdaptiveMaxPooling, self).__init__()

    def forward(self, imgs_emb, caps_emb, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs_emb = imgs_emb[:,:max_r,:]
        caps_emb = caps_emb[:,:max_w,:]

        img_mask = get_mask(img_lens)
        imgs_emb = imgs_emb.masked_fill(img_mask==0, -1e3)
        imgs_emb = imgs_emb.max(dim=1)[0]

        cap_mask = get_mask(cap_lens)
        caps_emb = caps_emb.masked_fill(cap_mask==0, -1e3)
        caps_emb = caps_emb.max(dim=1)[0]

        imgs_emb = l2norm(imgs_emb, dim=-1)
        caps_emb = l2norm(caps_emb, dim=-1)
        sims = imgs_emb @ caps_emb.t()
        return sims

class asyMaxPooling(nn.Module):
    def __init__(self):
        super(asyMaxPooling, self).__init__()
        self.alpha = 0.1

    def forward(self, imgs_emb, caps_emb, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs_emb = imgs_emb[:,:max_r,:]
        caps_emb = caps_emb[:,:max_w,:]

        img_mask = get_mask(img_lens)
        imgs_emb = imgs_emb.masked_fill(img_mask==0, -1e3)
        imgs_emb = imgs_emb.max(dim=1)[0]

        cap_mask = get_mask(cap_lens)
        caps_emb = caps_emb.masked_fill(cap_mask==0, -1e3)
        caps_emb = caps_emb.max(dim=1)[0]

        imgs_emb = l2norm(imgs_emb, dim=-1)
        caps_emb = l2norm(caps_emb, dim=-1)
        # sims = imgs_emb @ caps_emb.t()

        sims = self.alpha*(imgs_emb @ caps_emb.t().detach())+(1-self.alpha)*(imgs_emb.detach() @ caps_emb.t())

        return sims

class AdaptiveMinPooling(nn.Module):
    def __init__(self):
        super(AdaptiveMinPooling, self).__init__()

    def forward(self, imgs_emb, caps_emb, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs_emb = imgs_emb[:,:max_r,:]
        caps_emb = caps_emb[:,:max_w,:]

        img_mask = get_mask(img_lens)
        imgs_emb = imgs_emb.masked_fill(img_mask==0, 1e3)
        imgs_emb = imgs_emb.min(dim=1)[0]

        cap_mask = get_mask(cap_lens)
        caps_emb = caps_emb.masked_fill(cap_mask==0, 1e3)
        caps_emb = caps_emb.min(dim=1)[0]

        imgs_emb = l2norm(imgs_emb, dim=-1)
        caps_emb = l2norm(caps_emb, dim=-1)
        sims = imgs_emb @ caps_emb.t()
        return sims

class AdaptiveAvePooling(nn.Module):
    def __init__(self):
        super(AdaptiveAvePooling, self).__init__()

    def forward(self, imgs_emb, caps_emb, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs_emb = imgs_emb[:,:max_r,:]
        caps_emb = caps_emb[:,:max_w,:]

        img_mask = get_mask(img_lens)
        imgs_emb = imgs_emb.masked_fill(img_mask==0, 0)
        imgs_emb = imgs_emb.sum(dim=1)/img_lens.unsqueeze(dim=-1)
        imgs_emb = l2norm(imgs_emb, dim=-1)

        cap_mask = get_mask(cap_lens)
        caps_emb = caps_emb.masked_fill(cap_mask==0, 0)
        caps_emb = caps_emb.sum(dim=1)/cap_lens.unsqueeze(dim=-1)
        caps_emb = l2norm(caps_emb, dim=-1)

        sims = imgs_emb @ caps_emb.t()
        return sims

class aveEncoders(nn.Module):
    def __init__(self,):
        super(aveEncoders, self).__init__()
        self.img_pool = AvePool()
        self.txt_pool = AvePool()

    def forward(self, imgs_emb, caps_emb, img_lens, cap_lens):
        imgs_emb, _ = self.img_pool(imgs_emb, img_lens)
        imgs_emb = l2norm(imgs_emb, dim=-1)
        caps_emb, _ = self.txt_pool(caps_emb, cap_lens)
        caps_emb = l2norm(caps_emb, dim=-1)
        sims = imgs_emb @ caps_emb.t()
        return sims