# coding=utf-8
import torch
import torch.nn as nn
import math
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from lib.modules.utils import l2norm

def positional_encoding_1d(d_model, length):
    """
    :param d_model: dimension of the model
    :param length: length of positions
    :return: length*d_model position matrix
    """
    if d_model % 2 != 0:
        raise ValueError("Cannot use sin/cos positional encoding with "
                         "odd dim (got dim={:d})".format(d_model))
    pe = torch.zeros(length, d_model)
    position = torch.arange(0, length).unsqueeze(1)
    div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) *
                          -(math.log(10000.0) / d_model)))
    pe[:, 0::2] = torch.sin(position.float() * div_term)
    pe[:, 1::2] = torch.cos(position.float() * div_term)

    return pe


class GPO(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(GPO, self).__init__()
        self.d_pe = d_pe
        self.d_hidden = d_hidden

        self.pe_database = {}
        self.gru = nn.GRU(self.d_pe, d_hidden, 1, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(self.d_hidden, 1, bias=False)

    def compute_pool_weights(self, lengths, features):
        max_len = int(lengths.max())
        pe_max_len = self.get_pe(max_len)
        pes = pe_max_len.unsqueeze(0).repeat(lengths.size(0), 1, 1).to(lengths.device)
        mask = torch.arange(max_len).expand(lengths.size(0), max_len).to(lengths.device)
        mask = (mask < lengths.long().unsqueeze(1)).unsqueeze(-1)
        pes = pes.masked_fill(mask == 0, 0)

        self.gru.flatten_parameters()
        packed = pack_padded_sequence(pes, lengths.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        padded = pad_packed_sequence(out, batch_first=True)
        out_emb, out_len = padded
        out_emb = (out_emb[:, :, :out_emb.size(2) // 2] + out_emb[:, :, out_emb.size(2) // 2:]) / 2
        scores = self.linear(out_emb)
        scores[torch.where(mask == 0)] = -10000

        weights = torch.softmax(scores / 0.1, 1)
        return weights, mask

    def forward(self, features, lengths):
        """
        :param features: features with shape B x K x D
        :param lengths: B x 1, specify the length of each data sample.
        :return: pooled feature with shape B x D
        """
        pool_weights, mask = self.compute_pool_weights(lengths, features)

        features = features[:, :int(lengths.max()), :]
        sorted_features = features.masked_fill(mask == 0, -10000)
        sorted_features = sorted_features.sort(dim=1, descending=True)[0]
        sorted_features = sorted_features.masked_fill(mask == 0, 0)

        pooled_features = (sorted_features * pool_weights).sum(1)
        return pooled_features, pool_weights

    def get_pe(self, length):
        """

        :param length: the length of the sequence
        :return: the positional encoding of the given length
        """
        length = int(length)
        if length in self.pe_database:
            return self.pe_database[length]
        else:
            pe = positional_encoding_1d(self.d_pe, length)
            self.pe_database[length] = pe
            return pe

class gpoEncoders(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(gpoEncoders, self).__init__()
        self.img_gpo = GPO(d_pe, d_hidden)
        self.txt_gpo = GPO(d_pe, d_hidden)

    def forward(self, imgs_emb, caps_emb, img_lens, cap_lens):
        imgs_emb, _ = self.img_gpo(imgs_emb, img_lens)
        imgs_emb = l2norm(imgs_emb, dim=-1)
        caps_emb, _ = self.txt_gpo(caps_emb, cap_lens)
        caps_emb = l2norm(caps_emb, dim=-1)
        sims = imgs_emb @ caps_emb.t()
        return sims

MASK = -1 # padding value
from lib.modules.utils import get_mask
class DetectingRegion(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(DetectingRegion, self).__init__()
        self.txt_gpo = GPO(d_pe, d_hidden)

    def forward(self, imgs_emb, caps_emb, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs_emb = imgs_emb[:,:max_r,:]
        caps_emb = caps_emb[:,:max_w,:]

        caps_emb, _ = self.txt_gpo(caps_emb, cap_lens)
        caps_emb = l2norm(caps_emb, dim=-1)

        imgs_emb = l2norm(imgs_emb, dim=-1)
        img_mask = get_mask(img_lens)
        sims = imgs_emb @ caps_emb.t()
        sims = sims.masked_fill(img_mask == 0, MASK)
        sims = sims.max(dim=-2)[0]
        return sims


class DetectingSemantic(nn.Module):
    def __init__(self, d_pe, d_hidden):
        super(DetectingSemantic, self).__init__()
        self.img_gpo = GPO(d_pe, d_hidden)

    def forward(self, imgs_emb, caps_emb, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs_emb = imgs_emb[:,:max_r,:]
        caps_emb = caps_emb[:,:max_w,:]

        imgs_emb, _ = self.img_gpo(imgs_emb, img_lens)
        imgs_emb = l2norm(imgs_emb, dim=-1)

        caps_emb = l2norm(caps_emb, dim=-1)
        cap_mask = get_mask(cap_lens)
        sims = caps_emb @ imgs_emb.t()
        sims = sims.masked_fill(cap_mask == 0, MASK)
        sims = sims.max(dim=-2)[0]
        return sims.t()