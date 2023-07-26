"""VSE modules"""
import os
import torch
import torch.nn as nn
import numpy as np
import torchtext
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from transformers import BertModel

from lib.modules.resnet import ResnetFeatureExtractor
from lib.modules.utils import l2norm, MLP, SelfAttention, Transformer
from clip import clip

import logging
logger = logging.getLogger(__name__)

def load_clip_to_cpu(clip_model_name):
    url = clip._MODELS[clip_model_name]
    model_path = clip._download(url)
    # ckpt_path = "~/project/pzx/dataset/hub/clip"
    # model_path = osp.expanduser(osp.join(ckpt_path,osp.basename(url)))

    try:
        # loading JIT archive
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None

    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")

    model = clip.build_model(state_dict or model.state_dict())

    return model

def get_text_encoder(vocab_size, embed_size, word_dim, num_layers, text_enc_type="bigru", 
                    use_bi_gru=True, no_txtnorm=False, **args):
    """A wrapper to text encoders."""
    if text_enc_type == "bigru":
        txt_enc = EncoderTextBigru(vocab_size, embed_size, word_dim, num_layers, use_bi_gru=use_bi_gru, no_txtnorm=no_txtnorm, **args)
    elif text_enc_type == "bert":
        txt_enc = EncoderTextBert(embed_size, no_txtnorm=no_txtnorm)
    elif text_enc_type == "clip":
        clip_model = load_clip_to_cpu("ViT-B/32")
        txt_enc = EncoderTextCLIP(clip_model, embed_size=embed_size, no_txtnorm=no_txtnorm)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(text_enc_type))
    return txt_enc


def get_image_encoder(img_dim, embed_size, precomp_enc_type='basic', backbone_source=None, 
                        backbone_path=None, no_imgnorm=False, visual_mask_ratio=0.2, **args):
    """A wrapper to image encoders."""
    if precomp_enc_type == 'backbone':
        backbone_cnn = ResnetFeatureExtractor(backbone_source, backbone_path, fixed_blocks=2)
        img_enc = EncoderImageFull(backbone_cnn, img_dim, embed_size, precomp_enc_type, no_imgnorm, visual_mask_ratio, **args)
    elif precomp_enc_type == 'clip':
        clip_model = load_clip_to_cpu("ViT-B/32")
        img_enc = EncoderImageCLIP(clip_model, embed_size=embed_size, no_imgnorm=no_imgnorm, mask_ratio=visual_mask_ratio)
    else:
        img_enc = EncoderImagePrecomp(img_dim, embed_size, precomp_enc_type, no_imgnorm, visual_mask_ratio, **args)
    return img_enc

class EncoderImageFull(nn.Module):
    def __init__(self, backbone_cnn, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, mask_ratio=0.2):
        super(EncoderImageFull, self).__init__()
        self.backbone = backbone_cnn
        self.backbone_freezed = False
        self.image_encoder = EncoderImagePrecomp(img_dim, embed_size, precomp_enc_type, no_imgnorm, mask_ratio)

    def forward(self, images, feat_lengths):
        """Extract image feature vectors."""
        base_features = self.backbone(images)

        base_features, feat_lengths = self.image_encoder(base_features, feat_lengths)

        return base_features, feat_lengths

    def freeze_backbone(self):
        for param in self.backbone.parameters():
            param.requires_grad = False
        logger.info('Backbone freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.backbone.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.backbone.set_fixed_blocks(fixed_blocks)
        self.backbone.unfreeze_base()
        logger.info('Backbone unfreezed, fixed blocks {}'.format(self.backbone.get_fixed_blocks()))


class EncoderImagePrecomp(nn.Module):
    def __init__(self, img_dim, embed_size, precomp_enc_type='basic', no_imgnorm=False, mask_ratio=0.2, **args):
        super(EncoderImagePrecomp, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.mask_ratio = mask_ratio
        self.precomp_enc_type = precomp_enc_type
        self.fc = nn.Linear(img_dim, embed_size)
        if precomp_enc_type=="basic":
            self.feedforward = nn.Identity()
        elif precomp_enc_type == "mlp" or precomp_enc_type=="backbone":
            self.feedforward = MLP(img_dim, embed_size // 2, embed_size, 2)
        elif precomp_enc_type=="selfattention":
            self.feedforward = SelfAttention(embed_size)
        elif precomp_enc_type=="transformer":
            self.feedforward = Transformer(embed_size)
        else:
            raise ValueError("Unknown precomp_enc_type: {}".format(precomp_enc_type))
        self.init_weights()

    def init_weights(self):
        """Xavier initialization for the fully connected layer
        """
        r = np.sqrt(6.) / np.sqrt(self.fc.in_features +
                                  self.fc.out_features)
        self.fc.weight.data.uniform_(-r, r)
        self.fc.bias.data.fill_(0)

    def forward(self, base_features, feat_lengths):
        """Extract image feature vectors."""
        if self.training:
            # Size Augmentation during training, randomly mask features
            base_features, mask, _ = self.random_masking(base_features, self.mask_ratio)
            feat_lengths = (mask==0).sum(dim=-1).to(dtype=torch.int64)
        else:
            feat_lengths = torch.zeros(base_features.size(0)).to(base_features.device)
            feat_lengths[:] = base_features.size(1)

        features = self.fc(base_features)

        if self.precomp_enc_type=="mlp" or self.precomp_enc_type=="backbone":
            features += self.feedforward(base_features)
        else:
            features = self.feedforward(features)

        if not self.no_imgnorm:
            features = l2norm(features, dim=-1)

        return features, feat_lengths

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

# Visual Model with VIT in CLIP
class EncoderImageCLIP(nn.Module):
    """
        Note: Only support for Visual Transformer
    """
    def __init__(self, clip_model, embed_size=1024, no_imgnorm=False, mask_ratio=0.2, fixed_blocks=2, **args):
        super(EncoderImageCLIP, self).__init__()
        self.embed_size = embed_size
        self.no_imgnorm = no_imgnorm
        self.mask_ratio = mask_ratio
        self.fixed_blocks = fixed_blocks
        self.dtype = clip_model.dtype

        self.visual = clip_model.visual
        if embed_size!=self.visual.output_dim:
            width = self.visual.proj.shape[0]
            scale = width ** -0.5
            # self.visual.proj = nn.Identity()
            self.visual.proj = nn.Parameter(scale * torch.randn(width, embed_size))

    def forward(self, x: torch.Tensor, feat_lengths, return_hidden=False):
        ## patchify
        x = self.visual.conv1(x.type(self.dtype))  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        ## random masking
        if self.training:
            x, mask, _ = self.random_masking(x, self.mask_ratio)
            feat_lengths = (mask==0).sum(dim=-1).to(dtype=torch.int64)
        else:
            feat_lengths = torch.zeros(x.size(0)).to(x.device)
            feat_lengths[:] = x.size(1)

        x = torch.cat([self.visual.class_embedding.to(x.dtype) + torch.zeros(x.shape[0], 1, x.shape[-1], 
                        dtype=x.dtype, device=x.device), x], dim=1)  # shape = [*, grid ** 2 + 1, width]
        x = x + self.visual.positional_embedding[:x.shape[1]].to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.visual.ln_post(x) 
        hidden = x.type(self.visual.proj.dtype) @ self.visual.proj

        if not self.no_imgnorm:
            hidden = l2norm(hidden, dim=-1)

        x = hidden[:, 0, :]

        if return_hidden:
            return x, feat_lengths, hidden

        return x, feat_lengths

    def random_masking(self, x, mask_ratio):
        """
        Perform per-sample random masking by per-sample shuffling.
        Per-sample shuffling is done by argsort random noise.
        x: [N, L, D], sequence
        """
        N, L, D = x.shape  # batch, length, dim
        len_keep = int(L * (1 - mask_ratio))
        
        noise = torch.rand(N, L, device=x.device)  # noise in [0, 1]
        
        # sort noise for each sample
        ids_shuffle = torch.argsort(noise, dim=1)  # ascend: small is keep, large is remove
        ids_restore = torch.argsort(ids_shuffle, dim=1)

        # keep the first subset
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).repeat(1, 1, D))

        # generate the binary mask: 0 is keep, 1 is remove
        mask = torch.ones([N, L], device=x.device)
        mask[:, :len_keep] = 0
        # unshuffle to get the binary mask
        mask = torch.gather(mask, dim=1, index=ids_restore)

        return x_masked, mask, ids_restore

    def unfreeze_base(self, ):
        assert (0 <= self.fixed_blocks < 4)
        if self.fixed_blocks == 3:
            for p in self.visual.transformer.resblocks[10:12].parameters(): p.requires_grad = False
            for p in self.visual.transformer.resblocks[8:10].parameters(): p.requires_grad = False
            for p in self.visual.transformer.resblocks[6:8].parameters(): p.requires_grad = False
            for p in self.visual.transformer.resblocks[4:6].parameters(): p.requires_grad = False
            for p in self.visual.transformer.resblocks[2:4].parameters(): p.requires_grad = False
            for p in self.visual.transformer.resblocks[0:2].parameters(): p.requires_grad = False
        elif self.fixed_blocks == 2:
            for p in self.visual.transformer.resblocks[10:12].parameters(): p.requires_grad = True
            for p in self.visual.transformer.resblocks[8:10].parameters(): p.requires_grad = True
            for p in self.visual.transformer.resblocks[6:8].parameters(): p.requires_grad = False
            for p in self.visual.transformer.resblocks[4:6].parameters(): p.requires_grad = False
            for p in self.visual.transformer.resblocks[2:4].parameters(): p.requires_grad = False
            for p in self.visual.transformer.resblocks[0:2].parameters(): p.requires_grad = False
        elif self.fixed_blocks == 1:
            for p in self.visual.transformer.resblocks[10:12].parameters(): p.requires_grad = True
            for p in self.visual.transformer.resblocks[8:10].parameters(): p.requires_grad = True
            for p in self.visual.transformer.resblocks[6:8].parameters(): p.requires_grad = True
            for p in self.visual.transformer.resblocks[4:6].parameters(): p.requires_grad = True
            for p in self.visual.transformer.resblocks[2:4].parameters(): p.requires_grad = False
            for p in self.visual.transformer.resblocks[0:2].parameters(): p.requires_grad = False
        elif self.fixed_blocks == 0:
            for p in self.visual.transformer.resblocks[10:12].parameters(): p.requires_grad = True
            for p in self.visual.transformer.resblocks[8:10].parameters(): p.requires_grad = True
            for p in self.visual.transformer.resblocks[6:8].parameters(): p.requires_grad = True
            for p in self.visual.transformer.resblocks[4:6].parameters(): p.requires_grad = True
            for p in self.visual.transformer.resblocks[2:4].parameters(): p.requires_grad = True
            for p in self.visual.transformer.resblocks[0:2].parameters(): p.requires_grad = True
            for p in self.parameters():
                p.requires_grad = True

        logger.info('Resnet backbone now has fixed blocks {}'.format(self.fixed_blocks))

    def freeze_base(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.visual.ln_post.parameters(): p.requires_grad = True
        self.visual.proj.requires_grad = True

    def set_fixed_blocks(self, fixed_blocks):
        self.fixed_blocks = fixed_blocks

    def get_fixed_blocks(self):
        return self.fixed_blocks

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        for p in self.visual.ln_post.parameters(): p.requires_grad = True
        self.visual.proj.requires_grad = True
        logger.info('Visual CLIP freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.visual.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.set_fixed_blocks(fixed_blocks)
        self.unfreeze_base()
        logger.info('Visual CLIP unfreezed, fixed blocks {}'.format(self.get_fixed_blocks()))


# Language Model with BiGRU
class EncoderTextBigru(nn.Module):
    def __init__(self, vocab_size, embed_size, word_dim, num_layers, use_bi_gru=True, no_txtnorm=False, **args):
        super(EncoderTextBigru, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # word embedding
        self.embed = nn.Embedding(vocab_size, word_dim)

        # caption embedding
        hidden_size = embed_size
        self.rnn = nn.GRU(word_dim, hidden_size, num_layers, batch_first=True, bidirectional=use_bi_gru)
        self.fc = nn.Linear(hidden_size, embed_size)
        self.init_weights(wemb_type=args["wemb_type"],word2idx=args["word2idx"],word_dim=word_dim)

    def init_weights(self, wemb_type="glove", word2idx=None, word_dim=300, cache_dir="~/.cache/torch/hub/"):
        if wemb_type is None or word2idx is None:
            nn.init.xavier_uniform_(self.embed.weight)
        else:
            cache_dir = os.path.expanduser(cache_dir+wemb_type)
            # Load pretrained word embedding
            if 'fasttext' == wemb_type.lower():
                wemb = torchtext.vocab.FastText(cache=cache_dir)
            elif 'glove' == wemb_type.lower():
                wemb = torchtext.vocab.GloVe(cache=cache_dir)
            else:
                raise Exception('Unknown word embedding type: {}'.format(wemb_type))
            assert wemb.vectors.shape[1] == word_dim

            # quick-and-dirty trick to improve word-hit rate
            missing_words = []
            for word, idx in word2idx.items():
                if word not in wemb.stoi:
                    word = word.replace('-', '').replace('.', '').replace("'", '')
                    if '/' in word:
                        word = word.split('/')[0]
                if word in wemb.stoi:
                    self.embed.weight.data[idx] = wemb.vectors[wemb.stoi[word]]
                else:
                    missing_words.append(word)
            ##
            self.embed.requires_grad = False
            logger.info('Words: {}/{} found in vocabulary; {} words missing'.format(
                len(word2idx) - len(missing_words), len(word2idx), len(missing_words)))

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        x_emb = self.embed(x)

        self.rnn.flatten_parameters()
        packed = pack_padded_sequence(x_emb, lengths.cpu(), batch_first=True)

        # Forward propagate RNN
        out, _ = self.rnn(packed)

        # Reshape *final* output to (batch_size, hidden_size)
        padded = pad_packed_sequence(out, batch_first=True)
        cap_emb, cap_len = padded
        cap_emb = (cap_emb[:, :, :cap_emb.size(2) // 2] + cap_emb[:, :, cap_emb.size(2) // 2:]) / 2

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb


# Language Model with BERT
class EncoderTextBert(nn.Module):
    def __init__(self, embed_size, no_txtnorm=False):
        super(EncoderTextBert, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm

        # self.bert = BertModel.from_pretrained('bert-base-uncased')
        root = os.path.expanduser("~/project/pzx/dataset/hub/transformers")
        self.bert = BertModel.from_pretrained(config=root,pretrained_model_name_or_path=root)
        
        self.linear = nn.Linear(768, embed_size)

    def forward(self, x, lengths):
        """Handles variable size captions
        """
        # Embed word ids to vectors
        bert_attention_mask = (x != 0).float()
        bert_emb = self.bert(x, bert_attention_mask)[0]  # B x N x D
        cap_len = lengths

        cap_emb = self.linear(bert_emb)

        # normalization in the joint embedding space
        if not self.no_txtnorm:
            cap_emb = l2norm(cap_emb, dim=-1)

        return cap_emb

# Language Model with Transformers in CLIP
class EncoderTextCLIP(nn.Module):
    def __init__(self, clip_model, embed_size=1024, no_txtnorm=False, fixed_blocks=2):
        super(EncoderTextCLIP, self).__init__()
        self.embed_size = embed_size
        self.no_txtnorm = no_txtnorm
        self.fixed_blocks = fixed_blocks
        self.dtype = clip_model.dtype
        
        self.transformer = clip_model.transformer
        self.token_embedding = clip_model.token_embedding
        self.positional_embedding = clip_model.positional_embedding
        self.ln_final = clip_model.ln_final

        transformer_width = clip_model.transformer.width
        output_dim = clip_model.text_projection.shape[0]
        if embed_size!=output_dim:
            self.text_projection = nn.Parameter(torch.empty(transformer_width, embed_size))
        else:
            self.text_projection = nn.Parameter(torch.empty(transformer_width, output_dim))

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=transformer_width ** -0.5)

    def forward(self, text, lengths, return_hidden=False):
        x = self.token_embedding(text).type(self.dtype)  # [batch_size, n_ctx, d_model]

        pos_emd = self.positional_embedding[:x.size(1), :].type(self.dtype)
        x = x + pos_emd
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD

        hidden = self.ln_final(x).type(self.text_projection.dtype) @ self.text_projection

        if not self.no_txtnorm:
            hidden = l2norm(hidden, dim=-1)

        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        x = hidden[torch.arange(hidden.shape[0]), text.argmax(dim=-1)]

        if return_hidden:
            return x, hidden

        return x

    def unfreeze_base(self):
        assert (0 <= self.fixed_blocks < 4)
        if self.fixed_blocks == 3:
            for p in self.transformer.resblocks[10:12].parameters(): p.requires_grad = False
            for p in self.transformer.resblocks[8:10].parameters(): p.requires_grad = False
            for p in self.transformer.resblocks[6:8].parameters(): p.requires_grad = False
            for p in self.transformer.resblocks[4:6].parameters(): p.requires_grad = False
            for p in self.transformer.resblocks[2:4].parameters(): p.requires_grad = False
            for p in self.transformer.resblocks[0:2].parameters(): p.requires_grad = False
        elif self.fixed_blocks == 2:
            for p in self.transformer.resblocks[10:12].parameters(): p.requires_grad = True
            for p in self.transformer.resblocks[8:10].parameters(): p.requires_grad = True
            for p in self.transformer.resblocks[6:8].parameters(): p.requires_grad = False
            for p in self.transformer.resblocks[4:6].parameters(): p.requires_grad = False
            for p in self.transformer.resblocks[2:4].parameters(): p.requires_grad = False
            for p in self.transformer.resblocks[0:2].parameters(): p.requires_grad = False
        elif self.fixed_blocks == 1:
            for p in self.transformer.resblocks[10:12].parameters(): p.requires_grad = True
            for p in self.transformer.resblocks[8:10].parameters(): p.requires_grad = True
            for p in self.transformer.resblocks[6:8].parameters(): p.requires_grad = True
            for p in self.transformer.resblocks[4:6].parameters(): p.requires_grad = True
            for p in self.transformer.resblocks[2:4].parameters(): p.requires_grad = False
            for p in self.transformer.resblocks[0:2].parameters(): p.requires_grad = False
        elif self.fixed_blocks == 0:
            for p in self.transformer.resblocks[10:12].parameters(): p.requires_grad = True
            for p in self.transformer.resblocks[8:10].parameters(): p.requires_grad = True
            for p in self.transformer.resblocks[6:8].parameters(): p.requires_grad = True
            for p in self.transformer.resblocks[4:6].parameters(): p.requires_grad = True
            for p in self.transformer.resblocks[2:4].parameters(): p.requires_grad = True
            for p in self.transformer.resblocks[0:2].parameters(): p.requires_grad = True
            for p in self.parameters():
                p.requires_grad = True

        logger.info('Resnet backbone now has fixed blocks {}'.format(self.fixed_blocks))

    def freeze_base(self):
        for p in self.parameters():
            p.requires_grad = False
        for p in self.ln_final.parameters(): p.requires_grad = True
        self.text_projection.requires_grad = True

    def set_fixed_blocks(self, fixed_blocks):
        self.fixed_blocks = fixed_blocks

    def get_fixed_blocks(self):
        return self.fixed_blocks

    def freeze_backbone(self):
        for param in self.parameters():
            param.requires_grad = False
        for p in self.ln_final.parameters(): p.requires_grad = True
        self.text_projection.requires_grad = True
        logger.info('Textual CLIP freezed.')

    def unfreeze_backbone(self, fixed_blocks):
        for param in self.transformer.parameters():  # open up all params first, then adjust the base parameters
            param.requires_grad = True
        self.set_fixed_blocks(fixed_blocks)
        self.unfreeze_base()
        logger.info('Textual CLIP unfreezed, fixed blocks {}'.format(self.get_fixed_blocks()))

# experiments on aggregation types
from lib.aggr.pooling import aveEncoders
from lib.aggr.gpo import gpoEncoders
from lib.aggr.pcme import PEM
from lib.aggr.coding import get_coding, get_pooling
from lib.aggr.ot import Wasserstain
class SimsEncoder(nn.Module):
    def __init__(self, coding_type, pooling_type, **args):
        super(SimsEncoder, self).__init__()
        self.opt = args["opt"]
        self.aggr_type=args["opt"].aggr_type

        if self.aggr_type=="ave":
            ## 1. average pooling
            self.ave = aveEncoders()
        elif self.aggr_type=="gpo":
            ## 2. gpo pooling
            self.gpo = gpoEncoders(32,32)
        elif self.aggr_type=="pem":
            self.pem = PEM()
        elif self.aggr_type=="coding":
            ## 3. coding
            self.coding = get_coding(coding_type, opt=self.opt)
            self.pooling = get_pooling(pooling_type, opt=self.opt)
        elif self.aggr_type=="ot":
            ## 4. optimal transport
            self.sinkhorn = Wasserstain(lamb=self.opt.alpha, iters=int(self.opt.belta))

    def forward(self, imgs_emb, caps_emb, img_lens, cap_lens):
        if self.aggr_type=="ave":
            ## 1. average pooling
            sims = self.ave(imgs_emb, caps_emb, img_lens, cap_lens)
        elif self.aggr_type=="gpo":
            ## 2. gpo pooling
            sims = self.gpo(imgs_emb, caps_emb, img_lens, cap_lens)
        elif self.aggr_type=="pem":
            sims = self.pem(imgs_emb, caps_emb, img_lens, cap_lens)
        elif self.aggr_type=="coding":
            ## 3. coding
            sims = self.coding(imgs_emb, caps_emb, img_lens, cap_lens)
            sims = self.pooling(sims)
        elif self.aggr_type=="ot":
            # 4. optimal transport
            sims = self.sinkhorn(imgs_emb, caps_emb, img_lens, cap_lens)
        elif self.aggr_type=="cosine":
            sims = imgs_emb @ caps_emb.t()
        return sims
    