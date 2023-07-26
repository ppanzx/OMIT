# 20230719 by panzx
# cross attention for image text matching

# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modules.utils import get_mask, get_fgsims, get_fgmask, l2norm, cosine_similarity, SCAN_attention

EPS = 1e-8 # epsilon 
MASK = -1 # padding value
INF = -math.inf

# cross-attention V1
# proposed by <Stacked Cross Attention for image text matching>
class SCAN(nn.Module):
    def __init__(self,smooth=9):
        super().__init__()
        self.labmda = smooth

    def forward(self, imgs, caps, img_lens, cap_lens):
        return self.xattn_score_t2i(imgs,caps,cap_lens)

    def xattn_score_t2i(self, images, captions, cap_lens, return_attn=False):
        """
        Images: (n_image, n_regions, d) matrix of images
        Captions: (n_caption, max_n_word, d) matrix of captions
        CapLens: (n_caption) array of caption lengths
        """
        attentions = []
        n_image = images.size(0)
        n_caption,max_len,_ = captions.shape
        similarities = torch.zeros(n_image,n_caption,max_len).to(device=captions.device)+MASK
        for i in range(n_caption):
            # Get the i-th text description
            n_word = int(cap_lens[i].item())
            cap_i = captions[i, :n_word, :].unsqueeze(0).contiguous()
            # --> (n_image, n_word, d)
            cap_i_expand = cap_i.repeat(n_image, 1, 1)
            """
                word(query): (n_image, n_word, d)
                image(context): (n_image, n_regions, d)
                weiContext: (n_image, n_word, d)
                attn: (n_image, n_region, n_word)
            """
            if return_attn:
                weiContext,attn = SCAN_attention(cap_i_expand, images,self.labmda)
                attentions.append(attn)
            else:
                weiContext,_ = SCAN_attention(cap_i_expand, images,self.labmda)
            cap_i_expand = cap_i_expand.contiguous()
            weiContext = weiContext.contiguous()
            # (n_image, n_word)
            col_sim = cosine_similarity(cap_i_expand, weiContext, dim=2)
            # col_sim = col_sim.mean(dim=1, keepdim=True)
            similarities[:, i, :n_word] = col_sim

        # (n_image, n_caption)
        if return_attn:return torch.cat(attentions, 0)
        else:return similarities

# cross-attention V2
# proposed by <Fine-grained image-text matching by cross-modal hard algining network>
class VSACN(nn.Module):
    def __init__(self,temperature = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]
        # sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        sims = get_fgsims(caps,imgs)[:,:,:max_w,:max_r] # Bt x Bi x L x K
        # mask = get_fgmask(img_lens,cap_lens)
        mask = get_fgmask(cap_lens,img_lens)

        # 
        img_mask = get_mask(img_lens)
        imgs = imgs.masked_fill(img_mask==0, 0) # Bi x K x D
        cap_mask = get_mask(cap_lens)
        caps = caps.masked_fill(cap_mask==0, 0) # Bt x L x D

        # reference to SCAN
        sims = nn.LeakyReLU(0.1)(sims)
        sims = sims.masked_fill(mask==0, 0)
        sims = l2norm(sims, -1)

        # calculate attention
        sims = sims / self.temperature
        sims = sims.masked_fill(mask==0, INF)
        sims = torch.softmax(sims,dim=-1)
        sims = sims.masked_fill(mask == 0, 0)
        sims = torch.matmul(sims,imgs) # (Bt x Bi x L x K) @ (Bi x K x D) -> (Bt x Bi x L x D)
        sims = torch.mul(sims.permute(1,0,2,3),caps).permute(1,0,2,3).sum(dim=-1) \
                    /(torch.norm(sims,p=2,dim=-1,keepdim=False)+EPS) # Bt x Bi x L

        mask = get_mask(cap_lens).permute(0,2,1).repeat(1,img_lens.size(0),1)
        sims = sims.masked_fill(mask==0, MASK).permute(1,0,2)
        # sims = sims.sum(dim=-1)/cap_lens.unsqueeze(dim=-1)
        # sims = sims.permute(1,0).contiguous()
        return sims


# cross-attention V3
# proposed by <X-pool: Cross-modal language-video attention for text-video retrieval>
class MultiHeadedAttention(nn.Module):
    def __init__(self, embed_dim, num_mha_heads):
        super(MultiHeadedAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_mha_heads
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

    
    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            o: num_vids x num_texts x embed_dim
        """
        num_texts, _ = text_embeds.shape
        # num_texts x embed_dim
        q = self.q_proj(text_embeds)
        q = q.reshape(num_texts, self.num_heads, self.head_dim)
        # num_heads x head_dim x num_texts
        q = q.permute(1,2,0)

        num_vids, num_frames, _ = video_embeds.shape
        # num_vids x num_frames x embed_dim
        k = self.k_proj(video_embeds)
        k = k.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x num_frames x head_dim
        k = k.permute(0,2,1,3)

        # num_vids x num_frames x embed_dim
        v = self.v_proj(video_embeds)
        v = v.reshape(num_vids, num_frames, self.num_heads, self.head_dim)
        # num_vids x num_heads x head_dim x num_frames
        v = v.permute(0,2,3,1)

        # num_vids x num_heads x num_frames x num_texts
        attention_logits = k @ q
        attention_logits = attention_logits / math.sqrt(self.head_dim)
        attention_weights = F.softmax(attention_logits, dim=2)

        # num_vids x num_heads x head_dim x num_texts
        attention = v @ attention_weights
        # num_vids x num_texts x num_heads x head_dim
        attention = attention.permute(0,3,1,2)
        attention = attention.reshape(num_vids, num_texts, self.embed_dim)

        # num_vids x num_texts x embed_dim
        o = self.out_proj(attention)
        return o


class Transformer(nn.Module):
    def __init__(self, embed_dim, num_mha_heads, dropout):
        super(Transformer, self).__init__()
        self.embed_dim = embed_dim

        self.cross_attn = MultiHeadedAttention(embed_dim, num_mha_heads)

        self.linear_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.dropout = nn.Dropout(dropout)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)


    def forward(self, text_embeds, video_embeds):
        """
        Input
            text_embeds: num_texts x embed_dim
            video_embeds: num_vids x num_frames x embed_dim
        Output
            out: num_vids x num_texts x embed_dim
        """
        text_embeds = self.layer_norm1(text_embeds)
        video_embeds = self.layer_norm1(video_embeds)

        # num_vids x num_texts x embed_dim
        attn_out = self.cross_attn(text_embeds, video_embeds)
        attn_out = self.layer_norm2(attn_out)

        linear_out = self.linear_proj(attn_out)
        out = attn_out + self.dropout(linear_out)
        out = self.layer_norm3(out)

        return out


class CAN(nn.Module):
    def __init__(self, embed_dim, num_heads=1):
        super(CAN, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        
        assert self.embed_dim % self.num_heads == 0
        self.head_dim = self.embed_dim // self.num_heads
        
        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)
            
        self.layer_norm1 = nn.LayerNorm(self.embed_dim)
        self.layer_norm2 = nn.LayerNorm(self.embed_dim)
        self.layer_norm3 = nn.LayerNorm(self.embed_dim)
        self.layer_norm4 = nn.LayerNorm(self.embed_dim)

        self._init_parameters()

    
    def _init_parameters(self):
        for name, param in self.named_parameters():
            if 'linear' in name or 'proj' in name:
                if 'weight' in name:
                    nn.init.eye_(param)
                elif 'bias' in name:
                    param.data.fill_(0.)

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]

        # 
        img_mask = get_mask(img_lens)
        imgs = imgs.masked_fill(img_mask==0, 0) # Bi x K x D
        cap_mask = get_mask(cap_lens)
        caps = caps.masked_fill(cap_mask==0, 0) # Bt x L x D

        q = self.q_proj(self.layer_norm1(caps))
        k = self.k_proj(self.layer_norm2(imgs))
        v = self.v_proj(self.layer_norm3(imgs))
        
        # sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        sims = get_fgsims(q, k)[:,:,:max_w,:max_r] # Bt x Bi x L x K
        # mask = get_fgmask(img_lens,cap_lens)
        mask = get_fgmask(cap_lens,img_lens)

        # reference to SCAN
        sims = sims / math.sqrt(self.head_dim)
        sims = sims.masked_fill(mask==0, INF)
        sims = F.softmax(sims,dim=-1)
        sims = sims.masked_fill(mask == 0, 0)
        sims = torch.matmul(sims,v) # (Bt x Bi x L x K) @ (Bi x K x D) -> (Bt x Bi x L x D)
        sims = self.out_proj(self.layer_norm4(sims))
        sims = torch.mul(sims.permute(1,0,2,3),q).permute(1,0,2,3).sum(dim=-1) \
                    /(torch.norm(sims,p=2,dim=-1,keepdim=False)+EPS) # Bt x Bi x L

        mask = get_mask(cap_lens).permute(0,2,1).repeat(1,img_lens.size(0),1)
        sims = sims.masked_fill(mask==0, MASK).permute(1,0,2)
        # sims = sims.sum(dim=-1)/cap_lens.unsqueeze(dim=-1)
        # sims = sims.permute(1,0).contiguous()
        return sims

def get_coding(coding_type, **args):
    alpha = args["opt"].alpha
    KNN = args["opt"].KNN
    embed_size = args["opt"].embed_size
    if coding_type=="SCAN":
        return SCAN()
    elif coding_type=="VSACN":
        return VSACN()
    elif coding_type=="CAN":
        return CAN(embed_size)
    else:
        raise ValueError("Unknown coding type: {}".format(coding_type))

if __name__=="__main__":
    pass