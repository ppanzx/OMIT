# coding=utf-8
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from lib.modules.utils import get_mask, get_fgsims, get_fgmask, l2norm, cosine_similarity, SCAN_attention

EPS = 1e-8 # epsilon 
MASK = -1 # padding value
INF = -math.inf

# orignal version of scan
class T2ICrossAttentionPool(nn.Module):
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
        similarities = []
        attentions = []
        n_image = images.size(0)
        n_caption = captions.size(0)
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
            col_sim = col_sim.mean(dim=1, keepdim=True)
            similarities.append(col_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1)
        if return_attn:return torch.cat(attentions, 0)
        else:return similarities

## probably due to the gradient accumlation of img_i, I2TCrossAttentionPool will raise error(CUDA out of memory).
class I2TCrossAttentionPool(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, caps, img_lens, cap_lens):
        return self.xattn_score_i2t(imgs,caps,img_lens)

    def xattn_score_i2t(self, images, captions, img_lens):
        """
        Images: (batch_size, n_regions, d) matrix of images
        Captions: (batch_size, max_n_words, d) matrix of captions
        CapLens: (batch_size) array of caption lengths
        """
        similarities = []
        n_image = images.size(0)
        n_caption = captions.size(0)
        for i in range(n_image):
            # Get the i-th text description
            n_region = int(img_lens[i].item())
            img_i = images[i,:n_region,:].unsqueeze(0).contiguous()
            # (n_caption, n_region, d)
            img_i_expand = img_i.repeat(n_caption, 1, 1)
            
            weiContext,_ = SCAN_attention(img_i_expand, captions)
            img_i_expand = img_i_expand.contiguous()
            weiContext = weiContext.contiguous()
            # (n_image, n_region)
            row_sim = cosine_similarity(img_i_expand, weiContext, dim=2)
            row_sim = row_sim.mean(dim=1, keepdim=True)
            similarities.append(row_sim)

        # (n_image, n_caption)
        similarities = torch.cat(similarities, 1).t()
        return similarities

class CollaborativeCoding(nn.Module):
    def __init__(self,tau=0.1):
        super().__init__()
        self.tau = tau
    
    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]

        # padding
        img_mask = get_mask(img_lens)
        imgs = imgs.masked_fill(img_mask==0, 0) # Bi x K x D
        cap_mask = get_mask(cap_lens)
        caps = caps.masked_fill(cap_mask==0, 0) # Bt x L x D

        # reshape
        Bi, K, D = imgs.shape
        Bt, L, D = caps.shape
        imgs = imgs.reshape(Bi*K,D).contiguous()
        caps = caps.reshape(Bt*L,D).contiguous()

        # collaborative coding
        # W' = (XX'+λI)XY'
        sims = torch.linalg.inv(imgs@imgs.t()+self.tau*torch.eye(Bi*K,device=imgs.device)) @imgs @caps.t()
        sims = sims.reshape(Bi, K, Bt, L).permute(2,0,3,1).contiguous() # (Bt x Bi x L x K)

        #
        imgs = imgs.reshape(Bi,K,D).contiguous()
        caps = caps.reshape(Bt,L,D).contiguous()

        # cosine similarity
        sims = torch.matmul(sims,imgs) # (Bt x Bi x L x K) @ (Bi x K x D) -> (Bt x Bi x L x D)
        sims = torch.mul(sims.permute(1,0,2,3),caps).permute(1,0,2,3).sum(dim=-1) \
                    /(torch.norm(sims,p=2,dim=-1,keepdim=False)+EPS) # Bt x Bi x L

        mask = get_mask(cap_lens).permute(0,2,1).repeat(1,img_lens.size(0),1)
        sims = sims.masked_fill(mask==0, MASK).permute(1,0,2)
        # sims = -(torch.norm(torch.matmul(sims,imgs).permute(1,0,2,3)-caps,p=2,dim=-1).mean(dim=-1))

        return sims

# Visual Hard Assignment Coding
class VHACoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        sims = sims.max(dim=-2)[0]
        return sims

# Texual Hard Assignment Coding
class THACoding(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)
        sims = sims.masked_fill(mask == 0, MASK)
        sims = sims.max(dim=-1)[0]
        return sims

EPS = -1e-8
# Texual Soft Assignment Coding
class TSACoding(nn.Module):
    def __init__(self,temperature = 0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        imgs = imgs[:,:max_r,:]
        caps = caps[:,:max_w,:]
        sims = get_fgsims(imgs, caps)[:,:,:max_r,:max_w]
        mask = get_fgmask(img_lens,cap_lens)

        # 
        img_mask = get_mask(img_lens)
        imgs = imgs.masked_fill(img_mask==0, 0)
        cap_mask = get_mask(cap_lens)
        caps = caps.masked_fill(cap_mask==0, 0)

        # sparse
        sims = nn.LeakyReLU(0.1)(sims)
        sims = sims.masked_fill(mask==0, 0)
        sims = l2norm(sims, -1)

        # calculate attention
        sims = sims / self.temperature
        sims = sims.masked_fill(mask==0, INF)
        sims = torch.softmax(sims,dim=-1) # Bi x Bt x K x L
        sims = sims.masked_fill(mask == 0, 0)
        sims = torch.matmul(sims,caps) # Bi x Bt x K x D
        sims = torch.mul(sims.permute(1,0,2,3),imgs).permute(1,0,2,3).sum(dim=-1) \
                    /(torch.norm(sims,p=2,dim=-1,keepdim=False)+EPS) # Bi x Bt x K

        mask = get_mask(img_lens).permute(0,2,1).repeat(1,cap_lens.size(0),1)
        sims = sims.masked_fill(mask==0, MASK)
        # sims = sims.sum(dim=-1)/img_lens.unsqueeze(dim=-1)
        return sims

# Visual Soft Assignment Coding
class VSACoding(nn.Module):
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

# visual sparse pooling
class VSCoding(nn.Module):
    def __init__(self,temperature = 0.1, KNN = 5):
        super().__init__()
        self.temperature = temperature
        self.KNN = KNN

    def forward(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        KNN = max_r if self.KNN > max_r else self.KNN
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

        # calculate attention
        sims = sims / self.temperature
        sims = sims.masked_fill(mask==0, INF)
        # sparse
        sims = sims.masked_fill(sims<torch.sort(sims,dim=-1)[0][:,:,:,max_r-KNN].unsqueeze(dim=-1), INF)
        sims = torch.softmax(sims,dim=-1)
        sims = sims.masked_fill(mask == 0, 0)
        sims = torch.matmul(sims,imgs) # (Bt x Bi x L x K) @ (Bi x K x D) -> (Bt x Bi x L x D)
        sims = torch.mul(sims.permute(1,0,2,3),caps).permute(1,0,2,3).sum(dim=-1) \
                    /(torch.norm(sims,p=2,dim=-1,keepdim=False)+EPS) # cos((Bt x Bi x L x D), (Bt x L x D)) -> Bt x Bi x L

        mask = get_mask(cap_lens).permute(0,2,1).repeat(1,img_lens.size(0),1)
        sims = sims.masked_fill(mask==0, MASK).permute(1,0,2)
        return sims

## due to the huge memory overhead of VSCoding, we build the iterative version
class CrossSparseAttention(nn.Module):
    def __init__(self, temperature = 0.1, KNN = 5, split=1):
        super().__init__()
        self.split = split
        self.temperature = temperature
        self.KNN = KNN

    @torch.no_grad()
    def forward_fgsims(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        KNN = max_r if self.KNN > max_r else self.KNN
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

        # calculate attention
        sims = sims / self.temperature
        sims = sims.masked_fill(mask==0, INF)
        # sparse
        sims = sims.masked_fill(sims<torch.sort(sims,dim=-1)[0][:,:,:,max_r-KNN].unsqueeze(dim=-1), INF)
        sims = torch.softmax(sims,dim=-1)
        sims = sims.masked_fill(mask == 0, 0)
        fgsims = sims.clone()
        sims = torch.matmul(sims,imgs) # (Bt x Bi x L x K) @ (Bi x K x D) -> (Bt x Bi x L x D)
        sims = torch.mul(sims.permute(1,0,2,3),caps).permute(1,0,2,3).sum(dim=-1) \
                    /(torch.norm(sims,p=2,dim=-1,keepdim=False)+EPS) # cos((Bt x Bi x L x D), (Bt x L x D)) -> Bt x Bi x L

        mask = get_mask(cap_lens).permute(0,2,1).repeat(1,img_lens.size(0),1)
        sims = sims.masked_fill(mask==0, MASK).permute(1,0,2)
        return sims, fgsims

    def sparse_attention(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        KNN = max_r if self.KNN > max_r else self.KNN
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

        # calculate attention
        sims = sims / self.temperature
        sims = sims.masked_fill(mask==0, INF)
        # sparse
        sims = sims.masked_fill(sims<torch.sort(sims,dim=-1)[0][:,:,:,max_r-KNN].unsqueeze(dim=-1), INF)
        sims = torch.softmax(sims,dim=-1)
        sims = sims.masked_fill(mask == 0, 0)
        sims = torch.matmul(sims,imgs) # (Bt x Bi x L x K) @ (Bi x K x D) -> (Bt x Bi x L x D)
        sims = torch.mul(sims.permute(1,0,2,3),caps).permute(1,0,2,3).sum(dim=-1) \
                    /(torch.norm(sims,p=2,dim=-1,keepdim=False)+EPS) # cos((Bt x Bi x L x D), (Bt x L x D)) -> Bt x Bi x L

        mask = get_mask(cap_lens).permute(0,2,1).repeat(1,img_lens.size(0),1)
        sims = sims.masked_fill(mask==0, MASK).permute(1,0,2)
        return sims

    def forward(self, imgs, caps, img_lens, cap_lens):
        n_image = imgs.size(0)
        n_caption,max_len,_ = caps.shape
        sims = torch.zeros(n_image,n_caption,max_len).to(device=caps.device)+MASK
        bt = img_lens.size(0)
        step = bt//self.split
        for i in range(self.split):
            beg = step*i
            ed = bt if i+1==self.split else step*(i+1) 
            max_w = int(cap_lens[beg:ed].max())
            sims[:,beg:ed,:max_w] = self.sparse_attention(imgs, caps[beg:ed], img_lens, cap_lens[beg:ed])
        return sims

## lightwised version of visual sparse pooling, exchange memory with time
class Subspace(nn.Module):
    def __init__(self, temperature = 0.1, KNN = 5, split=4):
        super().__init__()
        self.split = split
        self.temperature = temperature
        self.KNN = KNN
        
    def sparse_attention(self, imgs, caps, img_lens, cap_lens):
        max_r,max_w = int(img_lens.max()),int(cap_lens.max())
        KNN = max_r if self.KNN > max_r else self.KNN
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

        # calculate attention
        # sims = sims / self.temperature
        sims = sims.masked_fill(mask == 0, 0)
        sims = torch.matmul(sims,imgs) # (Bt x Bi x L x K) @ (Bi x K x D) -> (Bt x Bi x L x D)
        sims = torch.mul(sims.permute(1,0,2,3),caps).permute(1,0,2,3).sum(dim=-1) \
                    /(torch.norm(sims,p=2,dim=-1,keepdim=False)+EPS) # cos((Bt x Bi x L x D), (Bt x L x D)) -> Bt x Bi x L

        mask = get_mask(cap_lens).permute(0,2,1).repeat(1,img_lens.size(0),1)
        sims = sims.masked_fill(mask==0, MASK).permute(1,0,2)
        return sims

    def forward(self, imgs, caps, img_lens, cap_lens):
        n_image = imgs.size(0)
        n_caption,max_len,_ = caps.shape
        sims = torch.zeros(n_image,n_caption,max_len).to(device=caps.device)+MASK
        bt = cap_lens.size(0)
        step = bt//self.split
        for i in range(self.split):
            beg = step*i
            ed = bt if i+1==self.split else step*(i+1) 
            max_w = int(cap_lens[beg:ed].max())
            sims[:,beg:ed,:max_w] = self.sparse_attention(imgs, caps[beg:ed], img_lens, cap_lens[beg:ed])
        return sims
# 
class T2ICrossAttentionCoding(nn.Module):
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

# max pooling
class MaxPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        sims = sims.max(dim=-1)[0]
        return sims

# mean pooling
class MeanPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        lens = (sims!=MASK).sum(dim=-1)
        sims[sims==MASK] = 0
        sims = sims.sum(dim=-1)/lens
        return sims

# sum pooling
class SumPooling(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==MASK] = 0
        sims = sims.sum(dim=-1)
        return sims

# log-sum-exp pooling
class LSEPooling(nn.Module):
    def __init__(self,temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==MASK] = INF
        sims = torch.logsumexp(sims/self.temperature,dim=-1)
        return sims

# softmax pooling
class SoftmaxPooling(nn.Module):
    def __init__(self,temperature=0.1):
        super().__init__()
        self.temperature = temperature

    def forward(self, sims):
        assert len(sims.shape)==3
        sims[sims==MASK] = INF
        weight = torch.softmax(sims/self.temperature,dim=-1)
        sims = (weight*sims).sum(dim=-1)
        return sims

def get_coding(coding_type, **args):
    alpha = args["opt"].alpha
    KNN = args["opt"].KNN
    if coding_type=="VHACoding":
        return VHACoding()
    elif coding_type=="THACoding":
        return THACoding()
    elif coding_type=="VSACoding":
        return VSACoding(alpha)
    elif coding_type=="VSCoding":
        return VSCoding(alpha, KNN)
    elif coding_type=="TSACoding":
        return TSACoding(alpha)
    elif coding_type=="T2ICrossAttention":
        return T2ICrossAttentionCoding()
    elif coding_type=="CrossSparseAttention":
        return CrossSparseAttention(alpha, KNN)
    elif coding_type=="CC":
        return CollaborativeCoding(alpha)
    else:
        raise ValueError("Unknown coding type: {}".format(coding_type))

def get_pooling(pooling_type, **args):
    belta = args["opt"].belta
    if pooling_type=="MaxPooling":
        return MaxPooling()
    elif pooling_type=="MeanPooling":
        return MeanPooling()
    elif pooling_type=="SumPooling":
        return SumPooling()
    elif pooling_type=="SoftmaxPooling":
        return SoftmaxPooling(belta)
    elif pooling_type=="LSEPooling":
        return LSEPooling(belta)
    else:
        raise ValueError("Unknown pooling type: {}".format(pooling_type))
