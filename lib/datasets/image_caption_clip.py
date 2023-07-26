"""COCO dataset loader"""
import os
import os.path as osp
import random
import json
from PIL import Image

from pkg_resources import packaging

import torch
import torch.utils.data as data

from clip.simple_tokenizer import SimpleTokenizer as _Tokenizer
_tokenizer = _Tokenizer()

import logging
logger = logging.getLogger(__name__)

class CLIPDataset(data.Dataset):
    """
    Load captions and image for COCO or Flickr
    """
    def __init__(self, data_path, data_name, data_split, tokenizer, preprocess, opt, train):
        self.tokenizer = tokenizer
        self.preprocess = preprocess
        self.opt = opt
        self.train = train
        self.data_path = data_path
        self.data_name = data_name

        loc_cap = osp.join(data_path, 'precomp')
        loc_image = osp.join(data_path, 'precomp')
        loc_mapping = osp.join(data_path, 'id_mapping.json')
        if 'coco' in data_name:
            self.image_base = osp.join(data_path, 'images')
        else:
            self.image_base = osp.join(data_path, 'flickr30k-images')

        with open(loc_mapping, 'r') as f_mapping:
            self.id_to_path = json.load(f_mapping)

        # Read Captions
        self.captions = []
        with open(osp.join(loc_cap, '%s_caps.txt' % data_split), 'r') as f:
            for line in f:
                self.captions.append(line.strip())

        # Get the image ids
        with open(osp.join(loc_image, '{}_ids.txt'.format(data_split)), 'r') as f:
            image_ids = f.readlines()
            self.images = [int(x.strip()) for x in image_ids]

        self.length = len(self.captions)
        
        # rkiros data has redundancy in images, we divide by 5, 10crop doesn't
        num_images = len(self.images)

        if num_images != self.length:
            self.im_div = 5
        else:
            self.im_div = 1
        # the development set for coco is large and so validation would be slow
        if data_split == 'dev':
            self.length = 5000

    def __getitem__(self, index):
        # handle the image redundancy
        img_index = index // self.im_div

        caption = self.captions[index]
        ## TODO:优化 process caption部分的代码         
        target = list()
        target.append(self.tokenizer.encoder["<|startoftext|>"])
        target.extend(self.tokenizer.encode(caption))
        target.append(self.tokenizer.encoder["<|endoftext|>"])

        image_id = self.images[img_index]
        image_path = os.path.join(self.image_base, self.id_to_path[str(image_id)])
        im_in = Image.open(image_path).convert("RGB")
        image = self.preprocess(im_in)

        return image, target, index, img_index

    def __len__(self):
        return self.length


def collate_fn(data):
    """Build mini-batch tensors from a list of (image, caption) tuples.
    Args:
        data: list of (image, caption) tuple.
            - image: torch tensor of shape (3, 256, 256).
            - caption: torch tensor of shape (?); variable length.

    Returns:
        images: torch tensor of shape (batch_size, 3, 256, 256).
        targets: torch tensor of shape (batch_size, padded_length).
        lengths: list; valid length for each padded caption.
    """
    images, captions, ids, img_ids = zip(*data)
    img_lengths = [len(image) for image in images]

    # Merge images (convert tuple of 3D tensor to 4D tensor)
    images = torch.stack(images, 0)

    # Merget captions (convert tuple of 1D tensor to 2D tensor)
    lengths = [len(cap) for cap in captions]

    # copy from the source code of CLIP
    context_length = 77
    truncate = True
    
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    if packaging.version.parse(torch.__version__) < packaging.version.parse("1.8.0"):
        result = torch.zeros(len(captions), context_length, dtype=torch.long)
    else:
        result = torch.zeros(len(captions), context_length, dtype=torch.int)
    for i, tokens in enumerate(captions):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(f"Input {captions[i]} is too long for context length {context_length}")
        result[i, :len(tokens)] = torch.tensor(tokens)

    lengths = [min(item, context_length) for item in lengths]
    return images, img_lengths, result, lengths, ids


def get_loader(data_path, data_name, data_split, tokenizer, preprocess, opt, batch_size=100,
               shuffle=True, num_workers=2, train=True):
    """Returns torch.utils.data.DataLoader for custom coco dataset."""
    if train:
        drop_last = True
    else:
        drop_last = False
    if opt.precomp_enc_type in ["clip"]:
        dset = CLIPDataset(data_path, data_name, data_split, tokenizer, preprocess, opt, train)
        data_loader = torch.utils.data.DataLoader(dataset=dset,
                                                  batch_size=batch_size,
                                                  shuffle=shuffle,
                                                  pin_memory=True,
                                                  collate_fn=collate_fn,
                                                  num_workers=num_workers,
                                                  drop_last=drop_last)
    else:
        raise ValueError("Unknown precomp_enc_type: {}".format(opt.precomp_enc_type))
    return data_loader


def get_loaders(data_path, data_name, tokenizer, preprocess, batch_size, workers, opt):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, preprocess, opt,
                              batch_size, True, workers, train=opt.mask)
    val_loader = get_loader(data_path, data_name, 'dev', tokenizer, preprocess, opt,
                            batch_size, False, workers, train=False)
    return train_loader, val_loader


def get_train_loader(data_path, data_name, tokenizer, batch_size, workers, opt, shuffle):
    train_loader = get_loader(data_path, data_name, 'train', tokenizer, opt,
                              batch_size, shuffle, workers)
    return train_loader


def get_test_loader(split_name, data_name, tokenizer, preprocess, batch_size, workers, opt):
    test_loader = get_loader(opt.data_path, data_name, split_name, tokenizer, preprocess, opt,
                             batch_size, False, workers, train=False)
    return test_loader
