import torch
import datasets
import numpy as np
from transformers import AutoTokenizer
from rdl.data_transform.collate_fn import common_collate_fn
from data_transform import build_train_transform
from omegaconf import OmegaConf


class FusingFill50kDataset(torch.utils.data.Dataset):
    def __init__(self, mode, cache_dir=None, tokenizer=None, transform=None):
        self.mode = mode
        self.data = datasets.load_dataset(
            'fusing/fill50k',
            cache_dir='/home/hejing/data/opensource/huggingface/data')['train']
        self.tokenizer = tokenizer
        self.transform = transform

    def __getitem__(self, idx):
        sample = {
            'keys': ['rgb_img', 'condition_rgb_img', 'text', 'token_ids'],
            'rgb_img': None,
            'condition_rgb_img': None,
            'text': None,
        }
        data = self.data[idx]
        sample['rgb_img'] = data['image']
        sample['condition_rgb_img'] = data['conditioning_image']
        sample['text'] = data['text']
        sample['token_ids'] = self.tokenizer(
            data['text'],
            max_length=self.tokenizer.model_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt")['input_ids']
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        # return len(self.data)
        return 10


def build_train_dataloader(cfg, tokenizer):
    transform = build_train_transform(cfg)
    dataset = FusingFill50kDataset('train',
                                   cache_dir=cfg.dataset.train.cache_dir,
                                   tokenizer=tokenizer,
                                   transform=transform)
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=True,
                                             collate_fn=common_collate_fn,
                                             **cfg.dataset.train.kwargs)
    return dataloader


if __name__ == '__main__':
    data = datasets.load_dataset(
        'fusing/fill50k',
        cache_dir='/home/hejing/data/opensource/huggingface/data')
    print(type(data))
    print(data.keys())
    print(type(data['train']))
    print(data['train'][0])
    print(len(data['train']))
    print(type(data['train'][0]['image']))

    tokenizer = AutoTokenizer.from_pretrained(
        'runwayml/stable-diffusion-v1-5',
        subfolder='tokenizer',
        use_fast=False,
        cache_dir='/home/hejing/data/opensource/huggingface/hub')
    tokens = tokenizer('a dog',
                       max_length=tokenizer.model_max_length,
                       padding="max_length",
                       truncation=True,
                       return_tensors="pt")
    print(tokens)

    cfg = OmegaConf.load('./config/base.yaml')
    dataloader = build_train_dataloader(cfg, tokenizer)
    for idx, batch_sample in enumerate(dataloader):
        print(batch_sample['batch_token_ids'].shape)
        break