from pathlib import Path
import random, json
import cv2
from PIL import Image

import torch
import datasets
import numpy as np
from transformers import AutoTokenizer
from rdl.data_transform.collate_fn import common_collate_fn
from data_transform import ImageTransform, build_train_transform
from omegaconf import OmegaConf


class StyleDataseet(torch.utils.data.Dataset):
    def __init__(self,
                 mode,
                 root_dir,
                 sample_list_file_path,
                 tokenizer=None,
                 transform=None):
        self.mode = mode
        self.root_dir = Path(root_dir)
        self.sample_list_file_path = Path(sample_list_file_path)
        self.tokenizer = tokenizer
        self.transform = transform

        self.sample_names = None
        self._load_sample_list()

    def _load_sample_list(self):
        with self.sample_list_file_path.open() as f:
            self.sample_names = f.read().splitlines()
        random.shuffle(self.sample_names)

    def _parse_anno(self, sample, anno):
        text = anno['text']
        sample['text'] = text
        token_ids = self.tokenizer(text,
                                   max_length=self.tokenizer.model_max_length,
                                   padding="max_length",
                                   truncation=True,
                                   return_tensors="pt")['input_ids']
        sample['token_ids'] = token_ids

    def __getitem__(self, idx):
        sample = {
            'keys': ['rgb_img', 'text', 'token_ids'],
            'rgb_img': None,
            'text': None,
            'token_ids': None
        }
        sample_name = self.sample_names[idx]
        subfolder_name, img_name, _ = sample_name.split(',')
        sample_name = img_name.split('.')[0]
        img_path = self.root_dir / subfolder_name / 'imgs' / img_name
        json_path = self.root_dir / subfolder_name / 'jsons' / f"{sample_name}.json"
        # img = cv2.imread(img_path.as_posix())
        # sample['rgb_img'] = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        sample['rgb_img'] = Image.open(img_path.as_posix()).convert('RGB')
        with json_path.open() as f:
            anno = json.load(f)
        self._parse_anno(sample, anno)
        if self.transform is not None:
            sample = self.transform(sample)
        return sample

    def __len__(self):
        return len(self.sample_names)
        # return 10


def build_train_dataloader(root_dir, sample_list_file_path, tokenizer):
    # transform = build_train_transform(cfg)  # TODO: update
    transform = ImageTransform()
    dataset = StyleDataseet(
        mode='train',
        root_dir=root_dir,
        sample_list_file_path=sample_list_file_path,
        tokenizer=tokenizer,
        transform=transform,
    )
    dataloader = torch.utils.data.DataLoader(dataset,
                                             shuffle=True,
                                             collate_fn=common_collate_fn,
                                             batch_size=4,
                                             num_workers=1)
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