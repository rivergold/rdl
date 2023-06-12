from typing import Optional, Union
from pathlib import Path
from functools import partial
import numpy as np
import pickle
import cv2
import torch
import torchvision

import rdl
from rdl.data_transform.collate_fn import common_collate_fn
from rdl.sampler import InfiniteSampler, FixedSampler
from rdl.data_transform.common import ImgNormalize, ToTensor


class Cifar10Dataset(torch.utils.data.Dataset):
    def __init__(self,
                 mode=None,
                 data_dir: Union[str, Path] = None,
                 data_transform=None):
        self.data_dir = Path(data_dir)
        self.raw_data = []
        self.raw_labels = []
        self.mode = mode
        self.data_transform = data_transform
        self.keys = ['rgb_img', 'class_id']

        # Read data
        if mode == 'train':
            for idx in range(1, 6):
                file_path = self.data_dir / f"data_batch_{idx}"
                with file_path.open('rb') as f:
                    cur_raw_data = pickle.load(f, encoding='bytes')
                self.raw_data.append(cur_raw_data[b'data'])
                self.raw_labels.extend(cur_raw_data[b'labels'])
            self.raw_data = np.concatenate(self.raw_data, axis=0)
        elif mode == 'val':
            raise ValueError('No val data.')
        elif mode == 'test':
            file_path = self.data_dir / f"test_batch"
            with file_path.open('rb') as f:
                cur_raw_data = pickle.load(f, encoding='bytes')
            self.raw_data = cur_raw_data[b'data']
            self.raw_labels = cur_raw_data[b'labels']

    def _parse_rgb_img(self, data):
        img = data.reshape(3, 32, 32)
        img = np.transpose(img, (1, 2, 0))
        return img

    def __getitem__(self, idx):
        sample = {'rgb_img': None, 'class_id': None}
        img_data = self.raw_data[idx]
        sample['rgb_img'] = self._parse_rgb_img(img_data)
        sample['class_id'] = self.raw_labels[idx]
        if self.data_transform is not None:
            sample = self.data_transform(sample)
        sample['keys'] = self.keys
        return sample

    def __len__(self):
        return len(self.raw_data)


def build_train_dataloader(data_dir: Union[str, Path],
                           batch_size=1,
                           num_workers=0,
                           data_transform=None,
                           enable_ddp=False,
                           **kwargs):
    # Build dataset
    dataset = Cifar10Dataset(mode='train',
                             data_dir=data_dir,
                             data_transform=data_transform)
    # Build sampler
    sampler = InfiniteSampler(len(dataset), shuffle=True)
    # Build dataloader
    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             drop_last=True,
                                             collate_fn=partial(
                                                 common_collate_fn, **kwargs),
                                             sampler=sampler,
                                             num_workers=num_workers)
    return dataloader


def build_val_dataloader():
    pass


def build_test_dataloader():
    pass


# if __name__ == '__main__':
#     file_path = Path('/home/hejing/data/opensource/data/cifar-10-batches-py/data_batch_1')
#     with file_path.open('rb') as f:
#         raw_data = pickle.load(f, encoding='bytes')
#     print(len(raw_data))
#     print(type(raw_data))
#     print(raw_data.keys())
#     print(len(raw_data[b'data']))
#     print(type(raw_data[b'data']))
#     print(raw_data[b'data'].shape)
#     print(type(raw_data[b'labels']))

# if __name__ == '__main__':
#     train_dataset = Cifar10Dataset(
#         mode='train',
#         data_dir='/home/hejing/data/opensource/data/cifar-10-batches-py')
#     print(len(train_dataset))

#     sample = train_dataset[0]
#     cv2.imwrite('./debug.jpg', sample['rgb_img'])
#     print(sample['class_id'])

if __name__ == '__main__':
    data_dir = '/home/hejing/data/opensource/data/cifar-10-batches-py'
    train_data_transform = torchvision.transforms.Compose([
        ImgNormalize(),
        ToTensor(),
    ])
    train_dataloader = build_train_dataloader(
        data_dir,
        batch_size=4,
        num_workers=0,
        data_transform=train_data_transform,
        enable_ddp=False)
    for idx, batch_sample in enumerate(train_dataloader):
        print(batch_sample['keys'])
        print(batch_sample['rgb_imgs'].shape)
        print(batch_sample['class_ids'])