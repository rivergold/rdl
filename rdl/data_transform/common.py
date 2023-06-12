from typing import List
import numpy as np
import torch
from ..utils.common import img_normalize


class ImgNormalize:
    def __init__(self,
                 mean: List[float] = [0., 0., 0.],
                 std: List[float] = [1., 1., 1.],
                 gray_mean: List[float] = [0.],
                 gray_std: List[float] = [1.]):
        self.mean = mean
        self.std = std
        self.gray_mean = gray_mean
        self.gray_std = gray_std

    # @profile
    # @cal_run_time
    def normalize(self, sample):
        for k, v in sample.items():
            if k == 'rgb_img':
                v = img_normalize(v, mean=self.mean, std=self.std)
            elif k == 'gray_img':
                v = img_normalize(v, mean=self.gray_mean, std=self.gray_std)
            sample[k] = v
        return sample

    # @profile
    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.normalize, sample))
        return self.normalize(sample)


class ToTensor:
    def to_tensor(self, sample):
        for k, v in sample.items():
            if not isinstance(v, np.ndarray) and not isinstance(
                    v, torch.Tensor):
                continue

            if isinstance(v, np.ndarray):
                if 'class_id' not in k:
                    v = torch.from_numpy(v).to(torch.float32)
                else:
                    v = torch.from_numpy(v).to(torch.long)
            if k == 'rgb_img':
                v = v.permute(2, 0, 1)
            elif k == 'gray_img':
                v = v.permute(2, 0, 1)
            sample[k] = v
        return sample

    # @profile
    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.to_tensor, sample))
        return self.to_tensor(sample)