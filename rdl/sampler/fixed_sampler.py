import random
import itertools
import torch


class FixedSampler(torch.utils.data.Sampler):
    def __init__(self, num_samples, num_fixed_samples, seed=0):
        self.num_samples = num_samples
        self.num_fixed_samples = num_fixed_samples
        random.seed(seed)
        self.fixed_idxes = random.sample(range(self.num_samples),
                                         self.num_fixed_samples)

    def __iter__(self):
        for idx in self.fixed_idxes:
            yield idx

    def __len__(self):
        return self.num_fixed_samples