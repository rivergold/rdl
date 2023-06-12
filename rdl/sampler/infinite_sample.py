import itertools
import torch


class InfiniteSampler(torch.utils.data.Sampler):
    def __init__(self, num_samples, shuffle=False, seed=0):
        self.num_samples = num_samples
        self.shuffle = shuffle
        self.seed = seed

        self.generator = torch.Generator()
        self.generator.manual_seed(self.seed)

    def _gen_infinite_idx(self):
        while True:
            yield from torch.randperm(self.num_samples,
                                      generator=self.generator)

    def __iter__(self):
        if self.shuffle:
            yield from self._gen_infinite_idx()
        else:
            yield itertools.cycle(range(self.num_samples))

    def __len__(self):
        return self.num_samples
