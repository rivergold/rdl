import random
import torch


class MultiDatasetSampler(torch.utils.data.Sampler):
    def __init__(self, datasets, shuffle=False, seed=0, enable_infinite=True):
        self.dataset_idxes = [idx for idx in range(len(datasets))]
        self.dataset_len = [len(x) for x in datasets]
        self.num_samples = sum(self.dataset_len)
        self.shuffle = shuffle
        self.seed = seed
        self.enable_infinite = enable_infinite

        self.multidataset_idxes = []
        for dataset_idx in self.dataset_idxes:
            for sample_idx in range(self.dataset_len[dataset_idx]):
                self.multidataset_idxes.append((dataset_idx, sample_idx))

    def _gen_infiniete_idx(self):
        while True:
            yield from random.sample(self.multidataset_idxes, self.num_samples)

    def __iter__(self):
        if self.enable_infinite:
            yield from self._gen_infiniete_idx()
        else:
            yield from random.sample(self.multidataset_idxes, self.num_samples)

    def __len__(self):
        return self.num_samples


class MultiDatasetFixedSampler(torch.utils.data.Sampler):
    def __init__(self, datasets, num_fixed_samples=None, seed=0):
        self.dataset_idxes = [idx for idx in range(len(datasets))]
        self.dataset_len = [len(x) for x in datasets]
        self.num_samples = sum(self.dataset_len)
        self.seed = seed

        self.multidataset_idxes = []
        for dataset_idx in self.dataset_idxes:
            for sample_idx in range(self.dataset_len[dataset_idx]):
                self.multidataset_idxes.append((dataset_idx, sample_idx))
        random.shuffle(self.multidataset_idxes)
        if num_fixed_samples is not None:
            self.multidataset_idxes = self.multidataset_idxes[:
                                                              num_fixed_samples]

    def __iter__(self):
        for idx in self.multidataset_idxes:
            yield idx

    def __len__(self):
        return self.num_samples