import torch
from timm.scheduler.cosine_lr import CosineLRScheduler


def build_lr_scheduler(name, optimizer, max_epoch, **kwargs):
    if name == 'MultiStepLR':
        """
        E.g.
        "lr_scheduler": {
            "name": "MultiStepLR",
            "params": {
                "milestones": [
                    120,
                    200,
                    260
                ],
                "gamma": 0.1
            }
        }
        """
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
            optimizer, **kwargs)
    elif name == 'timm_CosineLRScheduler':
        """
        E.g.
        "lr_scheduler": {
            "name": "timm_CosineLRScheduler",
            "params": {
                "t_initial": "auto",
                "lr_min": 1e-8,
                "cycle_decay": 0.5,
                "warmup_t": 5,
                "warmup_lr_init": 1e-7,
                "warmup_prefix": false,
                "cycle_limit": 8
            }
        }
        """
        if kwargs['t_initial'] == 'auto':
            kwargs['t_initial'] = max_epoch / kwargs['cycle_limit']
        lr_scheduler = CosineLRScheduler(optimizer=optimizer, **kwargs)
    else:
        raise NotImplementedError
    return lr_scheduler
