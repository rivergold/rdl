from typing import OrderedDict, List, Union
import weakref, time
from pathlib import Path
from tqdm import tqdm
import torch
from omegaconf import DictConfig, OmegaConf
from ..event import EventStorage
from ..hook import HookBase
from .base_trainer import TrainerBase


class MultiModelTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.export_cfg()
        self.model = {}

    def export_cfg(self):
        # Export cfg
        cfg_out_dir = Path(self.cfg.work_dir) / self.cfg.exp_name
        cfg_out_dir.mkdir(parents=True, exist_ok=True)
        if isinstance(self.cfg, dict):
            cfg_out_path = cfg_out_dir / f"{self.cfg.exp_name}.json"
            self.cfg.to_json(cfg_out_path)
        elif isinstance(self.cfg, DictConfig):
            cfg_out_path = cfg_out_dir / f"{self.cfg.exp_name}.yaml"
            with cfg_out_path.open('w') as f:
                OmegaConf.save(self.cfg, f)
        else:
            raise TypeError()

    def build_hooks(self):
        raise NotImplementedError

    def set_model(self,
                  model_name,
                  model,
                  cuda_id: Union[int, None] = 0,
                  enable_accelerate_prepare=False):
        self.model[model_name] = model
        if torch.cuda.is_available() and cuda_id is not None:
            if not self.cfg.accelerate.enable:
                self.model[model_name].to(device=f"cuda:{cuda_id}")
            else:
                if enable_accelerate_prepare:
                    self.model[model_name] = self.accelerator.prepare(
                        self.model[model_name])
                else:
                    self.model[model_name].to(self.accelerator.device)

    def set_train_dataloder(self, train_dataloder):
        self.train_dataloader = train_dataloder
        if self.cfg.accelerate.enable:
            self.train_dataloader = self.accelerator.prepare(train_dataloder)
        self.num_epoch_step = len(self.train_dataloader)
        # self.train_dataloader_iter = iter(train_dataloder)

    def set_val_dataloader(self, val_dataloader):
        self.val_dataloader = val_dataloader

    def set_test_dataloader(self, test_dataloader):
        self.test_dataloader = test_dataloader

    def set_test_dataloaders(self, test_dataloaders: List):
        self.test_dataloader_map = OrderedDict()
        for idx, test_dataloader in enumerate(test_dataloaders):
            self.test_dataloader_map[f"test_{idx}"] = test_dataloader

    def set_criterion(self, criterion: torch.nn.Module):
        self.criterion = criterion
        criterion.trainer = weakref.proxy(self)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        if self.cfg.accelerate.enable:
            self.optimizer = self.accelerator.prepare(optimizer)

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler
        if self.cfg.accelerate.enable:
            self.lr_scheduler = self.accelerator.prepare(lr_scheduler)

    def set_accelerator(self, accelerator):
        self.accelerator = accelerator

    def run_step(self, batch_sample):
        raise NotImplementedError
