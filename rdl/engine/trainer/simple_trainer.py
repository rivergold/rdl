from typing import OrderedDict, List
import weakref, time
from pathlib import Path
from tqdm import tqdm
from omegaconf import DictConfig, OmegaConf
from ..event import EventStorage
from ..hook import HookBase
from .base_trainer import TrainerBase


class SimpleTrainer(TrainerBase):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.export_cfg()

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

    def set_model(self, model):
        self.model = model
        if self.cfg.gpu.enable:
            self.model.cuda()
            # TODO: distributed gpu

    def set_train_dataloder(self, train_dataloder):
        self.train_dataloader = train_dataloder
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

    # def set_criterion(self, criterion):
    #     criterion.trainer = weakref.proxy(self)
    #     self.criterion = criterion

    def set_criterion(self, criterion, criterion2=None):
        self.criterion = criterion
        self.criterion2 = criterion2
        criterion.trainer = weakref.proxy(self)
        if not criterion2 is None:
            criterion2.trainer = weakref.proxy(self)

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer

    def set_lr_scheduler(self, lr_scheduler):
        self.lr_scheduler = lr_scheduler

    def run_step(self, batch_sample):
        raise NotImplementedError
