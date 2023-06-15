from typing import OrderedDict, List
import weakref, time
from pathlib import Path
from tqdm import tqdm
from ..event import EventStorage
from ..hook import HookBase


class TrainerBase(object):
    def __init__(self):
        self._hooks = []
        self.start_step = -1
        self.is_from_resuming = False
        self.num_epoch_step = 0
        self.epoch = None
        self.step = None
        self.is_last_batch = False
        self.is_resume = False
        self.best_eval_metric_map = {'val': None, 'test': None}

    def register_hooks(self, hooks):
        for hook in hooks:
            assert isinstance(hook, HookBase)
            hook.trainer = weakref.proxy(self)
            self._hooks.append(hook)

    # @profile
    def train(self,
              start_step: int = 0,
              max_epoch: int = 0,
              enable_val=False,
              enable_test=False):
        self.step = 0
        self.max_epoch = max_epoch
        self.enable_val = enable_val
        self.enable_test = enable_test
        with EventStorage(start_step) as self.event_storge:
            self.before_train()

            # Resume checking
            # ResumeCheckpointHook will change trainer.start_step
            if self.start_step != -1 and start_step is not None:
                raise ValueError(
                    'Resume from checkpont, you need to set start_step correctly'
                )
            # Begin with step=0
            elif self.start_step == -1 and start_step is None:
                raise ValueError(
                    'No resume from checkpoint, but start_step is None')

            if start_step is not None:
                self.start_step = start_step

            # # Loop
            # for self.step in range(self.start_step, self.max_step):
            #     # TODO: before epoch
            #     start_time = time.time()
            #     self.update_event_storage({'step': ('step', self.step)})
            #     self.before_step()
            #     self.run_step()
            #     self.update_event_storage(
            #         {'train_step_time': ('scalar', time.time() - start_time)})
            #     self.after_step()
            #     # Clear event_storge
            #     self.event_storge.clear()

            # Loop
            for self.epoch in tqdm(range(self.max_epoch), desc='train-epoch'):
                self.before_epoch()
                # for step in tqdm(range(self.num_epoch_step),
                #                  desc='train-step'):
                for step, batch_sample in enumerate(tqdm(
                        self.train_dataloader)):
                    if step == self.num_epoch_step - 1:
                        self.is_last_batch = True
                    else:
                        self.is_last_batch = False
                    self.event_storge.clear()
                    start_time = time.time()
                    self.update_event_storage({'step': ('step', self.step)})
                    self.before_step()
                    self.run_step(batch_sample)
                    self.update_event_storage({
                        'train_step_time': ('scalar', time.time() - start_time)
                    })
                    self.after_step()
                    # Clear event_storge
                    self.step += 1
                if enable_val:
                    self.run_val()
                if enable_test:
                    self.run_test()
                self.after_epoch()
            self.after_train()

    def before_train(self):
        for hook in self._hooks:
            hook.before_train()

    def after_train(self):
        for hook in self._hooks:
            hook.after_train()

    def before_step(self):
        for hook in self._hooks:
            hook.before_step()

    def after_step(self):
        for hook in self._hooks:
            hook.after_step()

    def before_epoch(self):
        for hook in self._hooks:
            hook.before_epoch()

    def after_epoch(self):
        for hook in self._hooks:
            hook.after_epoch()

    def run_step(self, batch_sample):
        raise NotImplementedError

    def run_val(self):
        raise NotImplementedError

    def run_test(self):
        raise NotImplementedError

    def update_event_storage(self, event_data: dict):
        for name, data in event_data.items():
            data_type = data[0]
            data = data[1]
            if data_type == 'step':
                self.event_storge.set_step(data)
            elif data_type == 'text':
                self.event_storge.set_text(name, data)
            elif data_type == 'scalar':
                self.event_storge.set_scalar(name, data)
            elif data_type == 'images':
                self.event_storge.set_images(name, data)
            elif data_type == 'weight':
                self.event_storge.set_weight(name, data)
            elif data_type == 'info':
                self.event_storge.set_info(name, data)
            else:
                raise NotImplementedError()
