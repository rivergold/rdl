from pathlib import Path
import shutil
import torch
import cv2
from rlogger import RLogger
from .event import get_current_event_storage
from ..utils.common import grid_imgs


class HookBase(object):
    def before_train(self):
        pass

    def after_train(self):
        pass

    def before_step(self):
        pass

    def after_step(self):
        pass

    def before_epoch(self):
        pass

    def after_epoch(self):
        pass


class TensorBoardWriteHook(HookBase):
    def __init__(self, tensorboard_writer, step_period=500):
        self.step_period = step_period
        self.writer = tensorboard_writer

    def after_step(self):
        if self.trainer.step % self.step_period == 0:
            self.writer.record()

    def after_epoch(self):
        self.writer.record()


class ValHook(HookBase):
    pass


# class SaveCheckpointHook(HookBase):
#     def __init__(self, base_out_dir='./log', period=1000):
#         self.period = period
#         self.out_dir = Path(base_out_dir).resolve() / 'checkpoint'
#         self.out_dir.mkdir(parents=True, exist_ok=True)

#     def after_step(self):
#         if self.trainer.step > 0 and self.trainer.step % self.period == 0:
#             checkpoint = {}
#             # Step
#             checkpoint['step'] = self.trainer.step
#             # Model
#             checkpoint['model'] = self.trainer.model.state_dict()
#             # Optimizer
#             checkpoint['optimizer'] = self.trainer.optimizer.state_dict()
#             # # Lr scheduler
#             # if hasattr(self.trainer, 'lr_scheduler'):
#             #     checkpoint[
#             #         'lr_scheduler'] = self.trainer.lr_scheduler.state_dict()

#             checkpoint_path = self.out_dir / f'step_{self.trainer.step}.pth'
#             torch.save(checkpoint, checkpoint_path.as_posix())


class LoadPretrainedCheckpointHook(HookBase):
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(
            checkpoint_path) if checkpoint_path is not None else None

    def before_train(self):
        if self.checkpoint_path is None:
            return

        device = next(self.trainer.model.parameters()).device
        checkpoint = torch.load(self.checkpoint_path.as_posix(),
                                map_location=device)
        for k, v in checkpoint.items():
            if k != 'model':
                continue
            print('load pretrained checkpoint.')
            self.trainer.model.load_state_dict(v)


class SaveCheckpointHook(HookBase):
    def __init__(self,
                 base_out_dir='./out',
                 exp_name='exp',
                 epoch_period=1,
                 enable_accelerate=False):
        self.epoch_period = epoch_period
        self.out_dir = Path(base_out_dir).resolve() / exp_name / 'checkpoint'
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.enable_accelerate = enable_accelerate

    def _pytorch_save(self):
        checkpoint = {}
        # Step
        checkpoint['epoch'] = self.trainer.epoch
        checkpoint['step'] = self.trainer.step
        # Model
        checkpoint['model'] = self.trainer.model.state_dict()
        # Optimizer
        checkpoint['optimizer'] = self.trainer.optimizer.state_dict()
        # # Lr scheduler
        # if hasattr(self.trainer, 'lr_scheduler'):
        #     checkpoint[
        #         'lr_scheduler'] = self.trainer.lr_scheduler.state_dict()

        checkpoint_path = self.out_dir / f"epoch_{self.trainer.epoch}-step_{self.trainer.step}.pth"
        torch.save(checkpoint, checkpoint_path.as_posix())

    def _accelerate_save(self):
        if self.trainer.accelerator.sync_gradients:
            if self.trainer.accelerator.is_main_process:
                checkpoint_dir = self.out_dir / f"epoch_{self.trainer.epoch}-step_{self.trainer.step}"
                self.trainer.accelerator.save_state(checkpoint_dir)

    def after_epoch(self):
        if self.trainer.epoch > 0 and self.trainer.epoch % self.epoch_period == 0:
            if self.enable_accelerate:
                self._accelerate_save()
            else:
                self._pytorch_save()

    def after_train(self):
        for dataset_type, metric_info in self.trainer.best_eval_metric_map.items(
        ):
            if metric_info is not None:
                checkpoint_from_path = self.out_dir / metric_info['checkpoint']
                # checkpoint_to_path = self.out_dir.parent / f"best_val_metric-{self.trainer.best_val_metric['metric']:.5f}-{self.trainer.best_val_metric['checkpoint']}"
                checkpoint_to_path = self.out_dir.parent / f"best_{dataset_type}_metric-{metric_info['checkpoint']}"
                if checkpoint_from_path.exists():
                    shutil.copy2(checkpoint_from_path, checkpoint_to_path)
        # Finish flag
        finish_flag_txt_path = self.out_dir.parent / 'finish_flag.txt'
        with finish_flag_txt_path.open('w') as f:
            f.write('Finish\n')


class ResumeCheckpointHook(HookBase):
    def __init__(self, checkpoint_path: str):
        self.checkpoint_path = Path(
            checkpoint_path) if checkpoint_path is not None else None

    def before_train(self):
        if self.checkpoint_path is None:
            return
        self.trainer.is_resume = True
        device = next(self.trainer.model.parameters()).device
        checkpoint = torch.load(self.checkpoint_path.as_posix(),
                                map_location=device)
        for k, v in checkpoint.items():
            # FIXME:
            if k == 'step':
                self.trainer.start_step = v + 1
                if hasattr(self.trainer, 'lr_scheduler'):
                    self.trainer.lr_scheduler.last_epoch = v
            elif k == 'model':
                self.trainer.model.load_state_dict(v)
            elif k == 'optimizer':
                self.trainer.optimizer.load_state_dict(v)
                for param_group in self.trainer.optimizer.param_groups:
                    print(param_group['lr'])


# TODO:
class MultiModelsCheckpointSaveHook(HookBase):
    pass


class LRSchedulerHook(HookBase):
    def after_epoch(self):
        if hasattr(self.trainer, 'lr_scheduler'):
            # MEMO: LRScheduleHook need be in front of TensorBoardWriteHook
            # lr = self.trainer.lr_scheduler.get_lr()[0]
            lr = self.trainer.optimizer.param_groups[-1]['lr']
            self.trainer.update_event_storage({'lr': ('scalar', lr)})
            # Update lr
            if isinstance(self.trainer.lr_scheduler,
                          torch.optim.lr_scheduler._LRScheduler):
                self.trainer.lr_scheduler.step()
            else:
                self.trainer.lr_scheduler.step(epoch=self.trainer.epoch)


# TODO:
class LogHook(HookBase):
    def __init__(self,
                 base_out_dir='./out',
                 exp_name='exp',
                 max_bytes=1e8,
                 backup_count=1,
                 step_period=10):
        self.step_period = step_period
        log_path = Path(base_out_dir).resolve() / exp_name / 'info.log'
        log_path.parent.mkdir(parents=True, exist_ok=True)
        RLogger.init(log_path.as_posix(),
                     enable_screen=True,
                     max_bytes=max_bytes,
                     backup_count=backup_count)

    def before_train(self):
        # Log cfg
        if hasattr(self.trainer, 'cfg'):
            print('print cfg')
            RLogger.log(f'cfg: {self.trainer.cfg}')

    def after_step(self):
        if self.step_period > 0 and self.trainer.step % self.step_period == 0:
            current_event_storage = get_current_event_storage()
            log_info = f'epoch: {self.trainer.epoch}, step: {self.trainer.step}'
            for k, v in current_event_storage.scalar_dict.items():
                log_info = '\n'.join([log_info, f'{k}: {v}'])
            for k, v in current_event_storage.info_dict.items():
                log_info = '\n'.join([log_info, f'{k}: {v}'])
            RLogger.log(log_info)

    def after_epoch(self):
        current_event_storage = get_current_event_storage()
        log_info = f'epoch: {self.trainer.epoch}'
        for k, v in current_event_storage.scalar_dict.items():
            log_info = '\n'.join([log_info, f'{k}: {v}'])
        for k, v in current_event_storage.info_dict.items():
            log_info = '\n'.join([log_info, f'{k}: {v}'])
        RLogger.log(log_info)


class VisualizationHook(HookBase):
    """[summary]
    Image: color_mode=BGR

    Args:
        base_out_dir (str, optional): [description]. Defaults to './log'.
        period (int, optional): [description]. Defaults to 500.
    """
    def __init__(self,
                 base_out_dir='./out',
                 exp_name='exp',
                 step_period=500,
                 color_mode='bgr'):
        out_dir = Path(base_out_dir).resolve() / exp_name / 'visualization'
        out_dir.mkdir(parents=True, exist_ok=True)
        self.out_dir = out_dir
        self.step_period = step_period
        self.color_mode = color_mode

    def after_step(self):
        if self.trainer.step > 0 and self.trainer.step % self.step_period == 0:
            current_event_storage = get_current_event_storage()
            for name, imgs in current_event_storage.imgs_dict.items():
                # imgs shape [N, H, W, C], color_mode BGR
                img_path = self.out_dir / f"epoch_{self.trainer.epoch:04}" / f"epoch_{self.trainer.epoch:04}-step_{self.trainer.step}-{name.replace('/', '-')}.jpg"
                img_path.parent.mkdir(parents=True, exist_ok=True)
                img = grid_imgs(imgs)
                if self.color_mode == 'rgb':
                    img = img[:, :, ::-1]
                cv2.imwrite(img_path.as_posix(), img)

    def after_epoch(self):
        current_event_storage = get_current_event_storage()
        for name, imgs in current_event_storage.imgs_dict.items():
            # imgs shape [N, H, W, C], color_mode BGR
            img_path = self.out_dir / f"epoch_{self.trainer.epoch:04}" / f"epoch_{self.trainer.epoch:04}-step_{self.trainer.step}-{name.replace('/', '-')}.jpg"
            img_path.parent.mkdir(parents=True, exist_ok=True)
            img = grid_imgs(imgs)
            if self.color_mode == 'rgb':
                img = img[:, :, ::-1]
            cv2.imwrite(img_path.as_posix(), img)
