from pathlib import Path
import shutil
import numpy as np
import cv2
import torch.utils.tensorboard as tensorboard
from ..utils.common import bgr2rgb, grid_imgs

_OBSERVE_STORAGR_STACK = []


def get_current_event_storage():
    if len(_OBSERVE_STORAGR_STACK) > 0:
        return _OBSERVE_STORAGR_STACK[-1]
    else:
        raise RuntimeError(
            'You need run your code with `with EventStorage() as event_storage`'
        )


class EventStorage(object):
    def __init__(self, start_step):
        self.start_step = start_step
        self.step = None
        self.text_dict = {}
        self.scalar_dict = {}
        self.imgs_dict = {}
        self.weight_dict = {}
        self.info_dict = {}

    def set_step(self, step):
        self.step = step

    def set_text(self, name, text):
        self.text_dict[name] = text

    def set_scalar(self, name, value):
        self.scalar_dict[name] = value

    def set_images(self, name, imgs):
        """Set images

        Arguments:
            name {str} -- [description]
            imgs {np.ndarray} -- shape (N, H, W, C), bgr images
        """
        self.imgs_dict[name] = imgs

    def set_weight(self, name, weight):
        self.weight_dict[name] = weight

    def set_info(self, name, info):
        self.info_dict[name] = info

    def clear(self):
        self.text_dict.clear()
        self.scalar_dict.clear()
        self.imgs_dict.clear()
        self.weight_dict.clear()
        self.info_dict.clear()

    def __enter__(self):
        _OBSERVE_STORAGR_STACK.append(self)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        _OBSERVE_STORAGR_STACK.pop()


class EventWriter(object):
    """Observer record infomation during run
    """
    def record(self):
        pass

    def close(self):
        pass


class TensorboardWriter(EventWriter):
    """Record info into tensorboard
    """
    def __init__(self,
                 exp_name: str = 'debug',
                 base_out_dir: str = './log',
                 enable_draw_img=False):
        base_out_dir = Path(base_out_dir).resolve()
        log_dir = base_out_dir / exp_name / 'tensorboard'
        log_dir.mkdir(parents=True, exist_ok=True)
        self.enable_draw_img = enable_draw_img
        self._writer = tensorboard.SummaryWriter(log_dir=log_dir.as_posix())

    def record(self):
        current_event_storage = get_current_event_storage()

        for name, text in current_event_storage.text_dict.items():
            self._writer.add_text(name,
                                  text,
                                  global_step=current_event_storage.step)

        for name, value in current_event_storage.scalar_dict.items():
            self._writer.add_scalar(name,
                                    value,
                                    global_step=current_event_storage.step)

        if self.enable_draw_img:
            for name, imgs in current_event_storage.imgs_dict.items():
                # imgs shape [N, H, W, C]
                # imgs = bgr2rgb(imgs)
                # FIXME: PyTorch<=1.4.0 tensorboard add_images has bug, maybe be solved in 1.5.0
                img = grid_imgs(imgs)
                # cv2.imwrite('{}.jpg'.format(name), img)
                # self._writer.add_images(name,
                #                         imgs,
                #                         global_step=current_event_storage.step,
                #                         dataformats='NHWC')

                self._writer.add_image(name,
                                       img,
                                       global_step=current_event_storage.step,
                                       dataformats='HWC')

        for name, weight in current_event_storage.weight_dict.items():
            self._writer.add_histogram(name,
                                       weight,
                                       global_step=current_event_storage.step)

    def close(self):
        pass
