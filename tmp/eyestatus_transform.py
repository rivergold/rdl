from typing import List, Dict
from pathlib import Path
import random
import torch
import cv2
import numpy as np
# from imgaug import augmenters as iaa
from ....utils.common import get_resize_matrix, get_rotate_matrix, get_axis_rotate_matrix, warp_boxes, normalize
from ....utils.aug import *

__all__ = [
    'EYE_STATUS_ENCODE_MAP',
    'EYE_STATUS_DECODE_MAP',
    'collate_fn',
    'RandomRotate',
    'RandomAxisRotate',
    'BboxAroundCropV2',
    'Resize',
    'Flip',
    'ColorJitter',
    # 'ImgaugColorChange',
    'AddBrightSplot',
    'Gray',
    'RandomErase',
    'RandomAddExtraImg',
    'ChangeGrayBrightness',
    'ChangeGrayContrast',
    'AddNoise',
    'AddGaussianBlur',
    'EqualizeHist',
    'Normalize',
    'EncodeToOnehot',
    'ToTensor',
    'Mixup'
]

EYE_STATUS_ENCODE_MAP = {
    'open': 0,
    'close': 1,
    'invisible': 2,
    # 'halfopen': 3,
}

# EYE_STATUS_DECODE_MAP = {
#     v: k
#     for k, v in EYE_STATUS_ENCODE_MAP.items() if k not in ['halfopen']
# }
EYE_STATUS_DECODE_MAP = {v: k for k, v in EYE_STATUS_ENCODE_MAP.items()}


# @profile
def collate_fn(in_batch, **kwargs):
    out_batch = {}

    if isinstance(in_batch[0], list):
        tmp_batch = []
        for each_batch in in_batch:
            tmp_batch.extend(each_batch)
        in_batch = tmp_batch

    # Init out batch
    for k in in_batch[0].keys():
        out_batch[f'{k}s'] = []
    # Feed data
    for _, sample in enumerate(in_batch):
        for k, v in sample.items():
            if v is not None:
                out_batch[f'{k}s'].append(v)
    # Stack data
    ks = ['rgb_imgs', 'gray_imgs', 'gts']
    for k in ks:
        if len(out_batch[k]):
            out_batch[k] = torch.stack(out_batch[k], dim=0)
    return out_batch


class RandomRotate:
    def __init__(self,
                 p_threshold=0.5,
                 rotate_angle_range: list = (-180, 180)):
        self.p_threshold = p_threshold
        self.rotate_angle_range = rotate_angle_range

    # @profile
    def __call__(self, sample):
        if random.uniform(0, 1) > self.p_threshold:
            return sample
        img = sample['rgb_img']
        angle = np.random.uniform(*self.rotate_angle_range)
        warp_m, warped_img_size = get_rotate_matrix(img.shape[:2][::-1],
                                                    angle=angle)
        warped_img = cv2.warpPerspective(img, warp_m, dsize=warped_img_size)
        sample['eye_bbox'] = warp_boxes(sample['eye_bbox'], warp_m,
                                        warped_img.shape[1],
                                        warped_img.shape[0])
        sample['rgb_img'] = warped_img
        return sample


class RandomAxisRotate:
    def __init__(self,
                 p_threshold=0.5,
                 x_angle_range: list = (0, 0),
                 y_angle_range: list = (0, 0),
                 z_angle_range: list = (0, 0)):
        self.p_threshold = p_threshold
        self.x_angle_range = x_angle_range
        self.y_angle_range = y_angle_range
        self.z_angle_range = z_angle_range

    # @profile
    def rotate(self, sample):
        if random.uniform(0, 1) > self.p_threshold:
            return sample
        # 眯眼不做axis rotate
        if sample['class_name'] in ['halfopen', 'eyehalpopen']:
            return sample
        img = sample['rgb_img']
        x_angle = np.random.uniform(*self.x_angle_range)
        y_angle = np.random.uniform(*self.y_angle_range)
        z_angle = np.random.uniform(*self.z_angle_range)
        img_h, img_w = img.shape[:2]
        warp_m = get_axis_rotate_matrix((img_w, img_h),
                                        x_angle=x_angle,
                                        y_angle=y_angle,
                                        z_angle=z_angle)
        img_warped = cv2.warpPerspective(img, warp_m, dsize=(img_w, img_h))
        sample['eye_bbox'] = warp_boxes(sample['eye_bbox'], warp_m,
                                        img_warped.shape[1],
                                        img_warped.shape[0])
        sample['rgb_img'] = img_warped
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.rotate, sample))
        return self.rotate(sample)


class BboxAroundCropV2:
    def __init__(self,
                 expand_ratio_range: list = (1.2, 1.5),
                 move_ratio_range: list = (-0.5, 0.5),
                 min_expand_pixel: int = 0,
                 num_lmks=0,
                 enable_choice_one=True) -> Dict:
        """_summary_

        Args:
            expand_ratio_range (list, optional): 控制bbox的扩大程度. Defaults to (1.2, 1.5).
            move_ratio_range (list, optional): 控制bbox相对于中心的偏移程度. Defaults to (-0.5, 0.5).
            min_expand_pixel (int, optional): _description_. Defaults to 0.
            num_lmks (int, optional): _description_. Defaults to 0.
            enable_choice_one (bool, optional): _description_. Defaults to True.

        Note:
            1. bbox的坐标是闭区间

        Returns:
            Dict: _description_
        """
        self.expand_ratio_range = expand_ratio_range
        self.min_expand_pixel = min_expand_pixel
        self.move_ratio_range = move_ratio_range
        self.num_lmks = num_lmks
        self.enable_choice_one = enable_choice_one

    # @profile
    def __call__(self, sample):
        img = sample['rgb_img']
        bboxes = sample['eye_bbox']

        img_h, img_w = img.shape[:2]
        raw_bboxes_wh = bboxes[:, [2, 3]] - bboxes[:, [0, 1]] + 1
        # Tidy eye bbox
        raw_bboxes_wh[:, 1] = raw_bboxes_wh[:, 0]  # set w = h
        eye_bbox_ctr = (bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2
        bboxes[:, [0, 1]] = eye_bbox_ctr - raw_bboxes_wh / 2
        bboxes[:, [2, 3]] = eye_bbox_ctr + raw_bboxes_wh / 2
        bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]],
                                    a_min=0,
                                    a_max=img.shape[1] - 1)
        bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]],
                                    a_min=0,
                                    a_max=img.shape[0] - 1)

        expanded_bboxes = bboxes.copy()
        expanded_bboxes = expanded_bboxes.astype(np.float32)
        if self.min_expand_pixel > 0:
            expanded_bboxes[:, [0, 1]] -= self.min_expand_pixel
            expanded_bboxes[:, [2, 3]] += self.min_expand_pixel
        # Width
        expand_ratio = random.uniform(*self.expand_ratio_range) - 1.0
        expand_w = expand_ratio * raw_bboxes_wh[:, 0]
        l_expand_ratio = 0.5 + random.uniform(*self.move_ratio_range)
        l_expand = l_expand_ratio * expand_w
        r_expand = expand_w - l_expand
        expanded_bboxes[:, [0]] -= l_expand
        expanded_bboxes[:, [2]] += r_expand
        # Height
        # expand_ratio = random.uniform(*self.expand_ratio_range) - 1.0
        expand_h = expand_ratio * raw_bboxes_wh[:, 1]
        up_expand_ratio = 0.5 + random.uniform(*self.move_ratio_range)
        up_expand = up_expand_ratio * expand_h
        down_expand = expand_h - up_expand
        expanded_bboxes[:, [1]] -= up_expand
        expanded_bboxes[:, [3]] += down_expand
        # Clip to keep expand_bbox in image
        expanded_bboxes = expanded_bboxes.astype(np.int32)  # Convert to int
        expanded_bboxes[:, [0, 2]] = np.clip(expanded_bboxes[:, [0, 2]],
                                             a_min=0,
                                             a_max=img_w - 1)
        expanded_bboxes[:, [1, 3]] = np.clip(expanded_bboxes[:, [1, 3]],
                                             a_min=0,
                                             a_max=img_h - 1)
        # Update value
        bboxes[:, [0, 1]] = expanded_bboxes[:, [0, 1]]
        bboxes[:, [2, 3]] = expanded_bboxes[:, [0, 1]]
        # face_gt[:, :4] = bboxes
        # face_gt[:, 5:-1] = lmks.reshape(num_bboxes, -1)
        # Crop image
        cropped_imgs = []
        for bbox in expanded_bboxes:
            cropped_img = img[int(bbox[1]):int(bbox[3]),
                              int(bbox[0]):int(bbox[2]), :]
            cropped_imgs.append(cropped_img)
        # Return
        if self.enable_choice_one:
            idx = random.choice(range(len(cropped_imgs)))
            sample['rgb_img'] = cropped_imgs[idx]
            sample['eye_bbox'] = bboxes
            return sample
        else:
            # raise NotImplementedError
            samples = []
            for cropped_img, cur_bbox in zip(cropped_imgs, bboxes):
                out_sample = {
                    'img_name': sample['img_name'],
                    'rgb_img': cropped_img,
                    'eye_bbox': np.expand_dims(cur_bbox, axis=0)
                }
                for k, v in sample.items():
                    if k not in out_sample:
                        out_sample[k] = v
                samples.append(out_sample)
                # if 'head_pose_gt' in sample.keys():
                #     samples[-1]['head_pose_gt'] = sample['head_pose_gt']
            return samples


class Resize:
    def __init__(self,
                 size: tuple = (224, 224),
                 max_length=224,
                 mode='keep_aspect_ratio_resize'):
        self.size = size  # [img_w, img_h]
        self.max_length = max_length
        self.mode = mode

    def fix_resize(self, sample):
        img, bboxes = sample['rgb_img'], sample['face_gt']
        h, w = img.shape[:2]
        x_ratio = self.size[0] / w
        y_ratio = self.size[1] / h

        img = cv2.resize(img, None, fx=x_ratio, fy=y_ratio)
        sample['rgb_img'] = img
        return sample

    def keep_aspect_ratio_resize(self, sample):
        img = sample['rgb_img']
        M = get_resize_matrix(img.shape[:2][::-1], self.size, True)
        img = cv2.warpPerspective(img, M, dsize=tuple(self.size))
        sample['rgb_img'] = img
        return sample

    # @profile
    def __call__(self, sample):
        if self.mode == 'fix_resize':
            if isinstance(sample, list):
                return list(map(self.fix_resize, sample))
            return self.fix_resize(sample)
        elif self.mode == 'keep_aspect_ratio_resize':
            if isinstance(sample, list):
                return list(map(self.keep_aspect_ratio_resize, sample))
            return self.keep_aspect_ratio_resize(sample)


class Flip:
    def __init__(self, p_threshold=0.5):
        self.p_threshold = p_threshold

    # @profile
    def flip(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample

        img = sample['rgb_img']
        img_flipped = cv2.flip(img, 1)
        sample['rgb_img'] = img_flipped
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.flip, sample))
        return self.flip(sample)


class ColorJitter:
    def __init__(self, p_threshold=0.5, h_gain=5, s_gain=50, v_gain=50):
        self.hsv_gain = [h_gain, s_gain, v_gain]
        self.p_threshold = p_threshold

    # @profile
    def color_jitter(self, sample):
        if random.uniform(0, 1) > self.p_threshold:
            return sample
        img = sample['rgb_img']
        img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        hsv_value = np.random.uniform(-1, 1, 3) * self.hsv_gain
        hsv_value *= np.random.randint(0, 2, 3)  # Random selection of h, s, v
        img_hsv[..., 0] = np.clip(img_hsv[..., 0] + hsv_value[0], 0, 180)
        img_hsv[..., 1] = np.clip(img_hsv[..., 1] + hsv_value[1], 0, 255)
        img_hsv[..., 2] = np.clip(img_hsv[..., 2] + hsv_value[2], 0, 255)
        img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2RGB)
        sample['rgb_img'] = img
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.color_jitter, sample))
        return self.color_jitter(sample)


# class ImgaugColorChange:
#     def __init__(self, p_threshold=0.5):
#         self.p_threshold = p_threshold
#         self.imgaug_seq = iaa.SomeOf(
#             1,
#             [
#                 # iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
#                 # iaa.MultiplyHueAndSaturation(mul_hue=(0.5, 1.5))
#                 # iaa.ChangeColorTemperature((1100, 10000))
#                 # iaa.MultiplyHue((0.5, 0.6))
#                 iaa.GammaContrast((0.5, 3.0), per_channel=True)
#             ])

#     # @profile
#     def process(self, sample):
#         if random.uniform(0, 1) > self.p_threshold:
#             return sample
#         sample['rgb_img'] = self.imgaug_seq(images=[sample['rgb_img']])[0]
#         return sample

#     def __call__(self, sample):
#         if isinstance(sample, list):
#             return list(map(self.process, sample))
#         return self.process(sample)


class AddBrightSplot:
    def __init__(self,
                 bs_img_dir=None,
                 p_threshold=0.5,
                 max_num_bs=5,
                 bs_area_ratio_range=(0, 0)):
        self.p_threshold = p_threshold
        self.max_num_bs = max_num_bs
        self.bs_area_ratio_range = bs_area_ratio_range
        print(f"bs_img_dir: {bs_img_dir}")
        bs_img_paths = Path(bs_img_dir).glob('**/*.png')
        self.bs_idxes = []
        self.bs_imgs = []
        for bs_img_idx, bs_img_path in enumerate(bs_img_paths):
            bs_img = cv2.imread(bs_img_path.as_posix(), cv2.IMREAD_UNCHANGED)
            self.bs_idxes.append(bs_img_idx)
            self.bs_imgs.append(bs_img)

    def _draw(self, img, bright_splot_img, x=0, y=0):
        bs_img_h, bs_img_w = bright_splot_img.shape[:2]
        bs_mask = bright_splot_img[:, :, 3:4]
        bs_mask = bs_mask.astype(np.float32) / 255

        if random.uniform(0, 1) < 0.7:
            bs_mask = cv2.cvtColor(bs_mask[:, :, 0], cv2.COLOR_GRAY2BGR)
            bs_mask = cv2.GaussianBlur(bs_mask, (5, 5), 0)
            bs_mask = cv2.cvtColor(bs_mask, cv2.COLOR_BGR2GRAY)
            bs_mask = bs_mask[:, :, np.newaxis]

        img[y:y + bs_img_h,
            x:x + bs_img_w] = bs_mask * bright_splot_img[:, :, :3] + (
                1 - bs_mask) * img[y:y + bs_img_h, x:x + bs_img_w]

    # @profile
    def add_bs(self, sample):
        if random.uniform(0, 1) > self.p_threshold:
            return sample
        num_bs = random.randint(1, self.max_num_bs)
        img = sample['rgb_img']
        img_h, img_w = img.shape[:2]
        img_area = img_h * img_w
        for _ in range(num_bs):
            bs_img_idx = random.choice(self.bs_idxes)
            bs_img = self.bs_imgs[bs_img_idx]
            # Rotate
            angle = np.random.uniform(-180, 180)
            warp_m, warped_img_size = get_rotate_matrix(bs_img.shape[:2][::-1],
                                                        angle=angle)
            bs_img = cv2.warpPerspective(bs_img, warp_m, dsize=warped_img_size)

            bs_area_ratio = np.random.uniform(*self.bs_area_ratio_range)
            bs_img_area = img_area * bs_area_ratio
            bs_img_max_length = np.sqrt(bs_img_area)
            bs_resize_ratio = bs_img_max_length / max(bs_img.shape[:2])
            bs_img = cv2.resize(bs_img,
                                None,
                                fx=bs_resize_ratio,
                                fy=bs_resize_ratio)
            img_h, img_w = img.shape[:2]
            min_y, min_x = bs_img.shape[:2]
            min_y += 1
            min_x += 1
            max_x = img_w - bs_img.shape[1] - 1
            max_y = img_h - bs_img.shape[0] - 1
            if min_x >= max_x or min_y >= max_y:
                return sample
            x = random.randint(min_x, max_x)
            y = random.randint(min_y, max_y)
            self._draw(img, bs_img, x=x, y=y)
        sample['rgb_img'] = img
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.add_bs, sample))
        return self.add_bs(sample)


class Gray:
    def gray(self, sample):
        img = sample['rgb_img']
        gray_img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        gray_img = np.expand_dims(gray_img, axis=-1)
        sample['gray_img'] = gray_img
        return sample

    # @profile
    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.gray, sample))
        return self.gray(sample)


class ChangeGrayBrightness:
    def __init__(self, p_threshold=0.5, brightness_range=(-30, 30)):
        self.p_threshold = p_threshold
        self.brightness_range = brightness_range

    # @profile
    def process(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample
        img = sample['gray_img']
        img = img.astype(np.float32)
        delta = random.randrange(*self.brightness_range)
        img += delta
        img = np.clip(img, 0, 255).astype(np.uint8)
        sample['gray_img'] = img
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.process, sample))
        return self.process(sample)


class ChangeGrayContrast:
    def __init__(self, p_threshold=0.5, contrast_range=(0.5, 1.5)):
        self.p_threshold = p_threshold
        self.contrast_range = contrast_range

    # @profile
    def process(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample
        img = sample['gray_img']
        alpha = random.uniform(*self.contrast_range)
        img = img * alpha
        img = np.clip(img, 0, 255).astype(np.uint8)
        sample['gray_img'] = img
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.process, sample))
        return self.process(sample)


class RandomErase:
    def __init__(self, p_threshold=0.5, erase_area_size_ratio_range=(0, 0.5)):
        self.p_threshold = p_threshold
        self.erase_area_size_ratio_range = erase_area_size_ratio_range

    def process(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample
        img = sample['gray_img']
        erase_area_size_ratio = random.uniform(
            *self.erase_area_size_ratio_range)
        img = random_erase(img,
                           erase_area_size_ratio=erase_area_size_ratio,
                           valid_region_bbox=None)
        sample['gray_img'] = img
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.process, sample))
        return self.process(sample)


class RandomAddExtraImg:
    def __init__(self, p_threshold=0.5, extra_img_dir=None):
        self.p_threshold = p_threshold
        # self.p_threshold = 1
        extra_img_dir = Path(extra_img_dir)
        extra_img_paths = list(extra_img_dir.glob('*.png'))
        self.extra_img_paths = extra_img_paths

    # @profile
    def process(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample
        eye_type = sample['img_name'].split(',')[-2]
        eye_bbox = sample['eye_bbox'].reshape(-1).astype(np.int32)
        face_landmark = sample['face_landmark']

        if eye_type == 'left':
            if sample['mode'] == 'st':
                x = int(face_landmark[104][0])
                y = int(face_landmark[104][1])
            elif sample['mode'] == 'st_with_arc_eye':
                x = int(face_landmark[57][0])
                y = int(face_landmark[57][1])
            else:
                raise ValueError()
        elif eye_type == 'right':
            if sample['mode'] == 'st':
                x = int(face_landmark[105][0])
                y = int(face_landmark[105][1])
            elif sample['mode'] == 'st_with_arc_eye':
                x = int(face_landmark[64][0])
                y = int(face_landmark[64][1])
            else:
                raise ValueError()
        else:
            raise ValueError(f"eye_type: {eye_type}, {sample['img_name']}")

        img = sample['rgb_img']
        img = random_add_extra_img(img,
                                   self.extra_img_paths,
                                   valid_region_bbox=None,
                                   x=x,
                                   y=y)
        # cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
        sample['rgb_img'] = img
        sample['class_name'] = 'invisible'
        sample['class_id'] = EYE_STATUS_ENCODE_MAP['invisible']
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.process, sample))
        return self.process(sample)


class AddNoise:
    def __init__(self,
                 p_threshold=0.5,
                 salt_and_pepper_noise_thresold_range=(0.001, 0.05),
                 gaussian_noise_threshold_range=(0, 1)):
        self.p_threshold = p_threshold
        self.salt_and_pepper_noise_thresold_range = salt_and_pepper_noise_thresold_range
        self.gaussian_noise_threshold_range = gaussian_noise_threshold_range
        # self.noise_type = ['salt_and_pepper', 'gaussian']
        self.noise_type = ['gaussian']

    # @profile
    def process(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample

        img = sample['gray_img']
        noise_type = random.choice(self.noise_type)

        if noise_type == 'salt_and_pepper':
            threshold = random.uniform(
                *self.salt_and_pepper_noise_thresold_range)
            img = add_salt_and_pepper_noise(img, threshold)
        elif noise_type == 'gaussian':
            threshold = random.uniform(*self.gaussian_noise_threshold_range)
            img = add_gaussian_noise(img, threshold)
        sample['gray_img'] = img
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.process, sample))
        return self.process(sample)


class AddGaussianBlur:
    def __init__(self, p_threshold=0.5, sigma_range=(0, 3)):
        self.p_threshold = p_threshold
        self.sigma_range = sigma_range

    # @profile
    def process(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample

        img = sample['gray_img']
        sigma = random.uniform(*self.sigma_range)
        img = add_gaussian_blur(img, sigma=sigma)
        sample['gray_img'] = img
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.process, sample))
        return self.process(sample)


class EqualizeHist:
    def __init__(self, p_threshold=0.5):
        self.p_threshold = p_threshold

    # @profile
    def process(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample

        img = sample['gray_img']
        img = equalize_hist(img)
        sample['gray_img'] = img
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.process, sample))
        return self.process(sample)


class Normalize:
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
    def normalize(self, sample):
        for k, v in sample.items():
            if k == 'rgb_img':
                v = normalize(v, mean=self.mean, std=self.std)
            elif k == 'gray_img':
                v = normalize(v, mean=self.gray_mean, std=self.gray_std)
            sample[k] = v
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.normalize, sample))
        return self.normalize(sample)


class EncodeToOnehot:
    def __init__(self, num_classes=2, mode='train'):
        self.num_classes = num_classes
        self.mode = mode

    # @profile
    def encode_to_onehot(self, sample):
        one_hot_value = np.zeros((self.num_classes), dtype=np.long)
        if sample['class_name'] in ['invisible', 'halfopen']:
            if self.num_classes != 3:
                raise ValueError('[ERROR] num_classes not match sample class!')
            else:
                sample['class_id'] = EYE_STATUS_ENCODE_MAP['invisible']
        one_hot_value[sample['class_id']] = 1
        # if self.mode == 'train':  # FIXME
        #     one_hot_value[class_id] = 1
        sample['gt'] = one_hot_value
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.encode_to_onehot, sample))
        return self.encode_to_onehot(sample)


class Mixup:
    def __init__(self,
                 p_threshold=1.0,
                 mixup_alpha=1.,
                 mode='batch',
                 correct_lam=True,
                 label_smoothing=0.1,
                 num_classes=3):
        self.mixup = MixupAug(mixup_alpha=mixup_alpha,
                              prob=p_threshold,
                              mode=mode,
                              correct_lam=correct_lam,
                              label_smoothing=label_smoothing,
                              num_classes=num_classes)

    def process(self, batch_sample):
        x = batch_sample['gray_imgs']
        target = batch_sample['gts']
        if target.shape[
                1] > 1:  # MEMO: target可能为onehot之后的结果，但mixup需要onehot前的值作为输入
            _, idx_y = torch.where(target == 1)
            target = idx_y.reshape(-1, 1)
        x, target = self.mixup(x, target)
        batch_sample['gray_imgs'] = x
        batch_sample['gts'] = target

    def __call__(self, batch_sample):
        return self.process(batch_sample)


class ToTensor:
    # @profile
    def to_tensor(self, sample):
        for k, v in sample.items():
            if isinstance(v, str) or isinstance(v, Path):
                continue
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).to(torch.float32)
            if k == 'rgb_img':
                v = v.permute(2, 0, 1)
            elif k == 'gray_img':
                v = v.permute(2, 0, 1)
                # v = v.cuda()
            sample[k] = v
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.to_tensor, sample))
        return self.to_tensor(sample)
