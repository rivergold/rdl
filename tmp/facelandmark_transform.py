from typing import List, Dict
from pathlib import Path
import random, math
import cv2
import numpy as np
import torch
from scipy.spatial.transform import Rotation

from rdl.data import sampler
from ....utils.common import cal_run_time
from ....utils.common import (flip_landmark_idxes, get_resize_matrix,
                              get_rotate_matrix, get_axis_rotate_matrix,
                              warp_boxes, warp_landmarks, distort_color,
                              normalize, eulerangles_to_rotation_matrix,
                              rotation_matrix_to_eulerangles, HeadFlip)
from ....utils.aug import RandomChooseKAugs, random_add_extra_img, gen_valid_region_bbox
from ....utils.aug.random_mask import box_erase, brush_erase


# @profile
def collate_fn(in_batch, **kwargs):
    # start_time = time.time()
    out_batch = {}

    if isinstance(in_batch[0], list):
        tmp_batch = []
        for each_batch in in_batch:
            tmp_batch.extend(each_batch)
        in_batch = tmp_batch

    # Init out_batch
    for k in in_batch[0].keys():
        out_batch[f'{k}s'] = []
    # Feed data
    for _, sample in enumerate(in_batch):
        for k, v in sample.items():
            if v is not None:
                #print(k)
                out_batch[f'{k}s'].append(v)
    # Stack data
    #ks = ['rgb_imgs', 'gray_imgs', 'head_pose_gts']
    ks = ['rgb_imgs', 'gray_imgs', 'head_pose_gts', 'head_pose_cls_gts']
    for k in ks:
        if len(out_batch[k]):
            # ! pass size_division as function param
            out_batch[k] = torch.stack(out_batch[k], dim=0)
    out_batch['face_gts'] = torch.cat(out_batch['face_gts'], dim=0)

    # Generate target
    if kwargs.get('anchorfree_generator'):
        out_batch['target'] = kwargs['anchorfree_generator'].gen_target(
            out_batch['rgb_imgs'].shape[2:4], out_batch['face_gts'])
    # print(f'[Debug] collate_fn time: {time.time() - start_time}')
    return out_batch


class RandomRotate:
    def __init__(self, p_threshold=0.6, rotate_angle_range: list = (-60, 60)):
        self.p_threshold = p_threshold
        self.rotate_angle_range = rotate_angle_range

    def headpose_outrange(self, head_pose, min=-99, max=99):
        mask1 = (head_pose >= max).astype(np.int8)
        mask2 = (head_pose < min).astype(np.int8)

        return (mask1 + mask2).sum()

    # @profile
    def __call__(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample
        raise ValueError("[RandomRotate]")
        img, face_gt = sample['rgb_img'], sample['face_gt']
        angle = np.random.uniform(*self.rotate_angle_range)
        r_m, warp_m, warped_img_size = get_rotate_matrix(img.shape[:2][::-1],
                                                         angle=angle)

        head_pose = sample['head_pose_gt']
        r = Rotation.from_euler('xyz', head_pose, degrees=True)
        head_pose_vec = r.as_matrix()
        head_pose_Vec_R = np.dot(r_m, head_pose_vec)
        r3 = Rotation.from_matrix(head_pose_Vec_R)
        head_pose_rotate = r3.as_euler('xyz', degrees=True)

        if self.headpose_outrange(head_pose_rotate, -99, 99):
            return sample
        #print('xxxxxxxxxxxxxxxxxxxxxxxx',img.dtype)
        warped_img = cv2.warpPerspective(img, warp_m, dsize=warped_img_size)
        bboxes = face_gt[:, :4]
        lmks = face_gt[:, 5:-1]
        K = bboxes.shape[0]

        bboxes = warp_boxes(bboxes,
                            warp_m,
                            width=warped_img_size[0],
                            height=warped_img_size[1])

        lmks = lmks.reshape(K, -1, 2).reshape(-1, 2)
        lmks = warp_landmarks(lmks, warp_m, warped_img_size[0],
                              warped_img_size[1])
        lmks = lmks.reshape(K, -1, 2).reshape(K, -1)
        face_gt[:, :4] = bboxes
        face_gt[:, 5:-1] = lmks

        sample['head_pose_gt'] = head_pose_rotate
        sample['rgb_img'] = warped_img
        sample['face_gt'] = face_gt
        # if 'head_pose_gt' in sample.keys():
        #     sample['head_pose_gt']['roll'] +=
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

    def headpose_outrange(self, head_pose, min=-99, max=99):
        mask1 = (head_pose >= max).astype(np.int8)
        mask2 = (head_pose < min).astype(np.int8)
        return (mask1 + mask2).sum()

    # @profile
    def rotate(self, sample):
        if random.uniform(0, 1) > self.p_threshold:
            return sample
        # TODO: 添加条件？
        img = sample['rgb_img']
        x_angle = np.random.uniform(*self.x_angle_range)
        y_angle = np.random.uniform(*self.y_angle_range)
        z_angle = np.random.uniform(*self.z_angle_range)
        img_h, img_w = img.shape[:2]
        warp_m, R = get_axis_rotate_matrix((img_w, img_h),
                                           x_angle=x_angle,
                                           y_angle=y_angle,
                                           z_angle=z_angle)

        #warp headpose
        head_pose = sample['head_pose_gt']
        #print(head_pose)
        r = Rotation.from_euler('xyz', head_pose, degrees=True)
        vec = r.as_rotvec()
        head_pose_Vec_R = np.dot(R, vec)

        r1 = Rotation.from_rotvec(head_pose_Vec_R)
        head_pose_rotate = r1.as_euler('xyz', degrees=True)
        #print(head_pose)
        if self.headpose_outrange(head_pose_rotate, -99, 99):
            return sample

        img_warped = cv2.warpPerspective(img, warp_m, dsize=(img_w, img_h))
        warped_img_size = (img_warped.shape[1], img_warped.shape[0])
        face_gt = sample['face_gt']
        bboxes = face_gt[:, :4]
        lmks = face_gt[:, 5:-1]
        K = bboxes.shape[0]
        bboxes = warp_boxes(bboxes,
                            warp_m,
                            width=warped_img_size[0],
                            height=warped_img_size[1])
        # lmks = lmks - np.array([img_w / 2, img_h / 2])
        lmks = lmks.reshape(K, -1, 2).reshape(-1, 2)
        lmks = warp_landmarks(lmks, warp_m, warped_img_size[0],
                              warped_img_size[1])
        lmks = lmks.reshape(K, -1, 2).reshape(K, -1)
        face_gt[:, :4] = bboxes
        face_gt[:, 5:-1] = lmks

        sample['head_pose_gt'] = head_pose_rotate
        sample['rgb_img'] = img_warped
        sample['face_gt'] = face_gt
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.rotate, sample))
        return self.rotate(sample)


class RandomCrop:
    def __init__(self, crop_ratio_range: list = (0.3, 1)):
        self.crop_ratio_range = crop_ratio_range

    def __call__(self, sample):
        img, bboxes = sample['rgb_img'], sample['face_gt']

        if bboxes.shape[0] < 5:
            return sample

        img_h, img_w = img.shape[:2]
        short_side = min(img_w, img_h)
        crop_size = int(random.uniform(*self.crop_ratio_range) * short_side)
        # print(crop_size)
        x1 = random.randint(0, img_w - crop_size)
        y1 = random.randint(0, img_h - crop_size)

        img_cropped = img[y1:y1 + crop_size, x1:x1 + crop_size, :]
        bboxes[:, [0, 2]] -= x1
        bboxes[:, [1, 3]] -= y1

        bboxes[:, [5, 7, 9, 11, 13]] -= x1
        bboxes[:, [6, 8, 10, 12, 14]] -= y1

        bboxes_ctr_xs = (bboxes[:, 0] + bboxes[:, 2]) / 2
        bboxes_ctr_ys = (bboxes[:, 1] + bboxes[:, 3]) / 2

        valid_idx = (bboxes_ctr_xs > 0) & (bboxes_ctr_xs < crop_size) & (
            bboxes_ctr_ys > 0) & (bboxes_ctr_ys < crop_size)
        bboxes = bboxes[valid_idx]
        if bboxes.shape[0] > 0:
            sample['rgb_img'] = img_cropped
            sample['face_gt'] = bboxes
        # If cropped image has no bbox, return raw sample
        return sample


class BboxAroundCrop:
    def __init__(self,
                 crop_ratio_range: list = (1.2, 1.5),
                 bbox_ctr_move_range: list = (-40, 40),
                 num_lmks=0,
                 enable_choice_one=True):
        self.crop_ratio_range = crop_ratio_range
        self.bbox_ctr_move_range = bbox_ctr_move_range
        self.num_lmks = num_lmks
        self.enable_choice_one = enable_choice_one

    def __call__(self, sample):
        rgb_imgs = []

        img = sample['rgb_img']
        img_h, img_w = img.shape[:2]

        bbox = sample['face_gt'][:, :4]

        # Random move bbox
        if self.bbox_ctr_move_range is not None:
            for idx in range(len(bbox)):
                bbox_ctr_move_x = random.uniform(*self.bbox_ctr_move_range)
                bbox_ctr_move_y = random.uniform(*self.bbox_ctr_move_range)
                bbox[idx, [0, 2]] += bbox_ctr_move_x
                bbox[idx, [1, 3]] += bbox_ctr_move_y

        center = (bbox[:, [0, 1]] + bbox[:, [2, 3]]) / 2
        wh = bbox[:, [2, 3]] - bbox[:, [0, 1]]
        wh_crop = wh * random.uniform(*self.crop_ratio_range)
        half_wh = wh_crop / 2
        bbox_crop = np.zeros_like(bbox, dtype=np.int32)
        bbox_crop[:, [0, 1]] = center - half_wh
        bbox_crop[:, [2, 3]] = center + half_wh
        bbox_crop[:, [0, 2]] = np.clip(bbox_crop[:, [0, 2]], 0, img_w - 1)
        bbox_crop[:, [1, 3]] = np.clip(bbox_crop[:, [1, 3]], 0, img_h - 1)

        face_gt = sample['face_gt'].copy()
        face_gt[:, :2] = bbox_crop[:, [0, 1]] - bbox_crop[:, [0, 1]]
        face_gt[:, 2:4] = bbox_crop[:, [2, 3]] - bbox_crop[:, [0, 1]]

        lmks = sample['face_gt'][:, 5:-1].reshape(
            -1, self.num_lmks, 2) - np.expand_dims(bbox_crop[:, [0, 1]], 1)
        face_gt[:, 5:-1] = lmks.reshape(-1, self.num_lmks * 2)
        for cur_bbox_crop in bbox_crop:
            img_crop = img[int(cur_bbox_crop[1]):int(cur_bbox_crop[3]),
                           int(cur_bbox_crop[0]):int(cur_bbox_crop[2]), :]
            rgb_imgs.append(img_crop)
        if self.enable_choice_one:
            idx = random.choice(range(len(rgb_imgs)))
            sample['rgb_img'] = rgb_imgs[idx]
            sample['face_gt'] = face_gt[[idx]]
            return sample
        else:
            samples = []
            for rgb_img, each_face_gt in zip(rgb_imgs, face_gt):
                samples.append({
                    'img_name': sample['img_name'],
                    'rgb_img': rgb_img,
                    'face_gt': np.expand_dims(each_face_gt, axis=0)
                })
        return samples


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

    def _check_lmks_valid(self, expanded_bboxes, lmks):
        valid_flags = (expanded_bboxes[:, [0]] <= lmks[:, :, [0]]) & (
            lmks[:, :, [0]] <= expanded_bboxes[:, [2]])
        if not valid_flags.all():
            return False
        return True

    # @profile
    def __call__(self, sample):
        img = sample['rgb_img']
        face_gt = sample['face_gt'].copy()
        bboxes = face_gt[:, :4]
        num_bboxes = bboxes.shape[0]
        lmks = face_gt[:, 5:-1]
        num_lmks = int(lmks.shape[1] / 2)
        assert num_lmks == self.num_lmks
        lmks = lmks.reshape(num_bboxes, self.num_lmks, 2)

        img_h, img_w = img.shape[:2]
        raw_bboxes_wh = bboxes[:, [2, 3]] - bboxes[:, [0, 1]] + 1
        # bbox_centers = (bboxes[:, [0, 1]] + bboxes[:, [2, 3]]) / 2
        expanded_bboxes = bboxes.copy()
        # Min expand pixel
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
        expand_ratio = random.uniform(*self.expand_ratio_range) - 1.0
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
        self._check_lmks_valid(expanded_bboxes, lmks)
        # Update value
        lmks[:, :, [0, 1]] -= expanded_bboxes[:, [0, 1]]
        bboxes[:, [0, 1]] -= expanded_bboxes[:, [0, 1]]
        bboxes[:, [2, 3]] -= expanded_bboxes[:, [0, 1]]
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
            sample['face_gt'] = face_gt[[idx]]
            return sample
        else:
            samples = []
            for cropped_img, cur_face_gt in zip(cropped_imgs, face_gt):
                samples.append({
                    'img_name': sample['img_name'],
                    'rgb_img': cropped_img,
                    'face_gt': np.expand_dims(cur_face_gt, axis=0)
                })
                if 'head_pose_gt' in sample.keys():
                    samples[-1]['head_pose_gt'] = sample['head_pose_gt']
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

        img_resized = cv2.resize(img, None, fx=x_ratio, fy=y_ratio)
        bboxes[:, [0, 2]] *= x_ratio
        bboxes[:, [1, 3]] *= y_ratio

        bboxes[:, [5, 7, 9, 11, 13]] *= x_ratio
        bboxes[:, [6, 8, 10, 12, 14]] *= y_ratio

        sample['rgb_img'] = img_resized
        sample['face_gt'] = bboxes
        return sample

    # @cal_run_time
    def keep_aspect_ratio_resize(self, sample):
        img = sample['rgb_img']
        bbox = sample['face_gt']
        M = get_resize_matrix(img.shape[:2][::-1], self.size, True)
        img = cv2.warpPerspective(img, M, dsize=tuple(self.size))
        # Bbox
        face_bbox = bbox[:, :4]
        face_bbox = warp_boxes(face_bbox, M, self.size[0], self.size[1])
        bbox[:, :4] = face_bbox

        lmks = bbox[:, 5:-1]
        K = lmks.shape[0]
        lmks = lmks.reshape(K, -1, 2).reshape(-1, 2)
        lmks = warp_landmarks(lmks, M, self.size[0], self.size[1])
        lmks = lmks.reshape(K, -1, 2).reshape(K, -1)
        bbox[:, 5:-1] = lmks
        sample['rgb_img'] = img
        sample['face_gt'] = bbox
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

    # @cal_run_time
    def flip(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample

        img, face_gt = sample['rgb_img'], sample['face_gt']
        bboxes = face_gt[:, :4]
        lmks = face_gt[:, 5:-1]
        num_lmks = int(lmks.shape[1] / 2)

        img_flip = cv2.flip(img, 1)
        img_center = np.array(img.shape[:2][::-1], dtype=np.float32) / 2

        bboxes[:, [0, 2]] += 2 * (img_center[0] - bboxes[:, [0, 2]])
        bboxes_w = np.abs(bboxes[:, 0] - bboxes[:, 2])
        bboxes[:, 0] -= bboxes_w
        bboxes[:, 2] += bboxes_w

        # Change x
        lmks[:, ::2] += 2 * (img_center[0] - lmks[:, ::2])
        # Change idx
        num_faces = len(bboxes)
        lmks = lmks.reshape(num_faces, num_lmks, 2)

        if num_lmks == 106:
            lmks_format = 'st'
        elif num_lmks == 100:
            lmks_format = 'st_with_arc_eye'
        for each_lmks in lmks:
            flip_landmark_idxes(each_lmks, lmk_format=lmks_format)
        # lmks = lmks.reshape(num_faces, -1)
        # face_gt[:, 5:-1] = lmks
        # flip head_pose

        head_pose = sample['head_pose_gt']
        head_pose = np.array([head_pose[0], -head_pose[1], -head_pose[2]])

        # head_pose = sample['head_pose_gt']
        # img_flip = sample['rgb_img']

        sample['head_pose_gt'] = head_pose

        sample['rgb_img'] = img_flip
        return sample

    # @profile
    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.flip, sample))
        return self.flip(sample)


class Gray:
    # @cal_run_time
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


# class ColorDistort(object):
#     def __init__(self,
#                  brightness: float = 0.2,
#                  contrast: List[float] = [0.8, 1.2],
#                  saturation: List[float] = [0.8, 1.2]):
#         self.brightness = brightness
#         self.contrast = contrast
#         self.saturation = saturation

#     def color_distort(self, sample):
#         img = sample['rgb_img']
#         img = distort_color(img,
#                             brightness=self.brightness,
#                             contrast=self.contrast,
#                             saturation=self.saturation)
#         sample['rgb_img'] = img
#         return sample

#     def __call__(self, sample):
#         if isinstance(sample, list):
#             return list(map(self.color_distort, sample))
#         return self.color_distort(sample)


class ChangeGrayBrightness:
    def __init__(self, p_threshold=0.5, brightness_range=(-30, 30)):
        self.p_threshold = p_threshold
        self.brightness_range = brightness_range

    # # @profile
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

    # @profile
    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.process, sample))
        return self.process(sample)


class ChangeGrayContrast:
    def __init__(self, p_threshold=0.5, contrast_range=(0.5, 1.5)):
        self.p_threshold = p_threshold
        self.contrast_range = contrast_range
        print('debug', self.contrast_range)

    # # @profile
    def process(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample
        img = sample['gray_img']
        alpha = random.uniform(*self.contrast_range)
        img = img * alpha
        img = np.clip(img, 0, 255).astype(np.uint8)
        sample['gray_img'] = img
        return sample

    # @profile
    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.process, sample))
        return self.process(sample)


class BoxErase:
    def __init__(self,
                 p_threshold=0.5,
                 max_num=1,
                 box_w_range=(0, 0),
                 box_h_range=(0, 0),
                 fill_value=255):
        self.p_threshold = p_threshold
        self.max_num = max_num
        self.box_w_range = box_w_range
        self.box_h_range = box_h_range
        self.fill_value = fill_value

    # @cal_run_time
    def process(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample
        img = sample['gray_img']
        img = box_erase(img,
                        max_num=self.max_num,
                        box_w_range=self.box_w_range,
                        box_h_range=self.box_h_range,
                        fill_value=self.fill_value)
        sample['gray_img'] = img
        return sample

    # @profile
    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.process, sample))
        return self.process(sample)


class BrushErase:
    def __init__(self,
                 p_threshold=0.5,
                 max_num=1,
                 num_vertex_range=[4, 12],
                 move_angle=2 * math.pi / 15,
                 mean_move_angle=(2 * math.pi / 5),
                 vertex_line_width_range=[12, 40],
                 fill_value=255):
        self.p_threshold = p_threshold
        self.max_num = max_num
        self.num_vertex_range = num_vertex_range
        self.move_angle = move_angle
        self.mean_move_angle = mean_move_angle
        self.vertex_line_width_range = vertex_line_width_range

        self.fill_value = fill_value

    # @cal_run_time
    def process(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample
        img = sample['gray_img']
        img = brush_erase(img,
                          max_num=self.max_num,
                          num_vertex_range=self.num_vertex_range,
                          move_angle=self.move_angle,
                          mean_move_angle=self.mean_move_angle,
                          vertex_line_width_range=self.vertex_line_width_range,
                          fill_value=self.fill_value)
        sample['gray_img'] = img
        return sample

    # @profile
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

    def process(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample
        img = sample['rgb_img']
        img_h, img_w = img.shape[:2]
        in_bbox = np.array([0, 0, img_w - 1, img_h - 1])
        valid_region_bbox = gen_valid_region_bbox(in_bbox, (0.6, 0.9))
        img = random_add_extra_img(img,
                                   self.extra_img_paths,
                                   valid_region_bbox=valid_region_bbox,
                                   x=None,
                                   y=None)
        # cv2.circle(img, (x, y), 5, (255, 255, 255), -1)
        sample['rgb_img'] = img
        return sample

    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.process, sample))
        return self.process(sample)


class WholeImgMixup:
    def __init__(self, p_threshold=0, mixup_alpha_range=(0.1, 0.4)):
        self.p_threshold = p_threshold
        self.mixup_alpha_range = mixup_alpha_range

    def process(self, sample):
        if random.uniform(0, 1) >= self.p_threshold:
            return sample
        mixup_alpha = random.uniform(*self.mixup_alpha_range)
        img = sample['rgb_img'].astype(np.float32)
        # Ref sample
        # ref_sample_idx = random.randint(0, len(self.dataset) - 1)
        # ref_sample = self.dataset[ref_sample_idx]
        # ref_img = ref_sample['rgb_img'].astype(np.float32)
        # ref_img = sample['aug_data']['whole_mixup_img'].astype(np.float32)
        ref_img = sample['aug_rgb_img'].astype(np.float32)
        # Mixup
        img = img * (1 - mixup_alpha) + ref_img * mixup_alpha
        sample['rgb_img'] = img.astype(np.uint8)
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
    # @cal_run_time
    def normalize(self, sample):
        for k, v in sample.items():
            if k == 'rgb_img':
                v = normalize(v, mean=self.mean, std=self.std)
            elif k == 'gray_img':
                v = normalize(v, mean=self.gray_mean, std=self.gray_std)
            elif k == 'face_gt':
                img_h, img_w = sample['rgb_img'].shape[:2]
                num_k = v.shape[0]
                v = v.astype(np.float32)
                lmks = v[:, 5:-1].reshape(num_k, -1, 2)
                lmks[:, :, 0] /= img_w
                lmks[:, :, 1] /= img_h
                lmks = lmks.reshape(num_k, -1)
                v[:, 5:-1] = lmks
            sample[k] = v
        return sample

    # @profile
    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.normalize, sample))
        return self.normalize(sample)


# # TODO: use utils.common normalize
# class ImageNetNormalize:
#     def __call__(self, sample):
#         for k, v in sample.items():
#             if k == 'rgb_img':
#                 # Must be RGB
#                 v = v.astype(np.float32)
#                 v = v / 255
#                 v = v - np.array([[0.485, 0.456, 0.406]])  # Image mean
#                 v = v / np.array([[0.229, 0.224, 0.225]])  # Image std
#             elif k == 'face_gt':
#                 v = v.astype(np.float32)
#             sample[k] = v
#         return sample


class ToTensor:
    def to_tensor(self, sample):
        for k, v in sample.items():
            if k == 'img_name':
                continue
            if isinstance(v, np.ndarray):
                v = torch.from_numpy(v).to(torch.float32)
            if k == 'rgb_img':
                v = v.permute(2, 0, 1)
            elif k == 'gray_img':
                v = v.permute(2, 0, 1)
            elif k == 'head_pose_cls_gt':
                v = v.to(torch.long)
            sample[k] = v
        return sample

    # @profile
    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.to_tensor, sample))
        return self.to_tensor(sample)


class ToHeadPoseBins:
    # angle(float) to bins(int)
    # angle range: [-99, 99), 3 degree a bin
    def __init__(self, max_angle=99, angle_per_bin=3):
        self.head_pose_bins = np.array(
            range(-max_angle, max_angle + 1, angle_per_bin))

    def get_bins(self, sample):
        head_pose = sample['head_pose_gt']
        head_pose_cls_gt = np.digitize(head_pose,
                                       self.head_pose_bins,
                                       right=True)
        # 保护
        head_pose_cls_gt[head_pose_cls_gt == 55] = 54
        # if (head_pose_cls_gt == 55).any():
        #     print(head_pose)
        # print(head_pose_cls_gt)
        sample.update({'head_pose_cls_gt': head_pose_cls_gt})
        return sample

    # @profile
    def __call__(self, sample):
        if isinstance(sample, list):
            return list(map(self.get_bins, sample))
        return self.get_bins(sample)
