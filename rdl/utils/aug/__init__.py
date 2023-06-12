from typing import List
from pathlib import Path
import random
import math
import numpy as np
import cv2
from torch import rand
from ..common import crop_img_with_alpha_channel, get_axis_rotate_matrix, get_rotate_matrix

__all__ = [
    'add_salt_and_pepper_noise', 'add_gaussian_noise', 'add_gaussian_blur',
    'equalize_hist', 'random_erase', 'random_add_extra_img'
]


# ******
# Choose m from n aug
# ******
class RandomChooseKAugs:
    def __init__(self, augs: List, k=1):
        self.augs = augs
        self.k = 1

    def __call__(self, sample):
        augs = random.choices(self.augs, k=self.k)
        for aug in augs:
            sample = aug(sample)
        return sample


# ******
# Noise
# ******
# Salt and pepper noise
def add_salt_and_pepper_noise(img, threshold=0.05):
    if threshold > 0.1:
        print('[add_salt_and_pepper_noise] not support threshold > 0.1')
    threshold = np.clip(threshold, 0, 0.1)

    img = img.copy()
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    img_h, img_w, img_c = img.shape[:3]
    p_m = np.random.uniform(0, 1, (img_h, img_w, img_c))
    p_m[p_m <= threshold] = 1
    p_m[p_m != 1] = 0
    salt_pepper_p_m = np.random.randn(img_h, img_w, img_c)
    if img_c == 1:
        img[(p_m == 1) & (salt_pepper_p_m <= 0.5)] = [0]
        img[(p_m == 1) & (salt_pepper_p_m > 0.5)] = [255]
    else:
        img[(p_m == 1) & (salt_pepper_p_m <= 0.5)] = [0, 0, 0]
        img[(p_m == 1) & (salt_pepper_p_m > 0.5)] = [255, 255, 255]
    return img


def add_gaussian_noise(img, threshold=0.7, sigma_range=(0, 0.1)):
    img = img.copy()
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    img_h, img_w, img_c = img.shape[:3]
    p_m = np.random.uniform(0, 1, (img_h, img_w, img_c))
    p_m[p_m <= threshold] = 1
    p_m[p_m != 1] = 0

    sigma = random.uniform(*sigma_range)
    gaussian_noise_p_m = np.random.normal(0, sigma,
                                          (img_h, img_w, img_c)) * 255
    gaussian_noise_p_m = np.clip(gaussian_noise_p_m, -255, 255)

    img = img.astype(np.float32)
    img[p_m == 1] += gaussian_noise_p_m[p_m == 1]
    img = np.clip(img, 0, 255)
    return img.astype(np.uint8)


# ******
# Blur
# ******


def add_gaussian_blur(img, sigma=0.1, ksize=None):
    def compute_gaussian_blur_ksize(sigma):
        ksize = 0
        if sigma < 3.0:
            ksize = 3.3 * sigma  # 99% of weight
        elif sigma < 5.0:
            ksize = 2.9 * sigma  # 97% of weight
        else:
            ksize = 2.6 * sigma  # 95% of weight

        # we use 5x5 here as the minimum size as that simplifies
        # comparisons with gaussian_filter() in the tests
        # TODO reduce this to 3x3
        ksize = int(ksize)
        if ksize % 2 == 0:
            ksize -= 1
        ksize = int(max(ksize, 5))
        return ksize

    if ksize is None:
        ksize = compute_gaussian_blur_ksize(sigma)
    sigma = np.clip(sigma, 0, 5)
    img[:, :, 0] = cv2.GaussianBlur(img[:, :, 0], (ksize, ksize), sigma)
    return img


# ******
# Color
# ******


def equalize_hist(img):
    if img.ndim == 2:
        img = np.expand_dims(img, -1)
    img[:, :, 0] = cv2.equalizeHist(img[:, :, 0])
    return img


# *****
# Random Erase
# *****


def random_erase(in_img, erase_area_size_ratio=0.5, valid_region_bbox=None):
    img = in_img.copy()
    if img.ndim == 2:
        img = np.expand_dims(img, -1)

    img_h, img_w, img_c = img.shape
    if valid_region_bbox is None:
        img_area = img_h * img_w
    else:
        valid_region_bbox_w = valid_region_bbox[2] - valid_region_bbox[0]
        valid_region_bbox_h = valid_region_bbox[3] - valid_region_bbox[1]
        img_area = valid_region_bbox_w * valid_region_bbox_h

    erase_area = int(img_area * erase_area_size_ratio)
    if erase_area == 0:
        return img
    half_erase_area = math.sqrt(erase_area)
    erase_bbox_h = random.randint(int(half_erase_area * 0.5),
                                  int(half_erase_area * 2))
    erase_bbox_w = erase_area / erase_bbox_h

    if valid_region_bbox is None:
        region_bbox = [0, 0, img_w - 1, img_h - 1]
    else:
        region_bbox = valid_region_bbox

    c_x = random.randint(region_bbox[0], region_bbox[2])
    c_y = random.randint(region_bbox[1], region_bbox[3])
    x1 = c_x - int(erase_bbox_w / 2)
    x2 = c_x + int(erase_bbox_w / 2)
    y1 = c_y - int(erase_bbox_h / 2)
    y2 = c_y + int(erase_bbox_h / 2)
    bbox = np.array([x1, y1, x2, y2])
    bbox[[0, 2]] = np.clip(bbox[[0, 2]], region_bbox[0], region_bbox[2])
    bbox[[1, 3]] = np.clip(bbox[[1, 3]], region_bbox[1], region_bbox[3])

    if img_c == 1:
        img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = [127]
    else:
        img[bbox[1]:bbox[3], bbox[0]:bbox[2]] = [127, 127, 127]
    return img


# ******
# Random add extra img
# ******


# @profile
def random_add_extra_img(in_img,
                         extra_img_paths,
                         valid_region_bbox=None,
                         x=None,
                         y=None):
    def gen_coordinates(region_bbox):
        region_bbox_wh = region_bbox[[2, 3]] - region_bbox[[0, 1]]
        num_pixels = region_bbox_wh[0] * region_bbox_wh[1]
        p_m = np.random.multivariate_normal([0, 0],
                                            [[0.5 / 3, 0], [0, 0.5 / 3]],
                                            size=num_pixels)
        p_m = p_m[(p_m[:, 0] > -0.5) & (p_m[:, 0] < 0.5) & (p_m[:, 1] > -0.5) &
                  (p_m[:, 1] < 0.5)]
        ct = (region_bbox[[0, 1]] + region_bbox[[2, 3]]) / 2
        coordinates = ct + p_m * region_bbox_wh
        coordinates = coordinates.astype(np.int32)
        # Debug
        # print(p_m)
        # import matplotlib.pyplot as plt
        # plt.plot(coordinates[:, 0], coordinates[:, 1], '.', alpha=0.01)
        # plt.axis('equal')
        # plt.grid()
        # plt.savefig('./out/1.jpg')
        # print(coordinates.min(), coordinates.max())
        return coordinates

    def aug_extra_img(img):
        # Rotate
        img_h, img_w = img.shape[:2]
        _, warp_m, warped_img_size = get_rotate_matrix(
            (img_w, img_h), angle=random.randint(-180, 180))
        img = cv2.warpPerspective(img, warp_m, dsize=warped_img_size)
        # Axis rotate
        img_h, img_w = img.shape[:2]
        x_angle = np.random.uniform(-30, 30)
        y_angle = np.random.uniform(-30, 30)
        warp_m = get_axis_rotate_matrix((img_w, img_h),
                                        x_angle=x_angle,
                                        y_angle=y_angle,
                                        z_angle=0)
        img = cv2.warpPerspective(img, warp_m, dsize=(img_w, img_h))
        # Flip
        if random.uniform(0, 1) < 0.5:
            img = cv2.flip(img, 1)
        return img

    # @profile
    def add_extra_img(img, extra_img, region_bbox, x=0, y=0):
        region_bbox = np.array(region_bbox)
        if extra_img.shape[2] != 4:
            print(f"[add_object_img] Error object_img.ndim != 4")
            return img
        img_h, img_w, img_c = img.shape

        # # Crop
        # extra_img = crop_img_with_alpha_channel(extra_img)
        # Resize
        region_bbox_wh = region_bbox[[2, 3]] - region_bbox[[0, 1]]
        extra_img_h, extra_img_w = extra_img.shape[:2]
        extra_img_long_len = max(extra_img_h, extra_img_w)
        region_bbox_short_len = np.min(region_bbox_wh)
        resize_ratio = region_bbox_short_len / extra_img_long_len
        # resize_ratio = (1 + random.uniform(-0.8, -0.1)) * resize_ratio
        extra_img = cv2.resize(extra_img,
                               None,
                               fx=resize_ratio,
                               fy=resize_ratio)
        extra_img_h, extra_img_w = extra_img.shape[:2]
        extra_img_alpha_c = extra_img[:, :, 3:4]
        # Add
        if img_c == 1:
            extra_img = cv2.cvtColor(extra_img[:, :, :3], cv2.COLOR_BGR2GRAY)
            extra_img = np.expand_dims(extra_img, axis=-1)
        else:
            extra_img = extra_img[:, :, :3]
        extra_img_bbox = np.array([
            x - extra_img_w / 2, y - extra_img_h / 2, x + extra_img_w / 2,
            y + extra_img_h / 2
        ], np.int32)
        extra_img_origin = extra_img_bbox[[0, 1]]
        extra_img_bbox[[0, 2]] = np.clip(extra_img_bbox[[0, 2]], 0, img_w - 1)
        extra_img_bbox[[1, 3]] = np.clip(extra_img_bbox[[1, 3]], 0, img_h - 1)
        extra_img_bbox_in_local = -extra_img_origin + extra_img_bbox.reshape(
            -1, 2)
        extra_img_bbox_in_local = extra_img_bbox_in_local.reshape(-1)

        img_roi = img[extra_img_bbox[1]:extra_img_bbox[3],
                      extra_img_bbox[0]:extra_img_bbox[2]]
        extra_img_alpha_c_roi = extra_img_alpha_c[
            extra_img_bbox_in_local[1]:extra_img_bbox_in_local[3],
            extra_img_bbox_in_local[0]:extra_img_bbox_in_local[2]]
        extra_img_alpha_c_roi = extra_img_alpha_c_roi.astype(np.float32) / 255
        extra_img_roi = extra_img[
            extra_img_bbox_in_local[1]:extra_img_bbox_in_local[3],
            extra_img_bbox_in_local[0]:extra_img_bbox_in_local[2]]
        img_roi = (1 - extra_img_alpha_c_roi
                   ) * img_roi + extra_img_alpha_c_roi * extra_img_roi
        img[extra_img_bbox[1]:extra_img_bbox[3],
            extra_img_bbox[0]:extra_img_bbox[2]] = img_roi

    img = in_img.copy()
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    img_h, img_w, img_c = img.shape
    if valid_region_bbox is None:
        region_bbox = np.array([0, 0, img_w - 1, img_h - 1])
    else:
        region_bbox = np.array(valid_region_bbox)

    if x is None or y is None:
        coordinates = gen_coordinates(region_bbox)
        cur_coordinate = random.choice(coordinates)
    else:
        cur_coordinate = [x, y]

    extra_img_path = random.choice(extra_img_paths)
    # print(f"[Debug] extra_img_path: {extra_img_path}")
    extra_img = cv2.imread(extra_img_path.as_posix(), cv2.IMREAD_UNCHANGED)
    # Crop
    extra_img = crop_img_with_alpha_channel(extra_img)
    # Aug extra_img
    extra_img = aug_extra_img(extra_img)
    # print(f"[Debug] extra_img shape: {extra_img.shape}")
    # cv2.imwrite(f"./tmp_out/{extra_img_path.name}", extra_img)
    # Add
    add_extra_img(img, extra_img, region_bbox, cur_coordinate[0],
                  cur_coordinate[1])
    return img


def gen_valid_region_bbox(in_bbox, len_range_ratio=None):
    in_bbox_wh = in_bbox[[2, 3]] - in_bbox[[0, 1]]
    valid_region_bbox_len_ratio = random.uniform(*len_range_ratio)
    valid_region_bbox_len = valid_region_bbox_len_ratio * in_bbox_wh.min()
    valid_region_bbox_half_len = int(valid_region_bbox_len / 2)
    x = random.randint(0, in_bbox_wh[0])
    y = random.randint(0, in_bbox_wh[1])
    valid_region_bbox = np.array([
        x - valid_region_bbox_half_len, y - valid_region_bbox_half_len,
        x + valid_region_bbox_half_len, y + valid_region_bbox_half_len
    ])
    valid_region_bbox = valid_region_bbox.astype(np.int32)
    return valid_region_bbox


# import line_profiler

# profiler = line_profiler.LineProfiler()
# profiler.add_function(random_add_extra_img)

extra_img_paths = []
# extra_img_paths.extend(
#     Path('/mnt/data/dataset/aug/coco_object-抠图/hand').glob('*.png'))
# extra_img_paths.extend(
#     Path('/mnt/data/dataset/aug/coco_object-抠图/glasses').glob('*.png'))
# extra_img_paths.extend(
#     Path('/mnt/data/dataset/aug/coco_object-抠图').glob('*.png'))
extra_img_paths.extend(
    Path('/mnt/data/dataset/aug/aug_mask_img-filtered').glob('*.png'))

if __name__ == '__main__':
    random.seed(0)
    np.random.seed(0)
    import cv2
    img = cv2.imread('./data/imgs/face_landmark/example_face_roi.jpg', 0)
    if img.ndim == 2:
        img = np.expand_dims(img, axis=-1)
    for idx in range(100):
        # img_augged = add_salt_and_pepper_noise(img, threshold=0.001)
        # img_augged = add_gaussian_noise(img, threshold=1)
        # img_augged = add_gaussian_blur(img, sigma=5)
        # img_augged = equalize_hist(img)
        # img_augged = random_erase(img, erase_area_size_ratio=0.1)
        img_augged = random_add_extra_img(img, extra_img_paths)
        img_shown = np.hstack([img, img_augged])
        cv2.imwrite(f"./out/out-test_rdl/tmp/augged-{idx}.jpg", img_shown)
        # break