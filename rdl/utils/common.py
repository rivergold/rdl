from asyncio import IncompleteReadError
from typing import Union, List
import random, time
from functools import wraps
import torch
import math
import numpy as np
import cv2
import matplotlib.font_manager as fm
from PIL import Image, ImageDraw, ImageFont
import textwrap
from rlogger import RLogger


def bgr2rgb(imgs: np.ndarray):
    """Convert np.ndarray bgr2rgb

        Arguments:
            imgs {np.ndarray} -- shape (N, H, W, C) bgr images

        Returns:
            np.ndarray -- shape (N, H, W, C) rgb images
    """
    imgs = imgs[..., ::-1]
    return imgs


def grid_imgs(imgs, num_cols=8):
    """[summary]

        Args:
            imgs (np.ndarray) -- shape (N, H, W, C):
            num_cols (int, optional): [description]. Defaults to 8.

        Returns:
            np.ndarray: shape (H, W, C)
        """
    num_imgs, h, w, c = imgs.shape
    num_cols = num_cols if num_imgs > num_cols else num_imgs
    num_rows = num_imgs // num_cols if num_imgs >= num_cols else 1
    out_img = np.ones((num_rows * h, num_cols * w, c), dtype=np.uint8) * 255
    for idx in range(num_imgs):
        row_idx = idx // num_cols
        col_idx = idx % num_cols
        out_img[row_idx * h:(row_idx + 1) * h,
                col_idx * w:(col_idx + 1) * w, :] = imgs[idx]
    return out_img


# def imagenet_normalize(img: np.ndarray):
#     img = img.astype(np.float32) / 255
#     img -= np.array([[0.485, 0.456, 0.406]])
#     img /= np.array([[0.229, 0.224, 0.225]])
#     return img

# def imagenet_denormalize(img: np.ndarray):
#     img *= np.array([[0.229, 0.224, 0.225]])
#     img += np.array([[0.485, 0.456, 0.406]])
#     img *= 255
#     return img.astype(np.uint8)


# @profile
def img_normalize(img,
                  mean: List[float] = [0., 0., 0.],
                  std: List[float] = [1., 1., 1.]):
    img = img.astype(np.float32)
    mean = np.array(mean, dtype=np.float32)
    std = np.array(std, dtype=np.float32)
    assert mean.shape[0] == img.shape[-1]
    assert std.shape[0] == img.shape[-1]
    # img /= 255
    # img = img - mean
    # img = img / std
    img = (img / 255 - mean) / std
    return img


def img_denormalize(img,
                    mean: List[float] = [0., 0., 0.],
                    std: List[float] = [1., 1., 1.]):
    mean = np.array(mean)
    std = np.array(std)
    assert mean.shape[0] == img.shape[-1]
    assert std.shape[0] == img.shape[-1]
    img *= std
    img += mean
    img *= 255
    return img.astype(np.uint8)


def get_model_grad_norm(model):
    total_grad_norm = 0
    for p in model.parameters():
        param_norm = p.grad.detach().data.norm(2)
        total_grad_norm += param_norm.item()**2
    total_grad_norm = total_grad_norm**0.5
    return total_grad_norm


def x_color2imgs(x: torch.Tensor,
                 mean: List[float] = [0., 0., 0.],
                 std: List[float] = [1., 1., 1.]) -> np.ndarray:
    """[summary]

    Args:
        x (torch.Tensor): shape [N, C, H, W], RGB or gray
        mean (List[float], optional): [description]. Defaults to [0., 0., 0.].
        std (List[float], optional): [description]. Defaults to [1., 1., 1.].
        enable_covert_color (bool, optional): [description]. Defaults to False.

    Returns:
        np.ndarray: [description]
    """
    imgs = x.permute(0, 2, 3, 1).cpu().detach().numpy().copy()
    num_channels = imgs.shape[-1]
    assert len(mean) == num_channels
    assert len(std) == num_channels
    imgs = denormalize(imgs, mean=mean, std=std)
    imgs = imgs.astype(np.uint8)
    imgs = np.ascontiguousarray(imgs)

    # Convert gray to rgb
    if imgs.shape[-1] == 1:
        tmp_imgs = []
        for idx in range(len(imgs)):
            tmp_imgs.append(
                cv2.cvtColor(imgs[idx][:, :, 0], cv2.COLOR_GRAY2RGB))
        tmp_imgs = np.stack(tmp_imgs, axis=0)
        imgs = tmp_imgs
    return imgs


def draw_bboxes(img: np.ndarray, bboxes: Union[np.ndarray,
                                               torch.Tensor]) -> np.ndarray:
    """Draw bboxes

    Args:
        img (np.ndarray): color_mode=BGR, shape [H, W, 3]
        bboxes (Union[np.ndarray, torch.Tensor]): shape [K, 4]

    Returns:
        np.ndarray
    """
    if isinstance(bboxes, torch.Tensor):
        bboxes = bboxes.cpu().data.numpy()
    h, w = img.shape[:2]
    k = bboxes.shape[0]
    if k == 0:
        return img
    bboxes[:, [0, 2]] = np.clip(bboxes[:, [0, 2]], 0, w - 1)
    bboxes[:, [1, 3]] = np.clip(bboxes[:, [1, 3]], 0, h - 1)
    bboxes = bboxes.astype(np.int32)
    for bbox in bboxes:
        cv2.rectangle(img, (bbox[0], bbox[1]), (bbox[2], bbox[3]),
                      (0, 255, 255), 2)
        # TODO: draw score
        # TODO: draw landmarks
    return img


def draw_lmks(
    img: np.ndarray, lmks: Union[np.ndarray, torch.Tensor],
    color=(0, 255, 0)) -> np.ndarray:
    """_summary_

    Args:
        img (np.ndarray): _description_
        lmks (Union[np.ndarray, torch.Tensor]): shape [k, num_lmks, 2]
        color (tuple, optional): RGB color. Defaults to (0, 255, 0).

    Returns:
        np.ndarray: _description_
    """
    if isinstance(lmks, torch.Tensor):
        lmks = lmks.cpu().data.numpy().copy()
    h, w = img.shape[:2]
    k = lmks.shape[0]
    if k == 0:
        return img
    lmks = lmks.reshape(k, -1, 2)
    lmks[:, :, 0] = np.clip(lmks[:, :, 0], 0, w - 1)
    lmks[:, :, 1] = np.clip(lmks[:, :, 1], 0, h - 1)
    radius = max(int(min(h, w) * 0.0025), 1)
    font = cv2.FONT_HERSHEY_SIMPLEX
    for lmk in lmks:
        for idx, point in enumerate(lmk):
            # if idx not in [52, 53, 72, 54, 55, 56, 57, 73, 74]:
            #     continue
            cv2.circle(img, (int(point[0]), int(point[1])),
                       radius,
                       color=color,
                       thickness=-1)
            # cv2.putText(img, f"{idx}", (int(point[0]), int(point[1])), font,
            #             0.3, (255, 0, 0), 1, cv2.LINE_AA)
    return img


def get_resize_matrix(raw_shape, dst_shape, keep_ratio):
    """
    Get resize matrix for resizing raw img to input size
    :param raw_shape: (width, height) of raw image
    :param dst_shape: (width, height) of input image
    :param keep_ratio: whether keep original ratio
    :return: 3x3 Matrix
    """
    r_w, r_h = raw_shape
    d_w, d_h = dst_shape
    Rs = np.eye(3)
    if keep_ratio:
        C = np.eye(3)
        C[0, 2] = -r_w / 2
        C[1, 2] = -r_h / 2

        if r_w / r_h < d_w / d_h:
            ratio = d_h / r_h
        else:
            ratio = d_w / r_w
        Rs[0, 0] *= ratio
        Rs[1, 1] *= ratio

        T = np.eye(3)
        T[0, 2] = 0.5 * d_w
        T[1, 2] = 0.5 * d_h
        return T @ Rs @ C
    else:
        Rs[0, 0] *= d_w / r_w
        Rs[1, 1] *= d_h / r_h
        return Rs


def get_rotate_matrix(img_size, angle):
    theta = angle / 180 * math.pi
    img_w, img_h = img_size
    t_m = np.eye(3)
    t_m[0, 2] = -(img_w - 1) / 2
    t_m[1, 2] = -(img_h - 1) / 2

    r_m = np.array([[math.cos(theta), -math.sin(theta), 0],
                    [math.sin(theta), math.cos(theta), 0], [0, 0, 1]])

    rect = np.array([[0, 0, 1], [img_w - 1, 0, 1], [img_w - 1, img_h - 1, 1],
                     [0, img_h - 1, 1]])
    rotated_rect = (r_m @ t_m @ rect.T).T
    x1, y1, x2, y2 = min(rotated_rect[:, 0]), min(rotated_rect[:, 1]), max(
        rotated_rect[:, 0]), max(rotated_rect[:, 1])
    rotated_img_size = (int(x2 - x1 + 1), int(y2 - y1 + 1))

    # print(rotated_rect)

    t_inv_m = np.eye(3)
    t_inv_m[0, 2] = (rotated_img_size[0] - 1) / 2
    t_inv_m[1, 2] = (rotated_img_size[1] - 1) / 2
    m = t_inv_m @ r_m @ t_m
    # out_img = cv2.warpPerspective(img, m, dsize=rotated_img_size)
    return r_m, m, rotated_img_size


def HeadFlip(head):
    rot_vec = np.array(head)
    # assert head.shape == (
    #     3,), f"The shape of headrotvec must be (3,), which is {head.shape} currently."

    rot_mat = cv2.Rodrigues(rot_vec)[0]
    z = rot_mat[:, 2]
    y = rot_mat[:, 1]

    z[0] = -z[0]
    x = np.cross(y, z)

    newrot_mat = np.array([x, y, z])
    newrot_vec = cv2.Rodrigues(newrot_mat)[0].T[0]
    return newrot_vec


def _is_rotation_matrix(R):
    R_inv = np.linalg.inv(R)
    should_be_identity = np.dot(R_inv, R)
    I = np.identity(3, dtype=R.dtype)
    n = np.linalg.norm(I - should_be_identity)
    return n < 1e-6


def rotation_matrix_to_eulerangles(R):
    assert (_is_rotation_matrix(R))

    sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])

    singular = sy < 1e-6

    if not singular:
        x = math.atan2(R[2, 1], R[2, 2])
        y = math.atan2(-R[2, 0], sy)
        z = math.atan2(R[1, 0], R[0, 0])
    else:
        x = math.atan2(-R[1, 2], R[1, 1])
        y = math.atan2(-R[2, 0], sy)
        z = 0

    return np.array([x, y, z]) * 180 / math.pi


def eulerangles_to_rotation_matrix(angles):
    pitch = angles[0]
    yaw = angles[1]
    roll = angles[2]
    Rx = np.array([[1, 0, 0], [0, np.cos(pitch), -np.sin(pitch)],
                   [0, np.sin(pitch), np.cos(pitch)]],
                  dtype=np.float32)
    Ry = np.array([
        [np.cos(yaw), 0, np.sin(yaw)],
        [0, 1, 0],
        [-np.sin(yaw), 0, np.cos(yaw)],
    ],
                  dtype=np.float32)
    Rz = np.array([[np.cos(roll), -np.sin(roll), 0],
                   [np.sin(roll), np.cos(roll), 0], [0, 0, 1]],
                  dtype=np.float32)
    R = np.dot(Rz, np.dot(Ry, Rx))
    return R


def get_axis_rotate_matrix(img_size, x_angle, y_angle, z_angle):
    img_w, img_h = img_size

    def deg_to_rad(deg):
        return deg * math.pi / 180.0

    def get_rad(x_angle, y_angle, z_angle):
        return (deg_to_rad(x_angle), deg_to_rad(y_angle), deg_to_rad(z_angle))

    def get_M(img_w, img_h, focal, theta, phi, gamma, dx, dy, dz):
        w = img_w
        h = img_h
        f = focal

        # Projection 2D -> 3D matrix
        A1 = np.array([[1, 0, -w / 2], [0, 1, -h / 2], [0, 0, 1], [0, 0, 1]])

        # Rotation matrices around the X, Y, and Z axis
        RX = np.array([[1, 0, 0, 0], [0, np.cos(theta), -np.sin(theta), 0],
                       [0, np.sin(theta), np.cos(theta), 0], [0, 0, 0, 1]])

        RY = np.array([[np.cos(phi), 0, -np.sin(phi), 0], [0, 1, 0, 0],
                       [np.sin(phi), 0, np.cos(phi), 0], [0, 0, 0, 1]])

        RZ = np.array([[np.cos(gamma), -np.sin(gamma), 0, 0],
                       [np.sin(gamma), np.cos(gamma), 0, 0], [0, 0, 1, 0],
                       [0, 0, 0, 1]])

        # Composed rotation matrix with (RX, RY, RZ)
        R = np.dot(np.dot(RX, RY), RZ)

        # Translation matrix
        T = np.array([[1, 0, 0, dx], [0, 1, 0, dy], [0, 0, 1, dz],
                      [0, 0, 0, 1]])

        # Projection 3D -> 2D matrix
        A2 = np.array([[f, 0, w / 2, 0], [0, f, h / 2, 0], [0, 0, 1, 0]])

        # Final transformation matrix
        return np.dot(A2, np.dot(T, np.dot(R, A1))), R[:3, :3]

    # Get radius of rotation along 3 axes
    rtheta, rphi, rgamma = get_rad(x_angle, y_angle, z_angle)

    # Get ideal focal length on z axis
    # NOTE: Change this section to other axis if needed
    d = np.sqrt(img_h**2 + img_w**2)
    focal = d / (2 * np.sin(rgamma) if np.sin(rgamma) != 0 else 1)
    dz = focal

    # Get projection matrix
    M = get_M(img_w, img_h, focal, rtheta, rphi, rgamma, 0, 0, dz)
    return M[0]


def warp_boxes(boxes, M, width, height):
    n = len(boxes)
    if n:
        # warp points
        xy = np.ones((n * 4, 3))
        xy[:, :2] = boxes[:, [0, 1, 2, 3, 0, 3, 2, 1]].reshape(
            n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = (xy[:, :2] / xy[:, 2:3]).reshape(n, 8)  # rescale
        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        xy = np.concatenate(
            (x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
        # clip boxes
        xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
        xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
        return xy.astype(np.float32)
    else:
        return boxes


def warp_landmarks(landmarks, M, width, height):
    # landmarks shape [K*5, 2]
    n = len(landmarks)
    if n:
        landmarks = np.concatenate([landmarks, np.ones((n, 1))], axis=-1)
        landmarks = landmarks @ M.T
        return landmarks[:, :2] / landmarks[:, 2:3]
    return landmarks


def random_brightness(img, delta):
    img += random.uniform(-delta, delta)
    return img


def random_contrast(img, alpha_low, alpha_up):
    img *= random.uniform(alpha_low, alpha_up)
    return img


def random_saturation(img, alpha_low, alpha_up):
    hsv_img = cv2.cvtColor(img.astype(np.float32), cv2.COLOR_BGR2HSV)
    hsv_img[..., 1] *= random.uniform(alpha_low, alpha_up)
    img = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
    return img


def distort_color(img, **kwargs):
    img = img.astype(np.float32) / 255
    if kwargs.get('brightness') and random.randint(0, 1):
        img = random_brightness(img, kwargs['brightness'])
    if 'contrast' in kwargs.get('contrast') and random.randint(0, 1):
        img = random_contrast(img, *kwargs['contrast'])
    if 'saturation' in kwargs and random.randint(0, 1):
        img = random_saturation(img, *kwargs['saturation'])
    img = img * 255
    return img.astype(np.uint8)


st_flip_idx_map = {
    # 眉毛-上
    37: 38,
    36: 39,
    35: 40,
    34: 41,
    33: 42,
    # 眉毛-下
    67: 68,
    66: 69,
    65: 70,
    64: 71,
    # 眼周围
    55: 58,
    54: 59,
    72: 75,
    53: 60,
    52: 61,
    56: 63,
    73: 76,
    57: 62,
    # 瞳孔
    104: 105,
    74: 77,
    # 鼻周
    78: 79,
    80: 81,
    82: 83,
    47: 51,
    48: 50,
    # 嘴
    84: 90,
    85: 89,
    86: 88,
    96: 100,
    97: 99,
    103: 101,
    95: 91,
    94: 92
}

st_with_arc_eye_flip_idx_map = {
    # 眉毛-上
    33: 46,
    34: 45,
    35: 44,
    36: 43,
    37: 42,
    # 眉毛-下
    41: 47,
    40: 48,
    39: 49,
    38: 50,
    # 眼周围
    51: 61,
    52: 60,
    53: 59,
    54: 58,
    55: 63,
    56: 62,
    # 瞳孔
    57: 64,
    # 鼻周
    74: 75,
    76: 77,
    78: 79,
    69: 73,
    70: 72,
    # 嘴
    80: 86,
    81: 85,
    82: 84,
    91: 87,
    90: 88,
    92: 96,
    93: 95,
    99: 97
}


def flip_landmark_idxes(lmks, lmk_format='st'):
    def flip_st_lmk_idxes(lmks):
        assert len(lmks) == 106
        for idx in range(len(lmks)):
            if idx <= 15:
                cor_idx = 16 + (16 - idx)
                lmks[[idx, cor_idx]] = lmks[[cor_idx, idx]]
            elif idx in st_flip_idx_map.keys():
                cor_idx = st_flip_idx_map[idx]
                lmks[[idx, cor_idx]] = lmks[[cor_idx, idx]]

    def flip_st_with_arc_eye_idxes(face_landmark):
        assert len(face_landmark) == 100
        for idx in range(len(face_landmark)):
            if idx <= 15:
                cor_idx = 16 + (16 - idx)
                face_landmark[[idx, cor_idx]] = face_landmark[[cor_idx, idx]]
            elif idx in st_with_arc_eye_flip_idx_map.keys():
                cor_idx = st_with_arc_eye_flip_idx_map[idx]
                face_landmark[[idx, cor_idx]] = face_landmark[[cor_idx, idx]]

    if lmk_format == 'st':
        flip_st_lmk_idxes(lmks)
    elif lmk_format == 'st_with_arc_eye':
        flip_st_with_arc_eye_idxes(lmks)
    else:
        raise IncompleteReadError


def draw_text(img,
              text,
              font_name='DejaVu Sans',
              font_size=20,
              font_color=(200, 200, 200),
              start_x=0,
              start_y=0):
    img = Image.fromarray(img)
    font_path = fm.findfont(fm.FontProperties(family=font_name))
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(img)
    draw.text((start_x, start_y), text, font=font, fill=font_color)
    img = np.array(img)
    return img


def draw_texts(img,
               texts,
               font_name='DejaVu Sans',
               font_color=(200, 200, 200),
               font_size=20,
               start_x=0,
               start_y=0):
    img = Image.fromarray(img)
    font_path = fm.findfont(fm.FontProperties(family=font_name))
    font = ImageFont.truetype(font_path, font_size)
    draw = ImageDraw.Draw(img)
    for text in texts:
        draw.text((start_x, start_y), text, font=font, fill=font_color)
        text_w, text_h = font.getsize(text)
        start_y += text_h
    img = np.array(img)
    return img


def draw_class_name_and_score(img: np.ndarray,
                              class_name: str,
                              score: float,
                              font_name='DejaVu Sans',
                              font_size=20,
                              font_color=(200, 200, 200),
                              start_x=10,
                              start_y=10):
    text = f"{class_name}: {score:.3f}"
    img = draw_text(img,
                    text,
                    font_name=font_name,
                    font_size=font_size,
                    font_color=font_color,
                    start_x=start_x,
                    start_y=start_y)
    return img


def crop_img_with_alpha_channel(img):
    alpha_img = img[:, :, 3]
    ys, xs = np.where(alpha_img != 0)
    x1 = xs.min()
    x2 = xs.max()
    y1 = ys.min()
    y2 = ys.max()
    img = img[y1:y2 + 1, x1:x2 + 1]
    return img


def cal_accuracy_precision_recall_f1(tp, fn, fp, tn):
    f1 = None
    if tp + fn + fp + tn == 0:
        accuracy = 0
    else:
        accuracy = (tp + tn) / (tp + tn + fp + fn)
    if tp + fp == 0:
        precision = 0
        f1 = 0
    else:
        precision = tp / (tp + fp)
    if tp + fn == 0:
        recall = 0
        f1 = 0
    else:
        recall = tp / (tp + fn)
    if f1 is None:
        f1 = 2 * (precision * recall) / (precision + recall)
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }
    # TODO: calculate ap


def euler_to_matrix(euler_angle):
    # pitch - x axis
    # yaw  - y axis
    # roll - z axis
    # pitch, yaw, roll = euler_angle

    pitch = -euler_angle[0] / 180 * np.pi
    yaw = -euler_angle[1] / 180 * np.pi
    roll = -euler_angle[2] / 180 * np.pi

    Rz = [[math.cos(roll), -math.sin(roll), 0],
          [math.sin(roll), math.cos(roll), 0], [0, 0, 1]]

    Ry = [[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0],
          [-math.sin(yaw), 0, math.cos(yaw)]]

    Rx = [[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)],
          [0, math.sin(pitch), math.cos(pitch)]]

    matrix = np.matmul(Rx, Ry)
    matrix = np.matmul(matrix, Rz)

    return matrix


def draw_headpose_axis(img, headpose, draw_point=None, length=50, mode='pred'):

    if draw_point is not None:
        tdx = int(draw_point[0])
        tdy = int(draw_point[1])
    else:
        tdx = int(img.shape[1] / 2)
        tdy = int(img.shape[0] / 2)

    if mode == 'pred':
        X_COLOR = (0, 0, 255)  # Red
        Y_COLOR = (0, 255, 0)  # Green
        Z_COLOR = (255, 0, 0)  # Blue
    else:
        X_COLOR = (0, 0, 139)  # DarkRed
        Y_COLOR = (0, 100, 0)  # DarkGreen
        Z_COLOR = (139, 0, 0)  # DarkBlue

    matrix = euler_to_matrix(headpose)

    Xaxis = np.array([matrix[0, 0], matrix[1, 0], matrix[2, 0]]) * length
    Yaxis = np.array([matrix[0, 1], matrix[1, 1], matrix[2, 1]]) * length
    Zaxis = np.array([matrix[0, 2], matrix[1, 2], matrix[2, 2]]) * length

    cv2.line(img, (int(tdx), int(tdy)),
             (int(Xaxis[0] + tdx), int(-Xaxis[1] + tdy)), X_COLOR, 2)
    cv2.line(img, (int(tdx), int(tdy)),
             (int(Yaxis[0] + tdx), int(-Yaxis[1] + tdy)), Y_COLOR, 2)
    cv2.line(img, (int(tdx), int(tdy)),
             (int(Zaxis[0] + tdx), int(-Zaxis[1] + tdy)), Z_COLOR, 2)

    return img


def cal_run_time(func):
    @wraps(func)
    def wrap_f(*args, **kwargs):
        start_time = time.time()
        res = func(*args, **kwargs)
        end_time = time.time()
        info = f"{func.__name__}: {end_time - start_time}"
        RLogger.log(info)
        return res

    return wrap_f