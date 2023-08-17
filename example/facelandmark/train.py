import sys
from argparse import ArgumentParser
from pathlib import Path
import math, random, time, json
from copy import deepcopy
from tqdm import tqdm

import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from rconfig import Config

sys.path.append('../../')
from rdl.engine.trainer import SimpleTrainer
from rdl.engine import hook, event
from rdl.data.dataset.face_landmark import build_train_dataloader, build_val_dataloader
from rdl.data.dataset.face_landmark import (RandomRotate, RandomAxisRotate,
                                            BboxAroundCropV2, Resize, Gray,
                                            Flip, BoxErase, BrushErase,
                                            ChangeGrayBrightness,
                                            ChangeGrayContrast, Normalize,
                                            ToTensor, ToHeadPoseBins)
from rdl.data.dataset.face_landmark import eval_metric, eval_metric_merge
from rdl.model.net.face_lmk_net import build_net
from rdl.utils.common import x_color2imgs, draw_lmks, draw_headpose_axis
from rdl.utils.aug import RandomChooseKAugs
from rdl.optimization.lr_scheduler import build_lr_scheduler

cv2.setNumThreads(1)


class MseLoss(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, y_hat, gt):
        lmks_gt = gt[:, 5:-1]
        loss = F.mse_loss(y_hat, lmks_gt.to(y_hat.device))
        return loss


class WingLoss(nn.Module):
    def __init__(self, w=10, epsilon=2) -> None:
        super().__init__()
        self.w = w
        self.epsilon = epsilon
        self.C = self.w - self.w * math.log(1 + self.w / self.epsilon)

    def forward(self, y_hat, gt):
        lmks_gt = gt[:, 5:-1]
        lmks_gt = lmks_gt.to(y_hat.device)
        delta = (y_hat - lmks_gt).abs() * 256
        delta1 = delta[delta < self.w]
        delta2 = delta[delta >= self.w]
        loss1 = self.w * torch.log(1 + delta1 / self.epsilon)
        loss2 = delta2 - self.C
        loss = (loss1.sum() + loss2.sum()) / (loss1.nelement() +
                                              loss2.nelement())
        return loss


class BoxCenterMseLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, gt):
        lmks_gt = gt[:, 5:-1].detach().clone()
        batch_size = len(lmks_gt)
        lmks_gt = lmks_gt.reshape(batch_size, -1, 2)
        # c_x = (gt[:, [0]] + gt[:, [2]]) / 2
        # c_y = (gt[:, [1]] + gt[:, [3]]) / 2
        img_w = gt[:, [2]] - gt[:, [0]]
        img_h = gt[:, [3]] - gt[:, [1]]
        # # print(img_w, img_h)
        lmks_gt[:, :, 0] = lmks_gt[:, :, 0] - 0.5
        lmks_gt[:, :, 1] = lmks_gt[:, :, 1] - 0.5
        lmks_gt = lmks_gt.reshape(batch_size, -1)
        loss = F.mse_loss(y_hat, lmks_gt.to(y_hat.device))
        return loss


class SmoothL1Loss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, y_hat, gt):
        lmks_gt = gt[:, 5:-1]
        lmks_gt = lmks_gt.to(y_hat.device)
        loss = F.smooth_l1_loss(y_hat * 256, lmks_gt * 256)
        return loss


class HeadPoseLoss(nn.Module):
    def __init__(self, weight=1.0):
        super().__init__()
        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor)

        self.cal_loss = torch.nn.CrossEntropyLoss()
        if weight is not None:
            self.weight = weight
        else:
            self.weight = 1.0

    def forward(self, y_hat, gt, gt_cls):
        gt = gt.to(y_hat[0].device)
        gt_cls = gt_cls.to(y_hat[0].device)
        self.idx_tensor = self.idx_tensor.to(y_hat[0].device)
        pitch_cls_pred, yaw_cls_pred, roll_cls_pred = y_hat

        # print(gt_cls.dtype)
        # print(pitch_cls_pred.dtype)
        # print(yaw_cls_pred.dtype)

        #print(pitch_cls_pred, gt_cls[:, 0])
        loss = self.cal_loss(pitch_cls_pred, gt_cls[:, 0])
        # print('yaw_cls_pred',yaw_cls_pred)
        #print('gt_cls[:, 1]',torch.min(gt_cls[:, 1]))
        loss += self.cal_loss(yaw_cls_pred, gt_cls[:, 1])
        loss += self.cal_loss(roll_cls_pred, gt_cls[:, 2])

        pitch_pred = F.softmax(pitch_cls_pred, dim=1)
        yaw_pred = F.softmax(yaw_cls_pred, dim=1)
        roll_pred = F.softmax(roll_cls_pred, dim=1)

        pitch_pred = torch.sum(pitch_pred * self.idx_tensor, 1) * 3 - 99
        yaw_pred = torch.sum(yaw_pred * self.idx_tensor, 1) * 3 - 99
        roll_pred = torch.sum(roll_pred * self.idx_tensor, 1) * 3 - 99

        loss += self.weight * F.mse_loss(
            gt, torch.stack([pitch_pred, yaw_pred, roll_pred], axis=1))

        return loss


def build_criterion(loss_name, **kwargs):
    print(loss_name)
    if loss_name == 'mse_loss':
        return MseLoss()
    elif loss_name == 'wing_loss':
        return WingLoss(w=kwargs['w'], epsilon=kwargs['epsilon'])
    elif loss_name == 'boxcenter_mse_loss':
        return BoxCenterMseLoss()
    elif loss_name == 'smoothl1_loss':
        return SmoothL1Loss()
    elif loss_name == 'headpose_loss':
        return HeadPoseLoss(weight=kwargs['weight'])
    else:
        raise NotImplementedError


class FaceLmkTrainer(SimpleTrainer):
    def __init__(self, cfg):
        super().__init__(cfg)
        self.softmax = nn.Softmax(
            dim=1).cuda() if self.cfg.gpu.enable else nn.Softmax(dim=1)
        self.idx_tensor = [idx for idx in range(66)]
        self.idx_tensor = torch.FloatTensor(self.idx_tensor).cuda() if self.cfg.gpu.enable \
             else torch.FloatTensor(self.idx_tensor)

    def forward(self, sample):
        x = sample['rgb_imgs'].cuda(
        ) if self.cfg.gpu.enable else sample['rgb_imgs']
        # Forward
        y_hat = self.model(x)
        return y_hat

    def forward_gray(self, sample):
        x = sample['gray_imgs']
        # x = torch.cat(x,dim=0)
        # x = torch.unsqueeze(x,dim=1)
        #print('xxxxxxxxxxxxxxxxx',x.shape)
        if self.cfg.gpu.enable:
            x = x.cuda()
        # Forward
        y_hat = self.model(x)
        return y_hat

    def parse_y_hat(self, y_hat):
        """_summary_

        Args:
            img_size (tuple or list): (img_w, img_h)
            y_hat (torch.Tensor): [batch_size, num_lmks * 2]

        Returns:
            np.ndarray: [batch_size, num_lmks * 2]
        """

        lmks_pred, pitch_bin_pred, yaw_bin_pred, roll_bin_pred = y_hat
        batch_size = len(lmks_pred)
        lmks = lmks_pred.detach().cpu().numpy()
        if self.cfg.train.loss_name == 'boxcenter_mse_loss':
            batch_size = len(y_hat)
            lmks = lmks.reshape(batch_size, -1, 2)
            # c_x, c_y = img_size[0] / 2, img_size[1] / 2
            lmks[:, :, 0] = lmks[:, :, 0] + 0.5
            lmks[:, :, 1] = lmks[:, :, 1] + 0.5
            # return lmks
            # else:
            #     lmks[::2] *= img_size[0]
            #     lmks[1::2] *= img_size[1]
        else:
            #return lmks.reshape(batch_size, -1, 2)
            lmks = lmks.reshape(batch_size, -1, 2)
        ### head pose

        pitch_pred = self.softmax(pitch_bin_pred)
        yaw_pred = self.softmax(yaw_bin_pred)
        roll_pred = self.softmax(roll_bin_pred)

        pitch_pred = torch.sum(pitch_pred * self.idx_tensor, 1) * 3 - 99
        yaw_pred = torch.sum(yaw_pred * self.idx_tensor, 1) * 3 - 99
        roll_pred = torch.sum(roll_pred * self.idx_tensor, 1) * 3 - 99

        return lmks, np.stack([
            pitch_pred.detach().cpu(),
            yaw_pred.detach().cpu(),
            roll_pred.detach().cpu()
        ],
                              axis=1)

    def draw_batch_lmks(self, x, batch_lmks, batch_gt_lmks=None) -> np.ndarray:
        """_summary_

        Args:
            x (_type_): _description_
            batch_lmks (_type_): shape [N, num_lmks, 2]
            batch_gt_lmks: shape [N, num_lmks, 2]

        Returns:
            np.ndarray: _description_
        """
        rgb_imgs = x_color2imgs(x, mean=[0.], std=[1.])
        img_h, img_w = rgb_imgs.shape[1], rgb_imgs.shape[2]
        batch_lmks = batch_lmks.copy()
        batch_lmks[:, :, 0] *= img_w
        batch_lmks[:, :, 1] *= img_h
        if batch_gt_lmks is not None:
            batch_gt_lmks[:, :, 0] *= img_w
            batch_gt_lmks[:, :, 1] *= img_h
        out_rgb_imgs = []
        for idx, rgb_img in enumerate(rgb_imgs):
            rgb_img = draw_lmks(rgb_img, batch_lmks[[idx]])
            if batch_gt_lmks is not None:
                rgb_img = draw_lmks(rgb_img,
                                    batch_gt_lmks[[idx]],
                                    color=(255, 0, 0))  # draw gt
            # rgb_img = cv2.resize(rgb_img, (0, 0), fx=0.5, fy=0.5)
            out_rgb_imgs.append(rgb_img)
        out_rgb_imgs = np.stack(out_rgb_imgs, axis=0)
        return out_rgb_imgs

    def draw_batch_results(self,
                           x,
                           batch_lmks,
                           batch_headposes,
                           batch_gt_lmks=None,
                           batch_gt_headposes=None):
        """_summary_

        Args:
            x (_type_): show images, shape [N, img.shape]
            batch_headposes (_type_): shape [N, 3]
            batch_gt_headposes: shape [N, 3]
            batch_lmks (_type_): shape [N, num_lmks, 2]
            batch_gt_lmks: shape [N, num_lmks, 2]

        Returns:
            np.ndarray: array of images
        """
        rgb_imgs = x_color2imgs(x, mean=[0.], std=[1.])
        img_h, img_w = rgb_imgs.shape[1], rgb_imgs.shape[2]
        batch_headposes = batch_headposes.copy()
        batch_lmks = batch_lmks.copy()

        batch_lmks[:, :, 0] *= img_w
        batch_lmks[:, :, 1] *= img_h
        if batch_gt_lmks is not None:
            batch_gt_lmks[:, :, 0] *= img_w
            batch_gt_lmks[:, :, 1] *= img_h

        out_rgb_imgs = []
        for idx, rgb_img in enumerate(rgb_imgs):
            rgb_img = draw_headpose_axis(rgb_img, batch_headposes[idx])
            rgb_img = draw_lmks(rgb_img, batch_lmks[[idx]])
            if batch_gt_headposes is not None:
                rgb_img = draw_headpose_axis(rgb_img,
                                             batch_gt_headposes[idx],
                                             mode='gt')  # draw gt
            if batch_gt_lmks is not None:
                rgb_img = draw_lmks(rgb_img,
                                    batch_gt_lmks[[idx]],
                                    color=(255, 0, 0))  # draw gt
            out_rgb_imgs.append(rgb_img)
        out_rgb_imgs = np.stack(out_rgb_imgs, axis=0)
        return out_rgb_imgs

    def run_step(self):
        self.model.train()
        sample = next(self.train_dataloader_iter)
        batch_size = len(sample['gray_imgs'])
        # Forward
        # y_hat = self.forward(sample)
        y_hat = self.forward_gray(sample)
        lmks_hat, pitch_hat, yaw_hat, roll_hat = y_hat
        # Loss
        # print(lmks_hat.shape)
        # print(pitch_hat.shape)
        # print(yaw_hat.shape)
        # print('this is ',len(sample['head_pose_cls_gts']))
        loss = self.criterion([pitch_hat, yaw_hat, roll_hat],
                              sample['head_pose_gts'],
                              sample['head_pose_cls_gts'])

        loss_lmk = self.criterion2(lmks_hat, sample['face_gts'])
        loss_total = loss + loss_lmk
        self.update_event_storage({f'lmk_loss': ('scalar', loss_lmk.data)})
        self.update_event_storage({f'hp_loss': ('scalar', loss.data)})
        self.update_event_storage({f'total_loss': ('scalar', loss_total.data)})
        # Backward
        self.optimizer.zero_grad()
        loss_total.backward()
        self.optimizer.step()

        if self.is_last_batch:
            batch_lmks, batch_headposes = self.parse_y_hat(y_hat)
            batch_gt_headposes = sample['head_pose_gts'].detach().cpu().numpy()
            batch_gt_lmks = sample['face_gts'][:, 5:-1].reshape(
                batch_size, -1, 2).clone()
            # x = torch.cat(sample['gray_imgs'],dim=0)
            # x = torch.unsqueeze(x,dim=1)
            rgb_imgs = self.draw_batch_results(
                sample['gray_imgs'],
                batch_lmks,
                batch_headposes,
                batch_gt_lmks=batch_gt_lmks,
                batch_gt_headposes=batch_gt_headposes)

            self.update_event_storage(
                {f"train/{self.epoch}-last_batch": ('images', rgb_imgs)})
            #batch_lmks = self.parse_y_hat(y_hat)
            cur_metric_lmk, cur_metric_hp = eval_metric(
                sample['face_gts'][:, 5:-1].reshape(batch_size, -1, 2),
                torch.from_numpy(batch_lmks),
                batch_headposes,
                batch_gt_headposes,
                face_landmark_preprocess_mode='st_with_arc_eye')
            self.update_event_storage({
                f'train/last_batch-metric':
                ('scalar', cur_metric_lmk.mean().data)
            })
            self.update_event_storage(
                {f'train/last_batch-metric-yaw': ('scalar', cur_metric_hp[1])})
            self.update_event_storage({
                f'train/last_batch-metric-roll': ('scalar', cur_metric_hp[2])
            })
            # cur_metric_lmk, cur_metric_hp = eval_metric_merge(
            #     batch_lmks, batch_headposes, batch_gt_lmks, batch_gt_headposes)
            # self.update_event_storage(
            #     {f'train/last_batch-metric-pitch': ('scalar', cur_metric_lmk)})
            # self.update_event_storage({
            #     f'train/last_batch-metric-pitch': ('scalar', cur_metric_hp[0])
            # })
            # self.update_event_storage(
            #     {f'train/last_batch-metric-yaw': ('scalar', cur_metric_hp[1])})
            # self.update_event_storage({
            #     f'train/last_batch-metric-roll': ('scalar', cur_metric_hp[2])
            # })

    def _run_val_or_test(self, mode='val'):
        def run(dataloader, mode):
            metric = {'score': 0.0}
            with torch.no_grad():
                for idx, sample in tqdm(enumerate(dataloader),
                                        desc=f"{mode}-step",
                                        total=len(dataloader)):
                    batch_size = len(sample['gray_imgs'])
                    y_hat = self.forward_gray(sample)
                    batch_lmks = self.parse_y_hat(y_hat)
                    cur_metric = eval_metric(
                        sample['face_gts'][:, 5:-1].reshape(batch_size, -1, 2),
                        torch.from_numpy(batch_lmks),
                        face_landmark_preprocess_mode=self.cfg.data.
                        train_dataset_kwargs.face_landmark_preprocess_mode)
                    cur_metric = cur_metric.mean().item()
                    metric['score'] = metric['score'] + cur_metric  # FIXME:
                    # Draw images
                    if idx < 30:
                        batch_gt_lmks = sample['face_gts'][:, 5:-1].reshape(
                            batch_size, -1, 2).clone()
                        rgb_imgs = self.draw_batch_lmks(
                            sample['gray_imgs'],
                            batch_lmks,
                            batch_gt_lmks=batch_gt_lmks)
                        self.update_event_storage({
                            f"{mode}/{self.epoch}-{idx}": ('images', rgb_imgs)
                        })
                metric['score'] = metric['score'] / len(dataloader)
                self.update_event_storage({
                    f'{mode}/metric@epoch={self.epoch}-step={self.step}':
                    ('info', f"{metric}")
                })
                self.update_event_storage({
                    f'{mode}/metric@epoch={self.epoch}-step={self.step}':
                    ('text', f"{metric}")
                })

                # Best metric with score
                print(f'epoch: {self.epoch}')
                if self.best_eval_metric_map.get(mode) is None:
                    self.best_eval_metric_map[mode] = None
                if self.best_eval_metric_map[mode] is None:
                    self.best_eval_metric_map[mode] = {
                        'metric': metric,
                        'checkpoint':
                        f"epoch_{self.epoch}-step_{self.step}.pth"
                    }
                elif metric['score'] < self.best_eval_metric_map[mode][
                        'metric']['score']:
                    self.best_eval_metric_map[mode] = {
                        'metric': metric,
                        'checkpoint':
                        f"epoch_{self.epoch}-step_{self.step}.pth"
                    }

                self.update_event_storage(
                    {f"{mode}_metric/score": ('scalar', metric['score'])})
                self.update_event_storage({
                    f"best_{mode}_metric":
                    ('info',
                     f"{json.dumps(self.best_eval_metric_map[mode]['metric'], indent=4)}, {self.best_eval_metric_map[mode]['checkpoint']}"
                     )
                })
                self.update_event_storage({
                    f"best_{mode}_metric":
                    ('text',
                     f"{json.dumps(self.best_eval_metric_map[mode]['metric'], indent=4)}, {self.best_eval_metric_map[mode]['checkpoint']}"
                     )
                })
            end_time = time.time()
            self.update_event_storage({
                f"run_{mode}_time":
                ('info', f"run_{mode}_time: {end_time - start_time}")
            })

        start_time = time.time()
        self.model.eval()
        if mode == 'val':
            dataloader = self.val_dataloader
            run(dataloader, mode)
        elif mode == 'test':
            for mode, test_dataloader in self.test_dataloader_map.items():
                dataloader = test_dataloader['dataloader']
                run(dataloader, mode)

    # def run_val(self):
    #     self._run_val_or_test(mode='val')

    # def run_test(self):
    #     self._run_val_or_test(mode='test')

    def run_val(self):
        self.model.eval()
        with torch.no_grad():
            metric = np.zeros(4)
            for idx, sample in enumerate(self.val_dataloader):
                batch_size = len(sample['gray_imgs'])
                # y_hat = self.forward(sample)
                y_hat = self.forward_gray(sample)
                batch_lmks, batch_headposes = self.parse_y_hat(y_hat)
                batch_gt_headposes = sample['head_pose_gts'].detach().cpu(
                ).numpy()
                batch_gt_lmks = sample['face_gts'][:, 5:-1].reshape(
                    batch_size, -1, 2).detach().cpu().numpy()

                cur_metric_lmk, cur_metric_hp = eval_metric(
                    sample['face_gts'][:, 5:-1].reshape(batch_size, -1, 2),
                    torch.from_numpy(batch_lmks),
                    batch_headposes,
                    batch_gt_headposes,
                    face_landmark_preprocess_mode='st_with_arc_eye')
                #print('xxxxxxxxxxxxx',cur_metric_lmk.shape,cur_metric_hp.shape)
                metric += np.hstack(
                    [cur_metric_lmk.numpy().mean(), cur_metric_hp])
                if idx < 10:
                    # x = torch.cat(sample['gray_imgs'],dim=0)
                    # x = torch.unsqueeze(x,dim=1)
                    rgb_imgs = self.draw_batch_results(
                        sample['gray_imgs'],
                        batch_lmks,
                        batch_headposes,
                        batch_gt_lmks=batch_gt_lmks,
                        batch_gt_headposes=batch_gt_headposes)
                    self.update_event_storage(
                        {f"val/{self.epoch}-{idx}": ('images', rgb_imgs)})
            metric = metric / len(self.val_dataloader)
            if self.best_eval_metric_map['val'] is None:
                self.best_eval_metric_map['val'] = {
                    'metric': metric,
                    'checkpoint': f"epoch_{self.epoch}-step_{self.step}.pth"
                }
            elif (metric[0] + metric[1] + metric[2]) < (self.best_eval_metric_map['val']['metric'][0] + \
                self.best_eval_metric_map['val']['metric'][1] + self.best_eval_metric_map['val']['metric'][2]):
                self.best_eval_metric_map['val'] = {
                    'metric': metric,
                    'checkpoint': f"epoch_{self.epoch}-step_{self.step}.pth"
                }
            self.update_event_storage(
                {f'eval_lmk_metric': ('scalar', metric[0])})
            self.update_event_storage(
                {f'eval_pitch_metric': ('scalar', metric[1])})
            self.update_event_storage(
                {f'eval_yaw_metric': ('scalar', metric[2])})
            self.update_event_storage(
                {f'eval_roll_metric': ('scalar', metric[3])})
            self.update_event_storage({
                f'best_val_metric':
                ('text',
                 f"{self.best_eval_metric_map['val']['metric']}, {self.best_eval_metric_map['val']['checkpoint']}"
                 )
            })
            self.update_event_storage({
                f'best_val_metric':
                ('info',
                 f"{self.best_eval_metric_map['val']['metric']}, {self.best_eval_metric_map['val']['checkpoint']}"
                 )
            })

    def run_test(self):
        self.model.eval()
        with torch.no_grad():
            for mode, test_dataloader in self.test_dataloader_map.items():
                metric = np.zeros(4)
                dataloader = test_dataloader['dataloader']
                for idx, sample in enumerate(dataloader):
                    batch_size = len(sample['gray_imgs'])
                    # y_hat = self.forward(sample)
                    y_hat = self.forward_gray(sample)
                    batch_lmks, batch_headposes = self.parse_y_hat(y_hat)
                    batch_gt_headposes = sample['head_pose_gts'].detach().cpu(
                    ).numpy()
                    batch_gt_lmks = sample['face_gts'][:, 5:-1].reshape(
                        batch_size, -1, 2).detach().cpu().numpy()

                    cur_metric_lmk, cur_metric_hp = eval_metric(
                        sample['face_gts'][:, 5:-1].reshape(batch_size, -1, 2),
                        torch.from_numpy(batch_lmks),
                        batch_headposes,
                        batch_gt_headposes,
                        face_landmark_preprocess_mode='st_with_arc_eye')

                    metric += np.hstack(
                        [cur_metric_lmk.numpy().mean(), cur_metric_hp])
                    if idx < 10:
                        # x = torch.cat(sample['gray_imgs'],dim=0)
                        # x = torch.unsqueeze(x,dim=1)
                        rgb_imgs = self.draw_batch_results(
                            sample['gray_imgs'],
                            batch_lmks,
                            batch_headposes,
                            batch_gt_lmks=batch_gt_lmks,
                            batch_gt_headposes=batch_gt_headposes)
                        self.update_event_storage({
                            f"{mode}/{self.epoch}-{idx}": ('images', rgb_imgs)
                        })

                metric = metric / len(dataloader)

                if self.best_eval_metric_map.get(mode) is None:
                    self.best_eval_metric_map[mode] = {
                        'metric': metric,
                        'checkpoint':
                        f"epoch_{self.epoch}-step_{self.step}.pth"
                    }
                elif (metric[0] + metric[1] + metric[2]) < (self.best_eval_metric_map[mode]['metric'][0] + \
                    self.best_eval_metric_map[mode]['metric'][1] + self.best_eval_metric_map[mode]['metric'][2]):
                    self.best_eval_metric_map[mode] = {
                        'metric': metric,
                        'checkpoint':
                        f"epoch_{self.epoch}-step_{self.step}.pth"
                    }

                self.update_event_storage(
                    {f'test_{mode}_lmk_metric': ('scalar', metric[0])})
                self.update_event_storage(
                    {f'test_{mode}_pitch_metric': ('scalar', metric[1])})
                self.update_event_storage(
                    {f'test_{mode}_yaw_metric': ('scalar', metric[2])})
                self.update_event_storage(
                    {f'test_{mode}_roll_metric': ('scalar', metric[3])})
                print(self.best_eval_metric_map[mode])
                self.update_event_storage({
                    f'best_{mode}_metric':
                    ('text',
                     f"{self.best_eval_metric_map[mode]['metric']}, {self.best_eval_metric_map[mode]['checkpoint']}"
                     )
                })
                self.update_event_storage({
                    f'best_{mode}_metric':
                    ('info',
                     f"{self.best_eval_metric_map[mode]['metric']}, {self.best_eval_metric_map[mode]['checkpoint']}"
                     )
                })


def build_trainer(cfg):
    # Set random seed
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)

    # Trainer
    trainer = FaceLmkTrainer(cfg)

    # Model
    model = build_net(model_name=cfg.model.model_name,
                      lmk_num_classes=cfg.model.lmk_num_classes,
                      hp_num_classes=cfg.model.hp_num_classes,
                      pretrained=cfg.model.use_pretrained)
    trainer.set_model(model)

    # Criterion
    # criterion = build_criterion(cfg.train.loss.name, **cfg.train.loss.params)

    # trainer.set_criterion(criterion)

    criterion_headpose = build_criterion(cfg.train.loss.headpose.loss_name,
                                         **cfg.train.loss.headpose.params)
    criterion_landmark = build_criterion(cfg.train.loss.landmarks.loss_name,
                                         **cfg.train.loss.landmarks.params)
    trainer.set_criterion(criterion_headpose, criterion_landmark)

    # Optimizer
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad,
                                        model.parameters()),
                                 lr=cfg.train.lr)
    trainer.set_optimizer(optimizer)
    lr_scheduler = build_lr_scheduler(cfg.train.lr_scheduler.name,
                                      optimizer,
                                      max_epoch=cfg.train.max_epoch,
                                      **cfg.train.lr_scheduler.params)
    trainer.set_lr_scheduler(lr_scheduler)

    # Dataloader
    # Train dataloader
    box_erase_aug = BoxErase(**cfg.train.aug.BoxErase.params)
    brush_erase_aug = BrushErase(**cfg.train.aug.BrushErase.params)
    random_erase_aug = RandomChooseKAugs([box_erase_aug, brush_erase_aug], k=1)
    train_transform = torchvision.transforms.Compose([
        RandomRotate(
            p_threshold=cfg.train.aug.RandomRotate.p_threshold,
            rotate_angle_range=cfg.train.aug.RandomRotate.rotate_angle_range),
        RandomAxisRotate(
            p_threshold=cfg.train.aug.RandomAxisRotate.p_threshold,
            x_angle_range=cfg.train.aug.RandomAxisRotate.x_angle_range,
            y_angle_range=cfg.train.aug.RandomAxisRotate.y_angle_range,
            z_angle_range=cfg.train.aug.RandomAxisRotate.z_angle_range),
        # TODO: be better
        BboxAroundCropV2(
            expand_ratio_range=cfg.train.aug.BboxAroundCropV2.
            expand_ratio_range,
            move_ratio_range=cfg.train.aug.BboxAroundCropV2.move_ratio_range,
            min_expand_pixel=cfg.train.aug.BboxAroundCropV2.min_expand_pixel,
            num_lmks=cfg.train.aug.BboxAroundCropV2.num_lmks,
            enable_choice_one=True),
        Resize(size=cfg.train.img_size, mode='keep_aspect_ratio_resize'),
        # ColorDistort(brightness=0.2,
        #              contrast=[0.8, 1.2],
        #              saturation=[0.8, 1.2]),
        Flip(p_threshold=cfg.train.aug.Flip.p_threshold),
        Gray(),
        random_erase_aug,
        ChangeGrayBrightness(
            p_threshold=cfg.train.aug.ChangeGrayBrightness.p_threshold,
            brightness_range=cfg.train.aug.ChangeGrayBrightness.
            brightness_range),
        ChangeGrayContrast(
            p_threshold=cfg.train.aug.ChangeGrayContrast.p_threshold,
            contrast_range=cfg.train.aug.ChangeGrayContrast.contrast_range),
        ToHeadPoseBins(
            max_angle=cfg.train.aug.ToHeadPoseBins.max_angle_region,
            angle_per_bin=cfg.train.aug.ToHeadPoseBins.angle_per_bin),
        Normalize(),
        ToTensor()
    ])
    train_dataloader = build_train_dataloader(
        dataset_name=cfg.data.train_dataset_name,
        transform=train_transform,
        **cfg.data.train_dataset_kwargs)
    trainer.set_train_dataloder(train_dataloader)
    print(
        '======================================================================================'
    )
    if cfg.train.enable_val:
        val_transform = torchvision.transforms.Compose([
            BboxAroundCropV2(expand_ratio_range=(1.3, 1.3),
                             move_ratio_range=(0, 0),
                             min_expand_pixel=0,
                             num_lmks=cfg.train.aug.BboxAroundCropV2.num_lmks,
                             enable_choice_one=False),
            Resize(size=cfg.train.img_size, mode='keep_aspect_ratio_resize'),
            Gray(),
            ToHeadPoseBins(
                max_angle=cfg.train.aug.ToHeadPoseBins.max_angle_region,
                angle_per_bin=cfg.train.aug.ToHeadPoseBins.angle_per_bin),
            Normalize(),
            ToTensor()
        ])
        val_dataloader = build_val_dataloader(
            dataset_name=cfg.data.val_dataset_name,
            transform=val_transform,
            **cfg.data.val_dataset_kwargs)
        trainer.set_val_dataloader(val_dataloader)
    if cfg.train.enable_test:
        test_transform = torchvision.transforms.Compose([
            BboxAroundCropV2(expand_ratio_range=(1.3, 1.3),
                             move_ratio_range=(0, 0),
                             min_expand_pixel=0,
                             num_lmks=cfg.train.aug.BboxAroundCropV2.num_lmks,
                             enable_choice_one=False),
            Resize(size=cfg.train.img_size, mode='keep_aspect_ratio_resize'),
            Gray(),
            ToHeadPoseBins(
                max_angle=cfg.train.aug.ToHeadPoseBins.max_angle_region,
                angle_per_bin=cfg.train.aug.ToHeadPoseBins.angle_per_bin),
            Normalize(),
            ToTensor()
        ])
        test_dataloaders = []
        for cfg_test_dataset in cfg.data.test_datasets:
            test_dataloader = build_val_dataloader(
                dataset_name=cfg_test_dataset.test_dataset_name,
                transform=test_transform,
                # num_fixed_samples=1000,
                **cfg_test_dataset.test_dataset_kwargs)
            test_dataloaders.append(test_dataloader)
        trainer.set_test_dataloaders(test_dataloaders)

    # Hooks
    hooks = [
        # Load pretrained checkpoint
        hook.LoadPretrainedCheckpointHook(cfg.train.pretrained_checkpoint),
        # Resume checkpoint hook
        hook.ResumeCheckpointHook(cfg.train.resume_from),
        # Lrschedule hook
        hook.LRSchedulerHook(),
        # Log hook
        hook.LogHook(base_out_dir=cfg.work_dir,
                     exp_name=cfg.exp_name,
                     step_period=cfg.vis.log_step_period),
        # Tensorboard hook
        hook.TensorBoardWriteHook(
            event.TensorboardWriter(base_out_dir=cfg.work_dir,
                                    exp_name=cfg.exp_name,
                                    enable_draw_img=False),
            step_period=cfg.vis.tensorboard_step_period,
        ),
        # Visualization hook
        hook.VisualizationHook(base_out_dir=cfg.work_dir,
                               exp_name=cfg.exp_name,
                               step_period=cfg.vis.visualzation_step_period,
                               color_mode='rgb'),
        # Checkpoint save hook
        hook.SaveCheckpointHook(
            base_out_dir=cfg.work_dir,
            exp_name=cfg.exp_name,
            epoch_period=cfg.vis.save_checkpoint_epoch_period),
    ]
    trainer.register_hooks(hooks)

    return trainer


if __name__ == '__main__':
    opt_parser = ArgumentParser()
    opt_parser.add_argument('--cfg_path',
                            type=str,
                            default='./config/face_lmk-gw_and_st-base.json')
    opt_parser.add_argument('--work_dir', type=str, default=None)
    opt_parser.add_argument('--flag_dir', type=str, default=None)

    opt = opt_parser.parse_args()
    cfg_path = Path(opt.cfg_path)
    cfg = Config.from_file(cfg_path)
    # Update work_dir
    if opt.work_dir is not None:
        cfg.work_dir = opt.work_dir
    print(cfg)
    trainer = build_trainer(cfg)
    trainer.train(start_step=0,
                  max_epoch=cfg.train.max_epoch,
                  enable_val=cfg.train.enable_val,
                  enable_test=cfg.train.enable_test)
