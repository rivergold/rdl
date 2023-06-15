from typing import Any
import torchvision


class ImageTransform:
    def __init__(self):
        self.img_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(512,
                                          interpolation=torchvision.transforms.
                                          InterpolationMode.BILINEAR),
            torchvision.transforms.CenterCrop(512),
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Normalize([0.5], [0.5])
        ])
        self.c_img_transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(512,
                                          interpolation=torchvision.transforms.
                                          InterpolationMode.BILINEAR),
            torchvision.transforms.CenterCrop(512),
            torchvision.transforms.ToTensor()
        ])

    def __call__(self, sample) -> Any:
        sample['rgb_img'] = self.img_transform(sample['rgb_img'])
        sample['condition_rgb_img'] = self.c_img_transform(
            sample['condition_rgb_img'])
        return sample


def build_train_transform(cfg):
    return ImageTransform()