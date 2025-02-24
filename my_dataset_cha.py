import os

import cv2
from PIL import Image, ImageOps
import numpy as np
from matplotlib import pyplot as plt
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

data_root = './unet/CHASEDB1'

class DriveDataset_cha(Dataset):
    def __init__(self, root: str, train: bool, transforms=None):
        super(DriveDataset_cha, self).__init__()
        self.flag = "training" if train else "test"
        # data_root = os.path.join(root, "DRIVE", self.flag)
        data_root = os.path.join(root, "CHASEDB1")
        # 定义数据集的根目录

        assert os.path.exists(data_root), f"path '{data_root}' does not exists."
        self.transforms = transforms

        # 获取原始图像名称
        self.img_names = [i for i in os.listdir(os.path.join(data_root, "images"))]
        # 原始图像路径
        self.img_list = [os.path.join(data_root, "images", i) for i in self.img_names]

        # 标签路径
        self.manual = [os.path.join(data_root, "1st_label", i.split(".")[0] + "_1stHO.png")
                  for i in self.img_names]

        # 感兴趣区域
        self.roi_mask = [os.path.join(data_root, "mask", i.split(".")[0] + ".png")
                    for i in self.img_names]

        # check files
        for i in self.manual:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # check files
        for i in self.roi_mask:
            if os.path.exists(i) is False:
                raise FileNotFoundError(f"file {i} does not exists.")

        # 首先对图像和标签进行拆分
        train_images, test_images = train_test_split(self.img_list, test_size=0.2, random_state=42)

        # 然后对标签 (manual) 进行拆分
        train_masks, test_masks = train_test_split(self.manual, test_size=0.2, random_state=42)

        # 最后对 ROI 掩码进行拆分
        train_roi_masks, test_roi_masks = train_test_split(self.roi_mask, test_size=0.2, random_state=42)

        if self.flag == "training":
            self.img_list = train_images
            self.manual = train_masks
            self.roi_mask = train_roi_masks
        else:
            self.img_list = test_images
            self.manual = test_masks
            self.roi_mask = test_roi_masks

    def __getitem__(self, idx):
        img = Image.open(self.img_list[idx]).convert('RGB')
        # 转成灰度图像
        manual = Image.open(self.manual[idx]).convert('L')
        # 白色 255（前景）
        # 黑色 0 （背景）
        # 在此分割任务中，只有一个前景，即可/255将其转换为1 ==> 前景的像素值1
        manual = np.array(manual) / 255
        # 感兴趣区域的图片 255
        roi_mask = Image.open(self.roi_mask[idx]).convert('L')
        # 用255减去感兴趣区域后 白色区域就变为0 不感兴趣区域变成255
        roi_mask = 255 - np.array(roi_mask)
        # 最终目的，将不感兴趣区域的像素值设置成255，计算损失时，即可将255的区域忽略
        # np.clip() 设置上线限
        # 最后，对于前景 1 背景 0 不感兴趣 255
        mask = np.clip(manual + roi_mask, a_min=0, a_max=255)

        # 这里转回PIL的原因是，transforms中是对PIL数据进行处理
        mask = Image.fromarray(mask)

        if self.transforms is not None:
            img, mask = self.transforms(img, mask)

        return img, mask

    def __len__(self):
        return len(self.img_list)

    # 将images金额targets打包成 batch
    @staticmethod
    def collate_fn(batch):
        images, targets = list(zip(*batch))
        batched_imgs = cat_list(images, fill_value=0)
        batched_targets = cat_list(targets, fill_value=255)
        return batched_imgs, batched_targets


def cat_list(images, fill_value=0):
    max_size = tuple(max(s) for s in zip(*[img.shape for img in images]))
    batch_shape = (len(images),) + max_size
    batched_imgs = images[0].new(*batch_shape).fill_(fill_value)
    for img, pad_img in zip(images, batched_imgs):
        pad_img[..., :img.shape[-2], :img.shape[-1]].copy_(img)
    return batched_imgs

