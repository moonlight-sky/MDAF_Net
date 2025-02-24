import os
import time

import torch
from torchvision import transforms
import numpy as np
from PIL import Image

from src import UNet
from src import UNet_att
from src import UnetPlusPlus
from src import MDAF_Net


def time_synchronized():
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    return time.time()


def main():
    os.chdir("./unet")
    classes = 1  # exclude background
    weights_path = "./save_weights/best_model.pth"
    img_path = "./DRIVE/test/images/01_test.tif"
    roi_mask_path = "./DRIVE/test/mask/01_test_mask.gif"
    manual_path = "./DRIVE/test/1st_manual/01_manual1.gif"
    assert os.path.exists(weights_path), f"weights {weights_path} not found."
    assert os.path.exists(img_path), f"image {img_path} not found."
    assert os.path.exists(roi_mask_path), f"image {roi_mask_path} not found."
    assert os.path.exists(manual_path), f"image {manual_path} not found."

    mean = (0.709, 0.381, 0.224)
    std = (0.127, 0.079, 0.043)

    # get devices
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("using {} device.".format(device))

    # create model
    # model = UNet(in_channels=3, num_classes=classes+1, base_c=64)
    # model = UNet_att(in_channels=3, num_classes=classes+1, base_c=64)
    # model = UnetPlusPlus(num_classes=classes+1)
    model = MDAF_Net(in_channels=3, num_classes=classes+1, base_c=64)

    # load weights
    model.load_state_dict(torch.load(weights_path, map_location='cpu')['model'], strict=False)
    model.to(device)

    # load roi mask
    roi_img = Image.open(roi_mask_path).convert('L')
    roi_img = np.array(roi_img)

    # load image
    original_img = Image.open(img_path).convert('RGB')

    # load manual
    manual_img = Image.open(manual_path).convert('L')
    manual = np.array(manual_img)
    # from pil image to tensor and normalize
    data_transform = transforms.Compose([transforms.ToTensor(),
                                         # transforms.RandomCrop(480),
                                         transforms.Normalize(mean=mean, std=std)])
    img = data_transform(original_img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)

    model.eval()  # 进入验证模式
    with torch.no_grad():
        # init model
        img_height, img_width = img.shape[-2:]
        init_img = torch.zeros((1, 3, img_height, img_width), device=device)
        model(init_img)

        t_start = time_synchronized()
        output = model(img.to(device))
        t_end = time_synchronized()
        print("inference time: {}".format(t_end - t_start))

        prediction = output['out'].argmax(1).squeeze(0)
        prediction = prediction.to("cpu").numpy().astype(np.uint8)
        # 将前景对应的像素值改成255(白色)
        prediction[prediction == 1] = 255

        # 将不敢兴趣的区域像素设置成0(黑色)
        prediction[roi_img == 0] = 0
        mask = Image.fromarray(prediction)
        mask.save("test_result.png")

    # 计算 IoU
    intersection = np.logical_and(prediction > 0, manual > 0).sum()
    union = np.logical_or(prediction > 0, manual > 0).sum()
    iou = intersection / union if union != 0 else 0

    # 计算 Dice 系数
    dice = 2 * intersection / (prediction.sum() + manual.sum()) if (prediction.sum() + manual.sum()) != 0 else 0

    # 计算准确率
    accuracy = (prediction == manual).sum() / manual.size

    # 打印结果
    print(f"IoU: {iou:.4f}")
    print(f"Dice: {dice:.4f}")
    print(f"Accuracy: {accuracy:.4f}")

if __name__ == '__main__':
    main()
