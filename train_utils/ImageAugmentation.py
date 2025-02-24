import random
import torch
import torch.nn as nn
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms


# 定义数据增强操作
class DataAugmentation:
    def __init__(self, operations):
        self.operations = operations  # 增强操作列表

    def apply_operation(self, image, operation, magnitude):
        # 如果输入的 image 是 Tensor，将其转换为 PIL.Image
        if isinstance(image, torch.Tensor):
            image = transforms.ToPILImage()(image)
        # 实现每个增强操作的细节
        if operation == 'rotate':
            return image.rotate(magnitude * 3)
        elif operation == 'shearX':
            return image.transform(image.size, Image.AFFINE, (1, magnitude / 10, 0, 0, 1, 0))
        elif operation == 'brightness':
            enhancer = ImageEnhance.Brightness(image)
            return enhancer.enhance(1 + magnitude / 10)
        # 添加其他操作逻辑...
        return image

    def apply_policy(self, image, policy):
        for operation, prob, magnitude in policy:
            if random.random() < prob:
                image = self.apply_operation(image, operation, magnitude)
        return image

# RNN控制器，用于生成增强策略
class RNNController(nn.Module):
    def __init__(self, operation_space, num_sub_policies=5):
        super(RNNController, self).__init__()
        self.operation_space = operation_space
        self.num_sub_policies = num_sub_policies
        self.rnn = nn.LSTM(input_size=len(operation_space), hidden_size=128, num_layers=2)
        self.fc_op = nn.Linear(128, len(operation_space))  # 选择操作
        self.fc_prob = nn.Linear(128, 11)  # 操作概率
        self.fc_magnitude = nn.Linear(128, 10)  # 操作幅度

    def forward(self, x):
        h0, c0 = torch.zeros(2, x.size(0), 128), torch.zeros(2, x.size(0), 128)
        out, _ = self.rnn(x, (h0, c0))
        ops = self.fc_op(out)
        probs = self.fc_prob(out)
        magnitudes = self.fc_magnitude(out)
        return ops, probs, magnitudes

    def sample_policy(self):
        # 使用softmax采样子策略
        ops, probs, mags = [], [], []
        for _ in range(self.num_sub_policies * 2):  # 每个子策略包含2个操作
            op = np.random.choice(self.operation_space)
            prob = np.random.choice(np.linspace(0, 1, 11))
            mag = np.random.choice(np.linspace(0, 1, 10))
            ops.append(op)
            probs.append(prob)
            mags.append(mag)
        return list(zip(ops, probs, mags))