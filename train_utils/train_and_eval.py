from enum import Enum

import cv2
import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim
from torchvision import transforms

import train_utils.distributed_utils as utils

from .dice_coefficient_loss import dice_loss, build_target

# 计算模型预测与真实标签之间的损失值
# inputs: 是一个字典，包含了模型的输出结果。在这里假设有两个键值对，分别是 'out' 和 'aux'，分别对应主要输出和辅助输出。
# target: 是真实的标签值，用于计算损失。
# loss_weight: 是损失函数的权重参数，用于加权不同类别的损失值。
# num_classes: 是类别的数量，默认为 2 类（二分类问题）。
# dice: 是一个布尔值，指示是否计算 Dice 损失。
# ignore_index: 是忽略的标签值，通常用于忽略边缘或填充像素。
def criterion(inputs, target, loss_weight=None, num_classes: int = 2, dice: bool = True, ignore_index: int = -100):
    losses = {}
    for name, x in inputs.items():
        # 忽略target中值为255的像素，255的像素是目标边缘或者padding填充
        # 计算交叉熵损失
        loss = nn.functional.cross_entropy(x, target, ignore_index=ignore_index, weight=loss_weight)
        if dice is True:
            # 调用 build_target 函数构建 Dice 损失的目标 dice_target
            dice_target = build_target(target, num_classes, ignore_index)
            # 引入dice_loss
            # 衡量预测结果的重叠度
            loss += dice_loss(x, dice_target, multiclass=True, ignore_index=ignore_index)
        losses[name] = loss

    # 如果 losses 字典中只有一个键值对，直接返回该损失值 'out'。
    # 否则，返回主要输出 'out' 的损失值加上辅助输出 'aux' 的损失值乘以 0.5 的加权和。
    if len(losses) == 1:
        return losses['out']

    return losses['out'] + 0.5 * losses['aux']

# 评估模型在验证或测试集上性能的函数
# model: 被评估的模型。
# data_loader: 数据加载器，用于加载验证或测试数据。
# device: 计算设备，例如 CPU 或 GPU。
# num_classes: 类别数量，用于构建混淆矩阵和计算 Dice 系数。
def evaluate(model, data_loader, device, num_classes):
    # 将模型设置为评估模式，这会关闭 dropout 和 batch normalization 层的影响
    model.eval()
    # 创建一个混淆矩阵对象，用于计算分类任务中的混淆矩阵。
    confmat = utils.ConfusionMatrix(num_classes)
    # dice的指标
    # 用于衡量预测结果的重叠度
    dice = utils.DiceCoefficient(num_classes=num_classes, ignore_index=255)

    # 创建一个日志记录器对象，用于记录评估过程中的指标
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # input_size = 2  # 根据你的 input_conditions 定 义
    # num_outputs = 5  # 输出的类别数
    # weight_generator = WeightGenerator(input_size, num_outputs).to(device)
    # 使用 torch.no_grad() 上下文管理器
    # 确保在评估阶段不进行梯度计算，以节省内存和加速计算
    with torch.no_grad():
        # 遍历数据加载器中的每个批次数据，并使用 metric_logger 记录指定间隔的日志。
        for image, target in metric_logger.log_every(data_loader, 100, header):
            image, target = image.to(device), target.to(device)
            output = model(image)['out']

            # 更新混淆矩阵，传入展平的目标标签和模型输出的预测标签
            confmat.update(target.flatten(), output.argmax(1).flatten())
            # 更新 Dice 系数计算器，传入模型输出和目标标签
            dice.update(output, target)

        # 如果在多 GPU 训练中使用了多个进程，则对混淆矩阵和 Dice 系数进行汇总
        confmat.reduce_from_all_processes()
        dice.reduce_from_all_processes()

    # 返回最终的混淆矩阵对象和计算得到的 Dice 系数的值
    return confmat, dice.value.item()

# model: 训练的模型。
# optimizer: 优化器，用于更新模型参数。
# data_loader: 数据加载器，用于加载训练数据。
# device: 计算设备，例如 CPU 或 GPU。
# epoch: 当前训练的epoch数。
# num_classes: 类别数量，用于设置损失函数的参数。
# lr_scheduler: 学习率调度器，用于动态调整学习率。
# print_freq: 打印日志的频率，即每处理多少个批次打印一次日志。
# scaler: 混合精度训练时使用的混合精度缩放器，用于减少内存占用和提高性能。
def train_one_epoch(model, optimizer, data_loader, device, epoch, num_classes,
                    lr_scheduler, print_freq=10, scaler=None):

    # 启用 dropout 和 batch normalization 层
    model.train()
    # 创建一个日志记录器对象，用于记录训练过程中的指标
    metric_logger = utils.MetricLogger(delimiter="  ")
    # 添加一个用于记录学习率的指标
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    if num_classes == 2:
        # 设置cross_entropy中背景和前景的loss权重(根据自己的数据集进行设置)
        loss_weight = torch.as_tensor([1.0, 2.0], device=device)
    else:
        loss_weight = None

    # 遍历数据加载器中的每个批次数据，并使用 metric_logger 记录指定间隔的日志
    for image, target in metric_logger.log_every(data_loader, print_freq, header):
        image, target = image.to(device), target.to(device)

        # 如果使用混合精度训练，则启用自动混合精度缩放
        # with torch.cuda.amp.autocast(enabled=scaler is not None):
        with torch.amp.autocast('cuda', enabled=scaler is not None):
            output = model(image)
            loss = criterion(output, target, loss_weight, num_classes=num_classes, ignore_index=255)

        # 清除优化器中模型参数的梯度
        optimizer.zero_grad()
        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        # 根据设定的调度策略更新学习率
        lr_scheduler.step()

        lr = optimizer.param_groups[0]["lr"]
        # 更新日志记录器中的损失和学习率指标
        metric_logger.update(loss=loss.item(), lr=lr)

    # 计算均值损失
    mean_loss = metric_logger.meters["loss"].global_avg

    # 返回全局平均损失和当前学习率
    # return metric_logger.meters["loss"].global_avg, lr
    return mean_loss, lr


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        """
        根据step数返回一个学习率倍率因子，
        注意在训练开始之前，pytorch会提前调用一次lr_scheduler.step()方法
        """
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            # warmup过程中lr倍率因子从warmup_factor -> 1
            return warmup_factor * (1 - alpha) + alpha
        else:
            # warmup后lr倍率因子从1 -> 0
            # 参考deeplab_v2: Learning rate policy
            return (1 - (x - warmup_epochs * num_step) / ((epochs - warmup_epochs) * num_step)) ** 0.9

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)