## MDAF_Net

## 项目简介

- 本项目旨在基于改进的U-Net神经网络对视网膜眼底图像进行血管分割。

## 安装指南

提供项目的安装步骤，包括依赖的库和环境设置。

### 环境依赖

- Python 版本：3.8

- 运行环境

| 类别       | 参数                                 |
| ---------- | ------------------------------------ |
| 操作系统   | Ubuntu 20.04                         |
| 处理器     | 15 vCPU Intel® Xeon®  Platinum 8474C |
| 显卡       | NVIDIA RTX 4090D  (24GB) × 1         |
| 内存       | 80GB                                 |
| Tensorflow | 2.7.0                                |
| CUDA       | 11.3                                 |
| pytorch    | 1.11.0                               |

### 安装步骤

1.克隆项目：

```bash
git clone https://github.com/moonlight-sky/MDAF_Net.git
```

2.数据集定义：

数据集有`DRIVE`和`CHASEDB1`

在`train.py`中修改`data-path`的路径为数据集的存放位置

3.修改数据预处理：

根据选择的数据集指定数据预处理

`train.py`中

```python
train_dataset = DriveDataset()	#对应DRIVE数据集
train_dataset = DriveDataset_cha()		#对应CHASEDB1数据集
```



### 运行

```py
python train.py
```



### 验证

修改`prediction.py`

```py
weights_path	#模型权重保存路径
img_path		#需要分割的眼底图像保存位置
roi_mask_path	#眼底图像的分割掩码
manual_path		#标签
```



```py
python prediction.py
```

