import torch
from torch import nn
from torchvision.ops import DeformConv2d

class MDDF(nn.Module):
    def __init__(self, in_channels):
        super(MDDF, self).__init__()
        # 分支1: 1x1卷积
        self.branch1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, padding=0)

        # 分支2: 1x1卷积 + 动态可变形卷积
        self.branch2_1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, padding=0)
        self.branch2_offset = nn.Conv2d(in_channels // 4, 18, kernel_size=3, stride=1, padding=1)  # 3x3卷积，偏移量通道数为18
        self.branch2_ddc = DeformConv2d(in_channels // 4, in_channels // 4, kernel_size=3, stride=1, padding=1)

        # 分支3: 1x1卷积 + 动态可变形卷积
        self.branch3_1 = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, padding=0)
        self.branch3_offset = nn.Conv2d(in_channels // 4, 50, kernel_size=5, stride=1, padding=2)  # 5x5卷积，偏移量通道数为50
        self.branch3_ddc = DeformConv2d(in_channels // 4, in_channels // 4, kernel_size=5, stride=1, padding=2)

        # 分支4: 3x3最大池化 + 1x1卷积
        self.branch4_pool = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.branch4_conv = nn.Conv2d(in_channels, in_channels // 4, kernel_size=1, padding=0)

        # 融合分支输出
        total_out_channels = (in_channels // 4) * 4  # 四个分支的输出通道总和
        self.concat_conv = nn.Conv2d(total_out_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 分支1: 普通卷积
        branch1_out = self.branch1(x)

        # 分支2: 动态可变形卷积
        branch2_intermediate = self.branch2_1(x)
        branch2_offset = self.branch2_offset(branch2_intermediate)
        branch2_out = self.branch2_ddc(branch2_intermediate, branch2_offset)

        # 分支3: 动态可变形卷积
        branch3_intermediate = self.branch3_1(x)
        branch3_offset = self.branch3_offset(branch3_intermediate)
        branch3_out = self.branch3_ddc(branch3_intermediate, branch3_offset)

        # 分支4: 最大池化 + 普通卷积
        branch4_out = self.branch4_conv(self.branch4_pool(x))

        # 连接所有分支的输出
        concatenated = torch.cat([branch1_out, branch2_out, branch3_out, branch4_out], dim=1)

        concatenated = concatenated * x

        # 应用1x1卷积和激活函数
        fused = self.concat_conv(concatenated)
        attention = self.sigmoid(fused)

        # 剩余连接
        output = x + attention
        return output


if __name__ == '__main__':
    mddf = MDDF(512)
    input_tensor = torch.randn(1, 512, 60, 60)  # batch_size=1, channels=512, height=60, width=60
    output = mddf(input_tensor)
    print(output.shape)  # 应输出 torch.Size([1, 512, 60, 60])
