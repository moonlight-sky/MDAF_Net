import torch
import torch.nn as nn
import torch.nn.functional as F

class ContextBlock(nn.Module):
    def __init__(self,
                 inplanes,
                 ratio=0.25,
                 pooling_type='att',
                 fusion_types=('channel_mul',)):
        super(ContextBlock, self).__init__()
        assert pooling_type in ['avg', 'att']
        assert isinstance(fusion_types, (list, tuple))
        valid_fusion_types = ['channel_add', 'channel_mul']
        assert all([f in valid_fusion_types for f in fusion_types])
        assert len(fusion_types) > 0, 'At least one fusion type should be used'

        self.inplanes = inplanes
        self.ratio = ratio
        self.planes = int(inplanes * ratio)
        self.pooling_type = pooling_type
        self.fusion_types = fusion_types

        if pooling_type == 'att':
            self.conv_mask = nn.Conv2d(inplanes, 1, kernel_size=1)
            self.softmax = nn.Softmax(dim=2)
        else:
            self.avg_pool = nn.AdaptiveAvgPool2d(1)

        # 优化 channel_add 和 channel_mul 操作
        if 'channel_add' in fusion_types:
            self.channel_add_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_add_conv = None

        if 'channel_mul' in fusion_types:
            self.channel_mul_conv = nn.Sequential(
                nn.Conv2d(self.inplanes, self.planes, kernel_size=1),
                nn.LayerNorm([self.planes, 1, 1]),
                nn.ReLU(inplace=True),
                nn.Conv2d(self.planes, self.inplanes, kernel_size=1)
            )
        else:
            self.channel_mul_conv = None

    def spatial_pool(self, x):
        batch, channel, height, width = x.size()
        if self.pooling_type == 'att':
            input_x = x.view(batch, channel, height * width)
            input_x = input_x.unsqueeze(1)
            context_mask = self.conv_mask(x)
            context_mask = context_mask.view(batch, 1, height * width)
            context_mask = self.softmax(context_mask)
            context_mask = context_mask.unsqueeze(-1)
            context = torch.matmul(input_x, context_mask)
            context = context.view(batch, channel, 1, 1)
        else:
            context = self.avg_pool(x)
        return context

    def forward(self, x):
        context = self.spatial_pool(x)

        out = x
        if self.channel_mul_conv is not None:
            channel_mul_term = torch.sigmoid(self.channel_mul_conv(context))
            out = out * channel_mul_term
        if self.channel_add_conv is not None:
            channel_add_term = self.channel_add_conv(context)
            out = out + channel_add_term

        return out

class HLAF(nn.Module):
    def __init__(self, in_channels_low, in_channels_high):
        super(HLAF, self).__init__()

        # 通道注意力模块（针对低级特征 XL）
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)  # H×W×C -> 1×1×C
        self.fc1 = nn.Linear(in_channels_low, in_channels_low // 4)  # 压缩通道数
        self.fc2 = nn.Linear(in_channels_low // 4, in_channels_low)  # 恢复通道数
        self.sigmoid = nn.Sigmoid()

        # 空间注意力模块替换为优化版 ContextBlock
        self.context_block = ContextBlock(in_channels_high, ratio=0.25, pooling_type='att')

        # 多尺度融合卷积
        self.fusion_conv = nn.Conv2d(in_channels_low + in_channels_high, in_channels_low, kernel_size=3, padding=1)

    def forward(self, x_low, x_high):
        """
        x_low: 低级特征 (H × W × C)
        x_high: 高级特征 (H'/2 × W'/2 × C*)
        """

        # ========== 通道注意力机制 ==========
        b, c_low, h_low, w_low = x_low.size()
        channel_avg = self.global_avg_pool(x_low).view(b, c_low)  # Global Average Pooling
        channel_fc1 = F.relu(self.fc1(channel_avg))  # 第一层全连接 + ReLU
        channel_fc2 = self.sigmoid(self.fc2(channel_fc1)).view(b, c_low, 1, 1)  # 第二层全连接 + Sigmoid
        y_low = x_low * channel_fc2  # XL · Y'L

        # ========== 空间注意力机制 ==========
        x_high_upsampled = F.interpolate(x_high, size=(h_low, w_low), mode='bilinear', align_corners=True)  # 上采样恢复尺寸
        y_high = self.context_block(x_high_upsampled)  # 使用优化后的 ContextBlock 处理高级特征

        # ========== 多尺度融合 ==========
        concatenated_features = torch.cat([y_low, y_high], dim=1)  # 拼接低级和高级特征
        fused_features = self.fusion_conv(concatenated_features)  # 3×3卷积融合
        return fused_features


if __name__ == '__main__':
    # 定义模型
    hlaf = HLAF(512, 512)

    # 测试输入
    xl = torch.randn(1, 512, 60, 60)  # 低级特征
    xh = torch.randn(1, 512, 30, 30)  # 高级特征
    output = hlaf(xl, xh)

    print("Output shape:", output.shape)  # 应输出: [1, 512, 60, 60]
