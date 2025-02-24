from typing import Dict
import torch
import torch.nn as nn
import torch.nn.functional as F


class DoubleConv(nn.Sequential):
    def __init__(self, in_channels, out_channels, mid_channels=None):
        if mid_channels is None:
            mid_channels = out_channels
        super(DoubleConv, self).__init__(
            # 不改变特征层的大小
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class Down(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__(
            nn.MaxPool2d(2, stride=2),
            DoubleConv(in_channels, out_channels)
        )

class up_att(nn.Sequential):
    def __init__(self, in_channels, out_channels):
        super(up_att, self).__init__(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        if bilinear:
            # 双线性插值 填充图片，从而改变图片的高和宽
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 转置卷积
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # print(f"x1 shape: {x1.shape}")
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class Up_co(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up_co, self).__init__()
        if bilinear:
            # 双线性插值 填充图片，从而改变图片的高和宽
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            # 转置卷积
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        x1 = self.up(x1)
        # [N, C, H, W]
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        # padding_left, padding_right, padding_top, padding_bottom
        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class OutConv(nn.Sequential):
    def __init__(self, in_channels, num_classes):
        super(OutConv, self).__init__(
            nn.Conv2d(in_channels, num_classes, kernel_size=1)
        )


class Attention_block(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super(Attention_block, self).__init__()
        self.W_g = nn.Sequential(
            nn.Conv2d(F_g, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(F_l, F_int, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(F_int)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(F_int, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, g, x):
        diff_y = x.size()[2] - g.size()[2]
        diff_x = x.size()[3] - g.size()[3]

        g = F.pad(g, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        # print(f"g1 shape: {g1.shape}")
        # print(f"x1 shape: {x1.shape}")

        psi = self.relu(g1 + x1)
        psi = self.psi(psi)

        return x * psi

class UNet_att(nn.Module):
    def __init__(self,
                 in_channels: int = 1,
                 num_classes: int = 2,
                 # 是否用双线性插值替代转置卷积
                 bilinear: bool = True,
                 # 第一个卷积层的channel
                 base_c: int = 64):
        super(UNet_att, self).__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.bilinear = bilinear

        self.in_conv = DoubleConv(in_channels, base_c)
        # self.conv1 = GLSA(in_conv)
        # glsa = GLSA(input_dim=32, embed_dim=32)
        # self.conv1 = glsa()
        self.down1 = Down(base_c, base_c * 2)
        self.down2 = Down(base_c * 2, base_c * 4)
        self.down3 = Down(base_c * 4, base_c * 8)
        factor = 2 if bilinear else 1
        self.down4 = Down(base_c * 8, base_c * 16 // factor)

        # self.up_a5 = up_att(base_c * 16, base_c * 8)
        self.att5 = Attention_block(F_g=base_c * 8, F_l=base_c * 8, F_int=base_c * 4)
        # self.up1 = DoubleConv(base_c * 16, base_c * 8)
        # self.up1 = Up_co(base_c * 16, base_c * 8 // factor, bilinear)
        self.up1 = Up(base_c * 16, base_c * 8 // factor, bilinear)

        # self.up_a4 = up_att(base_c * 8, base_c * 4)
        self.att4 = Attention_block(F_g=base_c * 4, F_l=base_c * 4, F_int=base_c * 2)
        # self.up2 = DoubleConv(base_c * 8, base_c * 4)
        # self.up2 = Up_co(base_c * 8, base_c * 4 // factor, bilinear)
        self.up2 = Up(base_c * 8, base_c * 4 // factor, bilinear)

        # self.up_a3 = up_att(base_c * 4, base_c * 2)
        self.att3 = Attention_block(F_g=base_c * 2, F_l=base_c * 2, F_int=base_c)
        # self.up3 = DoubleConv(base_c * 4, base_c * 2)
        # self.up3 = Up_co(base_c * 4, base_c * 2 // factor, bilinear)
        self.up3 = Up(base_c * 4, base_c * 2 // factor, bilinear)

        # self.up_a2 = up_att(base_c * 2, base_c)
        self.att2 = Attention_block(F_g=base_c, F_l=base_c, F_int=num_classes)
        # self.up4 = DoubleConv(base_c * 2, base_c)
        # self.up4 = Up_co(base_c * 2, base_c, bilinear)
        self.up4 = Up(base_c * 2, base_c, bilinear)


        self.out_conv = OutConv(base_c, num_classes)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        x1 = self.in_conv(x)

        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x4 = self.att5(g=x5, x=x4)
        d4 = self.up1(x5, x4)

        x3 = self.att4(g=d4, x=x3)
        d3 = self.up2(d4, x3)

        x2 = self.att3(g=d3, x=x2)
        d2 = self.up3(d3, x2)

        x1 = self.att2(g=d2, x=x1)
        x = self.up4(d2, x1)

        logits = self.out_conv(x)

        return {"out": logits}


# if __name__ == '__main__':
#     unet_att = UNet_att()
    # print(unet_att)
