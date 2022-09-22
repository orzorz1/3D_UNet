import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import modules.UNet_parts as up
from modules.modules_3D import SELayer_3d
import torch.optim as optim
import numpy as np
import modules.UNet_parts

class UNet_3D_SEblock(nn.Module):
    def __init__(self):
        super().__init__()
        self.left_conv_1 = up.double_conv(1, 64)
        self.pool_1 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv_2 = up.double_conv(64, 128)
        self.pool_2 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv_3 = up.double_conv(128, 256)
        self.pool_3 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv_4 = up.double_conv(256, 512)
        self.pool_4 = nn.MaxPool3d(kernel_size=2, stride=2)
        self.left_conv_5 = up.double_conv(512, 1024)

        self.deconv_1 = nn.ConvTranspose3d(1024, 512, kernel_size=2, stride=2)
        self.right_conv_1 = up.double_conv(1024, 512)
        self.deconv_2 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.right_conv_2 = up.double_conv(512, 256)
        self.deconv_3 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.right_conv_3 = up.double_conv(256, 128)
        self.deconv_4 = nn.ConvTranspose3d(128, 64, kernel_size=2, stride=2)
        self.right_conv_4 = up.double_conv(128, 64)
        self.SE = SELayer_3d(64)
        self.right_conv_5 = nn.Conv3d(64, 3, (3,3,3), padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1：进行编码过程
        feature_1 = self.left_conv_1(x)
        feature_1_pool = self.pool_1(feature_1)

        feature_2 = self.left_conv_2(feature_1_pool)
        feature_2_pool = self.pool_2(feature_2)

        feature_3 = self.left_conv_3(feature_2_pool)
        feature_3_pool = self.pool_3(feature_3)

        feature_4 = self.left_conv_4(feature_3_pool)
        feature_4_pool = self.pool_4(feature_4)

        feature_5 = self.left_conv_5(feature_4_pool)

        # 2：进行解码过程
        de_feature_1 = self.deconv_1(feature_5)
        # 特征拼接、
        temp = torch.cat((feature_4, de_feature_1), dim=1)
        de_feature_1_conv = self.right_conv_1(temp)

        de_feature_2 = self.deconv_2(de_feature_1_conv)
        temp = torch.cat((feature_3, de_feature_2), dim=1)
        de_feature_2_conv = self.right_conv_2(temp)

        de_feature_3 = self.deconv_3(de_feature_2_conv)

        temp = torch.cat((feature_2, de_feature_3), dim=1)
        de_feature_3_conv = self.right_conv_3(temp)

        de_feature_4 = self.deconv_4(de_feature_3_conv)
        temp = torch.cat((feature_1, de_feature_4), dim=1)
        de_feature_4_conv = self.right_conv_4(temp)

        SE = self.SE(de_feature_4_conv)
        out = self.right_conv_5(SE)
        # out = self.sigmoid(out)

        return out




if __name__ == "__main__":
    input = torch.rand(2, 1, 128, 128, 32)
    print("input_size:", input.size())
    model = UNet_3D_SEblock()
    ouput = model(input)
    print("output_size:", ouput.size())