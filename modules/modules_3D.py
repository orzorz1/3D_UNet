import torch.nn as nn
import torch
import numpy as np

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd
    def forward(self, x):
        return self.lambd(x)

class ResidualBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch):
        self.in_ch = in_ch
        self.out_ch = out_ch
        super(ResidualBlock3D, self).__init__()
        self.res = nn.Sequential(
            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, (1, 1, 1)),

            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, in_ch, (3, 3, 3), padding=1),

            nn.BatchNorm3d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_ch, out_ch, (1, 1, 1)),
        )
        self.conv = nn.Conv3d(self.in_ch,self.out_ch,(1, 1, 1))

    def forward(self, input):
        x = self.res(input)
        if self.out_ch == self.in_ch:
            x = x + input
        else:
            input = self.conv(input)
            x = x + input
        return x


class AttentionBlock3D(nn.Module):
    def __init__(self, in_ch, out_ch, depth):
        self.in_ch = in_ch
        self.out_ch = out_ch
        self.depth = depth
        super(AttentionBlock3D, self).__init__()
        # First Residual Block
        self.res1 = ResidualBlock3D(in_ch,in_ch)

        # Trunc Branch
        self.res2 = ResidualBlock3D(in_ch,in_ch)
        self.res3 = ResidualBlock3D(in_ch,in_ch)

        # Soft Mask Branch
        ## encoder
        self.pool1 = nn.MaxPool3d(2)
        self.res4 = ResidualBlock3D(in_ch,in_ch)

        self.net1 = nn.Sequential()
        for i in range(depth):
            self.net1.add_module('res1_{index}'.format(index=i),ResidualBlock3D(in_ch,in_ch))
            self.net1.add_module('pool{index}'.format(index=i),nn.MaxPool3d(2))
            self.net1.add_module('res2_{index}'.format(index=i),ResidualBlock3D(in_ch,in_ch))
        ##decoder
        self.net2 = nn.Sequential()
        for i in range(depth):
            self.net2.add_module('res1_{index}'.format(index=i),ResidualBlock3D(in_ch,in_ch))
            self.net2.add_module('up{index}'.format(index=i),nn.Upsample(scale_factor=2))

        #last upsampling
        self.res5 = ResidualBlock3D(in_ch,in_ch)
        self.up = nn.Upsample(scale_factor=2)

        self.conv1 = nn.Conv3d(in_ch, in_ch, (1,1,1))
        self.conv2 = nn.Conv3d(in_ch, in_ch, (1,1,1))
        self.activation = nn.Sigmoid()

        self.res6 = ResidualBlock3D(in_ch, out_ch)

        self.plus1 = LambdaLayer(lambda x: x + 1)

    def forward(self, input):
        input = self.res1(input)
        # Trunc Branch
        output_trunk = input
        output_trunk = self.res2(output_trunk)
        output_trunk = self.res3(output_trunk)

        # Soft Mask Branch
        ##encoder
        output_soft_mask = self.pool1(input)
        output_soft_mask = self.res4(output_soft_mask)

        skip_connections = []
        for i in range(self.depth):
            ## skip connections
            output_skip_connection = self.net1[(i-1)*3](output_soft_mask)
            skip_connections.append(output_skip_connection)
            ## down sampling
            output_soft_mask = self.net1[(i-1)*3+1](output_soft_mask)
            output_soft_mask = self.net1[(i-1)*3+2](output_soft_mask)

        ##decoder
        skip_connections = list(reversed(skip_connections))
        for i in range(self.depth):
            ## upsampling
            output_soft_mask = self.net2[(i-1)*2](output_soft_mask)
            output_soft_mask = self.net2[(i-1)*2+1](output_soft_mask)

            ## skip connections
            output_soft_mask = output_soft_mask.detach().add(skip_connections[i])


        ### last upsampling
        output_soft_mask = self.res5(output_soft_mask)
        output_soft_mask = self.up(output_soft_mask)

        ## Output
        output_soft_mask = self.conv1(output_soft_mask)
        output_soft_mask = self.conv2(output_soft_mask)
        output_soft_mask = self.activation(output_soft_mask)
        # Attention: (1 + output_soft_mask) * output_trunk
        output = self.plus1(output_soft_mask)
        output = output * output_trunk
        # Last Residual Block
        output = self.res6(output)

        return output


class SELayer_3d(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer_3d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.linear1 = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, X_input):
        batch_size, channel, _, _, _ = X_input.size()

        y = self.avg_pool(X_input)
        y = y.view(batch_size, channel)

        # 第1个线性层（含激活函数），即公式中的W1，其维度是[channel, channer/16], 其中16是默认的
        y = self.linear1(y)

        # 第2个线性层（含激活函数），即公式中的W2，其维度是[channel/16, channer], 其中16是默认的
        y = self.linear2(y)
        y = y.view(batch_size, channel, 1, 1, 1)

        return X_input * y.expand_as(X_input)

