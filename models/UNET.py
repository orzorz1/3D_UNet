import torch
from torch import nn

class block_down(nn.Module):

    def __init__(self, inp_channel, out_channel):
        super(block_down, self).__init__()
        self.conv1 = nn.Conv2d(inp_channel, out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU6(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class block_up(nn.Module):

    def __init__(self, inp_channel, out_channel, y):
        super(block_up, self).__init__()
        self.up = nn.ConvTranspose2d(inp_channel, out_channel, 2, stride=2)
        self.conv1 = nn.Conv2d(inp_channel, out_channel, 3, padding=1)
        self.conv2 = nn.Conv2d(out_channel, out_channel, 3, padding=1)
        self.bn = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU6(inplace=True)
        self.y = y

    def forward(self, x):
        x = self.up(x)
        x = torch.cat([x, self.y], dim=1)
        x = self.conv1(x)
        x = self.bn(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class U_net(nn.Module):

    def __init__(self, out_channel):
        super(U_net, self).__init__()
        self.out = nn.Conv2d(64, out_channel, 1)
        self.maxpool = nn.MaxPool2d(2)

    def forward(self, x):
        block1 = block_down(3, 64)
        x1_use = block1(x)
        x1 = self.maxpool(x1_use)
        block2 = block_down(64, 128)
        x2_use = block2(x1)
        x2 = self.maxpool(x2_use)
        block3 = block_down(128, 256)
        x3_use = block3(x2)
        x3 = self.maxpool(x3_use)
        block4 = block_down(256, 512)
        x4_use = block4(x3)
        x4 = self.maxpool(x4_use)
        block5 = block_down(512, 1024)
        x5 = block5(x4)
        block6 = block_up(1024, 512, x4_use)
        x6 = block6(x5)
        block7 = block_up(512, 256, x3_use)
        x7 = block7(x6)
        block8 = block_up(256, 128, x2_use)
        x8 = block8(x7)
        block9 = block_up(128, 64, x1_use)
        x9 = block9(x8)
        x10 = self.out(x9)
        out = nn.Softmax2d()(x10)
        return out


if __name__ == "__main__":
    test_input = torch.rand(8, 3, 512, 512)
    print("input_size:", test_input.size())
    model = U_net(out_channel=3)
    ouput = model(test_input)
    print("output_size:", ouput.size())
