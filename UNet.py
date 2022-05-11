import torch
import torch.nn as nn


class ResidualBlock(nn.Module):
    def __init__(self, *args):
        super(ResidualBlock, self).__init__()
        self.model_list = nn.ModuleList([*args])

    def forward(self, x):
        input = x
        for model in self.model_list:
            x = model(x)

        return x + input


class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, activate=True):
        '''
        do not change the shape
        :param in_channel:
        :param out_channel:
        '''
        super(ConvBlock, self).__init__()
        if activate is True:
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, padding=1),
                nn.BatchNorm2d(out_channel),
                nn.ReLU(),
                ResidualBlock(
                    nn.Conv2d(out_channel, out_channel, 1),
                    nn.ReLU(),
                    nn.Conv2d(out_channel, out_channel, 3, padding=1),
                    nn.BatchNorm2d(out_channel),
                    nn.ReLU(),
                )
            )
        else:
            self.model = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, 3, padding=1),
                nn.BatchNorm2d(out_channel),
                # nn.ReLU(),
                ResidualBlock(
                    nn.Conv2d(out_channel, out_channel, 1),
                    nn.ReLU(),
                    nn.Conv2d(out_channel, out_channel, 3, padding=1),
                    nn.BatchNorm2d(out_channel),
                    # nn.ReLU(),
                )
            )

    def forward(self, x):
        return self.model(x)


class UNet(nn.Module):
    def __init__(self, down_list=[3, 64, 128, 256, 512, 1024], ):
        super(UNet, self).__init__()
        up_list = list(reversed(down_list))
        up_list.pop(-1)  # [1024, 512, 256, 128, 64]
        self.down = nn.ModuleList()
        self.up = nn.ModuleList()
        for i in range(len(down_list) - 1):
            self.down.append(ConvBlock(down_list[i], down_list[i + 1]))
            if i != len(down_list) - 2:
                self.down.append(nn.MaxPool2d(2))

        for i in range(len(up_list) - 1):
            self.up.append(nn.ConvTranspose2d(up_list[i], up_list[i + 1], 2, stride=2))
            self.up.append(ConvBlock(up_list[i], up_list[i + 1]))

        self.final = ConvBlock(up_list[-1], 1, activate=False)

    def forward(self, x):
        cache = []
        for i, model in enumerate(self.down):  # 偶数conv 奇数下采样
            x = model(x)
            if i % 2 == 0:
                cache.append(x)

        cache.pop(-1)
        cache.reverse()

        for i, model in enumerate(self.up):  # 偶数上采样 conv
            x = model(x)
            if i % 2 == 0:
                x = torch.cat([x, cache[i // 2]], dim=1)

        return self.final(x)
