
import torch
import torch.nn as nn

class UnsuperShortcut(nn.Module):
    def __init__(self,  **kwargs):
        super(UnsuperShortcut, self).__init__(**kwargs)

        self.stage1 = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(inplace=True))

        self.pool = nn.MaxPool2d(2, 2)

        self.stage2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(inplace=True))

        self.stage3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(inplace=True),

            nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        layer1 = self.stage1(x)  # 32 channels
        layer2 = self.stage2(self.pool(layer1))  # 64 channels
        layer3 = self.stage3(self.pool(layer2))

        h_new, w_new = layer3.shape[-2:]
        layer1_down = nn.functional.interpolate(layer1, size=[h_new, w_new])
        layer2_down = nn.functional.interpolate(layer2, size=[h_new, w_new])
        out = torch.cat([layer1_down, layer2_down, layer3], axis=1)

        return out

class UnsuperVgg(nn.Module):
    def __init__(self,  **kwargs):
        super(UnsuperVgg, self).__init__(**kwargs)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(2, 2),

            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(2, 2),

            nn.Conv2d(64, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            # nn.MaxPool2d(2, 2),

            nn.Conv2d(128, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, 3, 1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.cnn(x)
        return out

class UnsuperVggTiny(nn.Module):
    def __init__(self,  **kwargs):
        super(UnsuperVggTiny, self).__init__(**kwargs)

        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, 2, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),

            nn.Conv2d(16, 32, 3, 1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 64, 3, 2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, 1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, 1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        out = self.cnn(x)
        return out


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = torch.nn.functional.relu(out)
        return out

class ResNet(nn.Module):
    def __init__(self):
        super(ResNet, self).__init__()
        self.inchannel = 16
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, self.inchannel, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(self.inchannel),
            nn.ReLU(inplace=True),
        )
        self.ResidualBlock = ResidualBlock

        # block, channels, num_blocks, stride
        self.layer1 = self.make_layer(self.ResidualBlock, self.inchannel,  2, stride=2)
        self.layer2 = self.make_layer(self.ResidualBlock, 32, 1, stride=1)
        self.layer3 = self.make_layer(self.ResidualBlock, 64, 2, stride=2)
        self.layer4 = self.make_layer(self.ResidualBlock, 128, 1, stride=1)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.conv1(x)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        return out

if __name__ == '__main__':
    model = UnsuperVggTiny()
    print(model)
    print('end process!!!')
