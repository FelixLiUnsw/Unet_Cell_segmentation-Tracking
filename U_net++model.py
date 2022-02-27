import torch
import torch.nn as nn



# Network model
class NU_net_block(nn.Module):
    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.conv1 = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )
        self.conv2 = nn.Sequential(
                        nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                        nn.BatchNorm2d(out_channels),
                        nn.ReLU(inplace=True)
                    )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        return x

class NU_net(nn.Module):
    def __init__(self, in_channels=1, n=16) -> None:
        super().__init__()
        self.concat = lambda fs: torch.cat(fs, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        ns = [n * 2 ** i for i in range(5)]

        self.to_00 = NU_net_block(in_channels, ns[0])
        self.to_10 = NU_net_block(ns[0], ns[1])
        self.to_20 = NU_net_block(ns[1], ns[2])
        self.to_30 = NU_net_block(ns[2], ns[3])
        self.to_40 = NU_net_block(ns[3], ns[4])

        self.to_01 = NU_net_block(ns[0] + ns[1], ns[0])
        self.to_11 = NU_net_block(ns[1] + ns[2], ns[1])
        self.to_21 = NU_net_block(ns[2] + ns[3], ns[2])
        self.to_31 = NU_net_block(ns[3] + ns[4], ns[3])

        self.to_02 = NU_net_block(ns[0] + ns[0] + ns[1], ns[0])
        self.to_12 = NU_net_block(ns[1] + ns[1] + ns[2], ns[1])
        self.to_22 = NU_net_block(ns[2] + ns[2] + ns[3], ns[2])

        self.to_03 = NU_net_block(ns[0] + ns[0] + ns[0] + ns[1], ns[0])
        self.to_13 = NU_net_block(ns[1] + ns[1] + ns[1] + ns[2], ns[1])

        self.to_04 = NU_net_block(ns[0] + ns[0] + ns[0] + ns[0] + ns[1], ns[0])

        self.final = nn.Conv2d(ns[0], 1, kernel_size=1)

    def forward(self, x):
        X00 = self.to_00(x)
        X10 = self.to_10(self.pool(X00))
        X20 = self.to_20(self.pool(X10))
        X30 = self.to_30(self.pool(X20))
        X40 = self.to_40(self.pool(X30))

        X01 = self.to_01(self.concat([X00, self.up(X10)]))
        X11 = self.to_11(self.concat([X10, self.up(X20)]))
        X21 = self.to_21(self.concat([X20, self.up(X30)]))
        X31 = self.to_31(self.concat([X30, self.up(X40)]))
        
        X02 = self.to_02(self.concat([X00, X01, self.up(X11)]))
        X12 = self.to_12(self.concat([X10, X11, self.up(X21)]))
        X22 = self.to_22(self.concat([X20, X21, self.up(X31)]))
        
        X03 = self.to_03(self.concat([X00, X01, X02, self.up(X12)]))
        X13 = self.to_13(self.concat([X10, X11, X12, self.up(X22)]))
        
        X04 = self.to_04(self.concat([X00, X01, X02, X03, self.up(X13)]))

        out = self.final(X04)

        return out
