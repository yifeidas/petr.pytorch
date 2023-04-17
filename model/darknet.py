from torch import nn
import math
from model.network_blocks import BaseConv, CSPLayer, DWConv, Focus, ResLayer, SPPBottleneck

def ceil(c, stride=8):
    return math.ceil(c / stride) * stride

class CSPDarknet(nn.Module):
    def __init__(
        self,
        dep_mul,
        wid_mul,
        out_features=("dark3", "dark4", "dark5"),
        depthwise=False,
        stem='conv',
        act="silu",
    ):
        super().__init__()
        assert out_features, "please provide output features of Darknet"
        self.out_features = out_features
        if 'dark6' in self.out_features:
            base_out_channels = [256, 512, 768, 1024]
        else:
            base_out_channels = [256, 512, 1024]
        self.out_channels = [ceil(x * wid_mul) for x in base_out_channels]
        Conv = DWConv if depthwise else BaseConv

        base_channels = int(wid_mul * 64)  # 64
        base_depth = max(round(dep_mul * 3), 1)  # 3

        # stem
        # self.stem = Focus(3, base_channels, ksize=3, act=act)
        # self.stem = Conv(3, base_channels, 3, 2, act=act)
        if stem == 'conv':
            self.stem = nn.Sequential(
                Conv(3, base_channels//2, 3, 1, act=act),
                Conv(base_channels//2, base_channels, 3, 2, act=act)
            )
        else:
            self.stem = Focus(3, base_channels, ksize=3, act=act)

        # dark2
        self.dark2 = nn.Sequential(
            Conv(base_channels, base_channels * 2, 3, 2, act=act),
            CSPLayer(
                base_channels * 2,
                base_channels * 2,
                n=base_depth,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark3
        self.dark3 = nn.Sequential(
            Conv(base_channels * 2, base_channels * 4, 3, 2, act=act),
            CSPLayer(
                base_channels * 4,
                base_channels * 4,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        # dark4
        self.dark4 = nn.Sequential(
            Conv(base_channels * 4, base_channels * 8, 3, 2, act=act),
            CSPLayer(
                base_channels * 8,
                base_channels * 8,
                n=base_depth * 3,
                depthwise=depthwise,
                act=act,
            ),
        )

        if 'dark6' in self.out_features:
            # dark5
            self.dark5 = nn.Sequential(
                Conv(base_channels * 8, base_channels * 12, 3, 2, act=act), # channel base 768
                CSPLayer(
                    base_channels * 12,
                    base_channels * 12,
                    n=base_depth,
                    shortcut=False,
                    depthwise=depthwise,
                    act=act,
                ),
            )

            # dark6
            self.dark6 = nn.Sequential(
                Conv(base_channels * 12, base_channels * 16, 3, 2, act=act), # channel base 1024
                SPPBottleneck(base_channels * 16, base_channels * 16, activation=act),
                CSPLayer(
                    base_channels * 16,
                    base_channels * 16,
                    n=base_depth,
                    shortcut=False,
                    depthwise=depthwise,
                    act=act,
                ),
            )
        else:
            # dark5
            self.dark5 = nn.Sequential(
                Conv(base_channels * 8, base_channels * 16, 3, 2, act=act),
                SPPBottleneck(base_channels * 16, base_channels * 16, activation=act), # channel base 1024
                CSPLayer(
                    base_channels * 16,
                    base_channels * 16,
                    n=base_depth,
                    shortcut=False,
                    depthwise=depthwise,
                    act=act,
                ),
            )

    def forward(self, x):
        outputs = {}
        x = self.stem(x)
        outputs["stem"] = x
        x = self.dark2(x)
        outputs["dark2"] = x
        x = self.dark3(x)
        outputs["dark3"] = x
        x = self.dark4(x)
        outputs["dark4"] = x
        x = self.dark5(x)
        outputs["dark5"] = x
        if 'dark6' in self.out_features:
            x = self.dark6(x)
            outputs["dark6"] = x
        return {k: v for k, v in outputs.items() if k in self.out_features}
