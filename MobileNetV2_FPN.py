import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv_bn(inp, oup, stride):  # convolution + batchnorm
    return nn.Sequential(
        nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


def conv_1x1_bn(inp, oup):  # convolution 1x1 + batchnorm
    return nn.Sequential(
        nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
        nn.BatchNorm2d(oup),
        nn.ReLU6(inplace=True)
    )


class InvertedResidual(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        self.use_res_connect = self.stride == 1 and inp == oup

        self.conv = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(in_channels=inp, out_channels=inp * expand_ratio, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # depthwise convolution via groups
            nn.Conv2d(inp * expand_ratio, inp * expand_ratio, 3, stride, 1, groups=inp * expand_ratio, bias=False),
            nn.BatchNorm2d(inp * expand_ratio),
            nn.ReLU6(inplace=True),
            # pointwise linear convolutionc
            nn.Conv2d(inp * expand_ratio, oup, 1, 1, 0, bias=False),
            nn.BatchNorm2d(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2_FPN(nn.Module):  # nn.Module is base class for nets
    def __init__(self, n_class=1000, input_size=224, width_mult=1.):
        super(MobileNetV2_FPN, self).__init__()

        # building first layer
        assert input_size % 32 == 0
        self.input_channel = int(32 * width_mult)
        self.width_mult = width_mult
        self.first_layer = conv_bn(3, self.input_channel, 2)

        # Inverted residual blocks
        self.interverted_residual_setting = [
            {'expansion_factor': 1, 'width_factor': 16, 'n': 1, 'stride': 1},
            {'expansion_factor': 6, 'width_factor': 24, 'n': 2, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 32, 'n': 3, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 64, 'n': 4, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 96, 'n': 3, 'stride': 1},
            {'expansion_factor': 6, 'width_factor': 160, 'n': 3, 'stride': 2},
            {'expansion_factor': 6, 'width_factor': 320, 'n': 1, 'stride': 1},
        ]
        self.inverted_residual_blocks = nn.ModuleList([self._make_interverted_residual_block(**setting)
                                                       for setting in self.interverted_residual_setting])

        # Top down layers
        self.toplayer = nn.Conv2d(320, 256, kernel_size=1, stride=1, padding=0)

        # Lateral layers
        # only needed if resulution decreases (stride > 1)
        self.lateral_layers = nn.ModuleList([nn.Conv2d(setting['width_factor'], 256, kernel_size=1, stride=1, padding=0)
                                             for setting in self.interverted_residual_setting if setting['stride'] > 1])

        # Smooth layers
        self.smooth_layers = nn.ModuleList([nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
                                            for layer in self.lateral_layers])

        self.last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280

        self._initialize_weights()

    def _make_interverted_residual_block(self, expansion_factor, width_factor, n, stride):
        interverted_residual_block = []
        output_channel = int(width_factor * self.width_mult)
        for i in range(n):
            if i == 0:
                stride = 1
            interverted_residual_block.append(InvertedResidual(self.input_channel, output_channel, stride, expansion_factor))
            self.input_channel = output_channel
        return nn.Sequential(*interverted_residual_block)

    def _upsample_add(self, x, y):
        '''Upsample and add two feature maps.
        Args:
          x: (Variable) top feature map to be upsampled.
          y: (Variable) lateral feature map.
        Returns:
          (Variable) added feature map.
        Note in PyTorch, when input size is odd, the upsampled feature map
        with `F.upsample(..., scale_factor=2, mode='nearest')`
        maybe not equal to the lateral feature map size.
        e.g.
        original input size: [N,_,15,15] ->
        conv2d feature map size: [N,_,8,8] ->
        upsampled feature map size: [N,_,16,16]
        So we choose bilinear upsample which supports arbitrary output sizes.
        '''
        _, _, H, W = y.size()
        return F.upsample(x, size=(H, W), mode='bilinear', align_corners=False) + y

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        # Bottom-up
        c1 = F.relu(self.bn1(self.conv1(x)))
        c1 = F.max_pool2d(c1, kernel_size=3, stride=2, padding=1)
        c2 = self.layer1(c1)
        c3 = self.layer2(c2)
        c4 = self.layer3(c3)
        c5 = self.layer4(c4)
        p6 = self.conv6(c5)
        p7 = self.conv7(F.relu(p6))
        p8 = self.conv8(F.relu(p7))
        p9 = self.conv9(F.relu(p8))
        # Top-down
        p5 = self.toplayer(c5)
        p4 = self._upsample_add(p5, self.latlayer1(c4))
        p3 = self._upsample_add(p4, self.latlayer2(c3))
        p4 = self.smooth1(p4)
        p3 = self.smooth2(p3)
        return p3, p4, p5, p6, p7, p8, p9


def test():
    net = MobileNetV2_FPN()
    fms = net(torch.randn(1, 3, 512, 512))
    for fm in fms:
        print(fm.size())


test()

# net = MobileNetV2_FPN()
# print(net)
