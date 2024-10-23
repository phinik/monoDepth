import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import os

from feature_level_alignment import FeatureAlignment

def weights_init(m):
    # Initialize kernel weights with Gaussian distributions
    if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.ConvTranspose2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.in_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
            m.bias.data.zero_()
    elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()


def conv(in_channels, out_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,kernel_size,stride=1,padding=padding,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )


def depthwise(in_channels, kernel_size):
    padding = (kernel_size-1) // 2
    assert 2*padding == kernel_size-1, "parameters incorrect. kernel={}, padding={}".format(kernel_size, padding)
    return nn.Sequential(
          nn.Conv2d(in_channels,in_channels,kernel_size,stride=1,padding=padding,bias=False,groups=in_channels),
          nn.BatchNorm2d(in_channels),
          nn.ReLU(inplace=True),
        )


def pointwise(in_channels, out_channels):
    return nn.Sequential(
          nn.Conv2d(in_channels,out_channels,1,1,0,bias=False),
          nn.BatchNorm2d(out_channels),
          nn.ReLU(inplace=True),
        )


class MobileNet(nn.Module):
    def __init__(self, input_channels: int = 3):
        super(MobileNet, self).__init__()
        self._n_channels = input_channels

        def relu():
            return nn.ReLU6(inplace=True)

        def conv_bn(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                nn.BatchNorm2d(oup),
                relu(),
            )

        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                relu(),
                nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                nn.BatchNorm2d(oup),
                relu(),
            )

        self.model = nn.Sequential(
            conv_bn(  3,  32, 2), 
            conv_dw( 32,  64, 1),
            conv_dw( 64, 128, 2),
            conv_dw(128, 128, 1),
            conv_dw(128, 256, 2),
            conv_dw(256, 256, 1),
            conv_dw(256, 512, 2),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 512, 1),
            conv_dw(512, 1024, 2),
            conv_dw(1024, 1024, 1),
            nn.AvgPool2d(7),
        )
        self.fc = nn.Linear(1024, 1000)

    def forward(self, x):
        x = self.model(x)
        x = x.view(-1, 1024)
        x = self.fc(x)
        return x

    def load(self):
        model_filename = os.path.join('pretrained', 'model_best.pth.tar')
    
        checkpoint = torch.load(model_filename, weights_only=True)
        s_dict = {}
        #print(checkpoint['state_dict'].keys())
        for key, value in checkpoint['state_dict'].items():
            s_dict[key[7:]] = value
        self.load_state_dict(s_dict)

    def set_input_channels(self, input_channels: int = 3) -> None:
        self.model[0][0] = nn.Conv2d(input_channels, 32, 3, 2, 1, bias=False)
        #print(self.model)


class FastDepth(nn.Module):
    def __init__(self, input_channels: int = 3):
        super(FastDepth, self).__init__()
        mobilenet = MobileNet(input_channels=input_channels)
        #mobilenet.apply(weights_init)
        mobilenet.load()
        if input_channels != 3:
            mobilenet.set_input_channels(input_channels)

        self.bilinear = False
        self.n_channels = input_channels
        
        self.encoder_model = mobilenet.model[:14]

        self.feature_alignment = FeatureAlignment(1024*8*8)

        kernel_size = 5
        
        self.decoder_model = nn.Sequential(
            nn.Sequential(
                depthwise(1024, kernel_size),
                pointwise(1024, 512)),
            nn.Sequential(
                depthwise(512, kernel_size),
                pointwise(512, 256)),
            nn.Sequential(
                depthwise(256, kernel_size),
                pointwise(256, 128)),
            nn.Sequential(
                depthwise(128, kernel_size),
                pointwise(128, 64)),
            nn.Sequential(
                depthwise(64, kernel_size),
                pointwise(64, 32)),
            nn.Conv2d(32,1,1,1,0))

        #for e in self.decoder_model.children():
        #    weights_init(e)

    def forward(self, x):
        # skip connections: dec4: enc1
        # dec 3: enc2 or enc3
        # dec 2: enc4 or enc5
        
        # Encoder
        for i, layer in enumerate(self.encoder_model):
            x = layer(x)
            # Set skips
            if i==1:
                x1 = x
            elif i==3:
                x2 = x
            elif i==5:
                x3 = x

        x_flattened = torch.flatten(x, start_dim=1, end_dim=-1)

        # Decoder
        for i, layer in enumerate(self.decoder_model[:-1]):
            x = layer(x)
            x = F.interpolate(x, scale_factor=2, mode='nearest')
            if i==3:
                x = x + x1
            elif i==2:
                x = x + x2
            elif i==1:
                x = x + x3
        
        x = self.decoder_model[-1](x)

        x_feature_alignment = self.feature_alignment(x_flattened)

        return x, x_feature_alignment

    def load(self, path):
        checkpoint = torch.load(path, weights_only=True)
        s_dict = {}
        #print(checkpoint['state_dict'].items())
        for key, value in checkpoint.items():
            s_dict[key[7:]] = value
        self.load_state_dict(s_dict)


