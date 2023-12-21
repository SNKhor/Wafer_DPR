import torch
import torch.nn as nn
import torch.nn.functional as F

class UNetWithDeformConv(nn.Module):
    def __init__(self, in_channels, out_channels, init_normal_stddev=0.01, **kwargs):
        super(UNetWithDeformConv, self).__init__()

        # Encoder
        self.encoder_conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, padding=1)
        self.encoder_conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.encoder_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Bottleneck
        self.bottleneck_conv1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bottleneck_conv2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)

        # Decoder
        self.decoder_upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.decoder_conv1 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.decoder_conv2 = nn.Conv2d(128, out_channels, kernel_size=3, padding=1)

        # Deformable convolution
        self.deform_conv1 = ConvOffset2D_train(128, init_normal_stddev=init_normal_stddev, **kwargs)
        self.deform_conv2 = ConvOffset2D_train(256, init_normal_stddev=init_normal_stddev, **kwargs)

    def forward(self, x):
        # Encoder
        x1 = F.relu(self.encoder_conv1(x))
        x2 = F.relu(self.encoder_conv2(x1))
        x_enc = self.encoder_pool(x2)

        # Bottleneck
        x_bottleneck = F.relu(self.bottleneck_conv1(x_enc))
        x_bottleneck = F.relu(self.bottleneck_conv2(x_bottleneck))

        # Decoder
        x_dec = self.decoder_upsample(x_bottleneck)
        x_dec = torch.cat([x_dec, x2], dim=1)
        x_dec = F.relu(self.decoder_conv1(x_dec))
        x_dec = F.relu(self.decoder_conv2(x_dec))

        # Deformable convolution
        x_dec = self.deform_conv1(x_dec)
        x_dec = self.deform_conv2(x_dec)

        return x_dec

# Example usage:
in_channels = 1  # Modify based on your input channels
out_channels = 1  # Modify based on your output channels
net = UNetWithDeformConv(in_channels, out_channels)
