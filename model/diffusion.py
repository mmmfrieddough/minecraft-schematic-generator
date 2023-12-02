import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock3D, self).__init__()
        self.conv1 = nn.Conv3d(in_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(out_channels)
        self.conv2 = nn.Conv3d(out_channels, out_channels,
                               kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        return x


class Encoder3D(nn.Module):
    def __init__(self, block_embedding_dim):
        super(Encoder3D, self).__init__()
        self.conv1 = ConvBlock3D(block_embedding_dim, 128)
        self.pool1 = nn.MaxPool3d(2)
        self.conv2 = ConvBlock3D(128, 256)
        self.pool2 = nn.MaxPool3d(2)
        self.conv3 = ConvBlock3D(256, 512)
        self.pool3 = nn.MaxPool3d(2)

    def forward(self, x):
        x = self.conv1(x)
        x1 = self.pool1(x)
        x2 = self.conv2(x1)
        x3 = self.pool2(x2)
        x4 = self.conv3(x3)
        return x4, [x, x2, x4]


class Decoder3D(nn.Module):
    def __init__(self, block_embedding_dim):
        super(Decoder3D, self).__init__()
        self.upconv3 = nn.ConvTranspose3d(512, 256, kernel_size=2, stride=2)
        self.conv3 = ConvBlock3D(512, 256)
        self.upconv2 = nn.ConvTranspose3d(256, 128, kernel_size=2, stride=2)
        self.conv2 = ConvBlock3D(256, 128)
        self.upconv1 = nn.ConvTranspose3d(
            128, block_embedding_dim, kernel_size=2, stride=2)
        self.conv1 = ConvBlock3D(160, block_embedding_dim)

    def forward(self, x, enc_features):
        x = self.upconv3(x)
        x = torch.cat([x, enc_features[2]], dim=1)
        x = self.conv3(x)
        x = self.upconv2(x)
        x = torch.cat([x, enc_features[1]], dim=1)
        x = self.conv2(x)
        x = self.upconv1(x)
        x = torch.cat([x, enc_features[0]], dim=1)
        x = self.conv1(x)
        return x


class DiffusionMinecraftStructureGenerator(nn.Module):
    def __init__(self, block_embedding_dim):
        super(DiffusionMinecraftStructureGenerator, self).__init__()
        self.encoder = Encoder3D(block_embedding_dim)
        self.bottleneck = ConvBlock3D(512, 1024)
        self.decoder = Decoder3D(block_embedding_dim)
        self.final_conv = nn.Conv3d(
            block_embedding_dim, block_embedding_dim, kernel_size=1)

    def forward(self, x, text_embedding):
        x, enc_features = self.encoder(x)
        x = self.bottleneck(x)
        x = x + text_embedding.view(text_embedding.size(0), -1, 1, 1, 1)
        x = self.decoder(x, enc_features)
        x = self.final_conv(x)
        return x
