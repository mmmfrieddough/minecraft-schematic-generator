import torch
import torch.nn as nn
import torch.nn.functional as F


class Generator(nn.Module):
    def __init__(self, prompt_embedding_dim, block_embedding_dim, noise_dim):
        super(Generator, self).__init__()

        self.fc = nn.Linear(prompt_embedding_dim + noise_dim, 128 * 4 * 4 * 4)
        self.deconv1 = nn.ConvTranspose3d(
            128, 64, kernel_size=4, stride=2, padding=1)
        self.deconv2 = nn.ConvTranspose3d(
            64, 48, kernel_size=4, stride=2, padding=1)
        self.deconv3 = nn.ConvTranspose3d(
            48, block_embedding_dim, kernel_size=4, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(64)
        self.batchnorm2 = nn.BatchNorm3d(48)

    def forward(self, prompt, noise):
        x = torch.cat([prompt, noise], 1)
        x = self.fc(x)
        # Reshape into a 4x4x4 volume with 128 channels
        x = x.view(-1, 128, 4, 4, 4)
        x = F.relu(self.batchnorm1(self.deconv1(x)))
        x = F.relu(self.batchnorm2(self.deconv2(x)))
        x = self.deconv3(x)
        return x

# [[B, 1536], [B, 100]] -> generator -> [B, 32, 64, 64, 64]
# [[B, 1536], [B, 32, 64, 64, 64]] -> discriminator -> [B, 1]
