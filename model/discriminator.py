import torch
import torch.nn as nn
import torch.nn.functional as F


class Discriminator(nn.Module):
    def __init__(self, prompt_embedding_dim, block_embedding_dim):
        super(Discriminator, self).__init__()

        self.conv1 = nn.Conv3d(block_embedding_dim, 48,
                               kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv3d(48, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=4, stride=2, padding=1)
        self.batchnorm1 = nn.BatchNorm3d(48)
        self.batchnorm2 = nn.BatchNorm3d(64)
        self.fc1 = nn.Linear(128 * 4 * 4 * 4 + prompt_embedding_dim, 1024)
        self.fc2 = nn.Linear(1024, 1)

    def forward(self, prompt, structure):
        x = F.leaky_relu(self.batchnorm1(self.conv1(structure)), 0.2)
        x = F.leaky_relu(self.batchnorm2(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3(x), 0.2)
        x = x.view(-1, 128 * 4 * 4 * 4)
        x = torch.cat([x, prompt], 1)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = torch.sigmoid(self.fc2(x))
        return x

# [[B, 1536], [B, 100]] -> generator -> [B, 32, 64, 64, 64]
# [[B, 1536], [B, 32, 64, 64, 64]] -> discriminator -> [B, 1]
