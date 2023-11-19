import torch.nn as nn
import torch.nn.functional as F


class MinecraftStructureGenerator(nn.Module):
    def __init__(self, input_embedding_dim, output_classes, output_size):
        super(MinecraftStructureGenerator, self).__init__()
        self.output_classes = output_classes
        self.output_size = output_size

        # Define the architecture
        self.fc1 = nn.Linear(input_embedding_dim, 128 * 8 * 8 * 8)
        # 1D batch normalization for the output of fc1
        self.bn1 = nn.BatchNorm1d(128 * 8 * 8 * 8)
        self.conv1 = nn.Conv3d(128, 64, kernel_size=3, padding=1)
        # Batch normalization for conv1
        self.bn2 = nn.BatchNorm3d(64)
        self.conv2 = nn.Conv3d(64, 32, kernel_size=5, padding=3)
        # Batch normalization for conv2
        self.bn3 = nn.BatchNorm3d(32)
        self.conv3 = nn.Conv3d(32, output_classes, kernel_size=7, padding=5)

    def forward(self, x):
        # Fully connected layer to expand the input
        x = self.fc1(x)
        # Apply 1D batch normalization before activation
        x = F.relu(self.bn1(x))
        # Reshape to a 5D tensor for 3D convolution
        x = x.view(-1, 128, 8, 8, 8)

        # Convolutional layers with batch normalization and activation
        # Apply 3D batch normalization before activation
        x = F.relu(self.bn2(self.conv1(x)))
        # Apply 3D batch normalization before activation
        x = F.relu(self.bn3(self.conv2(x)))
        x = self.conv3(x)  # No activation, as this is the output layer

        # Upsample to the desired output size
        x = F.interpolate(x, size=self.output_size,
                          mode='trilinear', align_corners=True)
        return x
