import torch
import torch.nn as nn
import torch.nn.functional as F


class MinecraftStructureGenerator(nn.Module):
    def __init__(self, prompt_embedding_dim, block_embedding_dim, num_classes):
        super(MinecraftStructureGenerator, self).__init__()
        self.prompt_embedding_dim = prompt_embedding_dim
        self.block_embedding_dim = block_embedding_dim
        self.num_classes = num_classes

        # Define the model
        self.channels_layer_1 = 512
        self.dim_layer_1 = 8
        self.size_layer_1 = self.channels_layer_1 * self.dim_layer_1 ** 3
        self.channels_layer_2 = self.channels_layer_1 // 2
        self.channels_layer_3 = self.channels_layer_2 // 2
        self.channels_layer_4 = self.channels_layer_3 // 2

        # Fully connected layer to expand the input
        self.fc = nn.Linear(prompt_embedding_dim, self.size_layer_1)
        self.bn_fc = nn.BatchNorm1d(self.size_layer_1)

        # 3D convolutional layers with batch normalization and skip connections
        self.up1 = nn.ConvTranspose3d(in_channels=self.channels_layer_1,
                                      out_channels=self.channels_layer_2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.bn_up1 = nn.BatchNorm3d(self.channels_layer_2)
        self.conv1 = nn.Conv3d(in_channels=self.channels_layer_2,
                               out_channels=self.channels_layer_2, kernel_size=3, stride=1, padding=1)
        self.bn_conv1 = nn.BatchNorm3d(self.channels_layer_2)

        self.up2 = nn.ConvTranspose3d(in_channels=self.channels_layer_2,
                                      out_channels=self.channels_layer_3, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn_up2 = nn.BatchNorm3d(self.channels_layer_3)
        self.conv2 = nn.Conv3d(in_channels=self.channels_layer_3,
                               out_channels=self.channels_layer_3, kernel_size=3, stride=1, padding=1)
        self.bn_conv2 = nn.BatchNorm3d(self.channels_layer_3)

        self.up3 = nn.ConvTranspose3d(in_channels=self.channels_layer_3,
                                      out_channels=self.channels_layer_4, kernel_size=5, stride=2, padding=2, output_padding=1)
        self.bn_up3 = nn.BatchNorm3d(self.channels_layer_4)
        self.conv3 = nn.Conv3d(in_channels=self.channels_layer_4,
                               out_channels=self.channels_layer_4, kernel_size=7, stride=1, padding=3)
        self.bn_conv3 = nn.BatchNorm3d(self.channels_layer_4)

        self.conv_final = nn.Conv3d(
            in_channels=self.channels_layer_4, out_channels=block_embedding_dim, kernel_size=1)

        self.out = nn.Linear(in_features=block_embedding_dim,
                             out_features=num_classes)

        # Initialize the weights
        self.init_weights()

    def forward(self, x):
        # Expand the input
        x = self.fc(x)
        x = self.bn_fc(x)
        x = F.relu(x)
        x = x.view(-1, self.channels_layer_1, self.dim_layer_1,
                   self.dim_layer_1, self.dim_layer_1)
        # print(x.shape)

        # First layer
        x = self.up1(x)
        x = self.bn_up1(x)
        x = F.relu(x)
        # x = self.conv1(x)
        # x = self.bn_conv1(x)
        # x = F.relu(x)
        # print(x.shape)

        # Second layer
        x = self.up2(x)
        x = self.bn_up2(x)
        x = F.relu(x)
        # x = self.conv2(x)
        # x = self.bn_conv2(x)
        # x = F.relu(x)
        # print(x.shape)

        # Third layer
        x = self.up3(x)
        x = self.bn_up3(x)
        x = F.relu(x)
        x = self.conv3(x)
        x = self.bn_conv3(x)
        x = F.relu(x)
        # print(x.shape)

        # Upsample to the desired output size
        x = self.conv_final(x)

        original_shape = x.shape
        x = x.view(-1, self.block_embedding_dim)
        x = self.out(x)
        x = x.view(original_shape[0], self.num_classes,
                   original_shape[2], original_shape[3], original_shape[4])

        return x

    def init_weights(self):
        """
        Initialize the weights of the model.
        """
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.ConvTranspose3d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
