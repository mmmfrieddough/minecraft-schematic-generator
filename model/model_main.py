import torch
import torch.nn as nn
import torch.nn.functional as F


class MinecraftStructureGenerator(nn.Module):
    def __init__(self, input_embedding_dim, output_embedding_dim, num_tokens):
        super(MinecraftStructureGenerator, self).__init__()
        self.block_embeddings = nn.Embedding(num_tokens, output_embedding_dim)
        self.output_embedding_dim = output_embedding_dim

        # Embedding transformation
        self.fc = nn.Linear(input_embedding_dim, 512)
        self.fc_dropout = nn.Dropout(p=0.1)
        self.reshaper = nn.Unflatten(1, (8, 8, 8))  # Reshaping to 3D tensor

        # 3D convolution layers
        self.conv1 = nn.Conv3d(8, 16, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(16, 32, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)  # Upsampling
        self.output_conv = nn.Conv3d(
            64, output_embedding_dim, kernel_size=3, padding=1)  # Output layer

    def forward(self, x):
        # Initial embedding transformation
        transformed_embedding = F.relu(self.fc(x))
        reshaped_embedding = self.reshaper(transformed_embedding)

        # First convolutional layer with skip connection
        x = F.relu(self.conv1(reshaped_embedding))
        x = self.upsample(x)
        skip_connection = F.interpolate(transformed_embedding, size=x.shape[2:], mode='trilinear', align_corners=True)
        x = x + skip_connection  # Element-wise addition

        # Second convolutional layer with skip connection
        x_res = x  # Store the current state of x for the skip connection
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        skip_connection = F.interpolate(x_res, size=x.shape[2:], mode='trilinear', align_corners=True)
        x = x + skip_connection  # Element-wise addition

        # Third convolutional layer with skip connection
        x_res = x  # Store the current state of x for the skip connection
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        skip_connection = F.interpolate(x_res, size=x.shape[2:], mode='trilinear', align_corners=True)
        x = x + skip_connection  # Element-wise addition

        x = self.output_conv(x)

        # Translate embeddings to tokens
        block_tokens = self.embeddings_to_tokens(x)
        return x, block_tokens

    def embeddings_to_tokens(self, embeddings):
        # Flatten the embeddings if they are in a 3D structure (e.g., from a 3D convolutional layer)
        flat_embeddings = embeddings.view(-1, self.output_embedding_dim)

        # Compute distances from each embedding to each block embedding
        distances = torch.cdist(flat_embeddings, self.block_embeddings.weight)

        # Find the index of the nearest embedding
        nearest_indices = torch.argmin(distances, dim=1)

        # Reshape back to the original shape, if necessary
        # For example, if your output structure is a 64x64x64 grid, reshape as below:
        reshaped_indices = nearest_indices.view(
            embeddings.shape[0], embeddings.shape[2], embeddings.shape[3], embeddings.shape[4])

        return reshaped_indices


class CustomLoss(nn.Module):
    def __init__(self, block_embeddings):
        super(CustomLoss, self).__init__()
        self.block_embeddings = block_embeddings

    def forward(self, predicted_embeddings, predicted_tokens, target_tokens):
        # Embedding loss - measure the distance between predicted and target embeddings
        target_embeddings = self.block_embeddings(target_tokens)
        embedding_loss = F.mse_loss(predicted_embeddings, target_embeddings)

        # Token loss - standard classification loss (e.g., CrossEntropy)
        token_loss = F.cross_entropy(predicted_tokens, target_tokens)

        # Combine losses, possibly with different weights
        return embedding_loss + token_loss


# Example usage
num_tokens = 10000
input_embedding_dim = 1536
output_embedding_dim = 64

model = MinecraftStructureGenerator(
    input_embedding_dim, output_embedding_dim, num_tokens)
loss_fn = CustomLoss(model.block_embeddings)

# During training
# ...
predicted_embeddings, predicted_tokens = model(input)
loss = loss_fn(predicted_embeddings, predicted_tokens, target_tokens)
