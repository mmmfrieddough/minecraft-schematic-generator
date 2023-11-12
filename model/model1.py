import torch
import torch.nn as nn
import torch.nn.functional as F


class MinecraftStructureGenerator(nn.Module):
    def __init__(self, num_tokens, embedding_dim, output_dim):
        super(MinecraftStructureGenerator, self).__init__()
        # Embedding for block tokens
        self.block_embedding = nn.Embedding(num_tokens, embedding_dim)
        # Increase the first dimension of the reshaper to match the embedding dimension
        self.reshaper = nn.Unflatten(1, (embedding_dim, 8, 8))
        # Convolutional layers and upsample remain the same
        # Adjusted the in_channels
        self.conv1 = nn.Conv3d(embedding_dim, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv3d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='trilinear', align_corners=True)
        self.output_conv = nn.Conv3d(128, output_dim, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.block_embedding(x)  # Convert block tokens to embeddings
        x = self.reshaper(x)
        x = F.relu(self.conv1(x))
        x = self.upsample(x)
        x = F.relu(self.conv2(x))
        x = self.upsample(x)
        x = F.relu(self.conv3(x))
        x = self.upsample(x)
        # Raw voxel values are now predicted embeddings
        x = self.output_conv(x)
        return x

# Custom loss function that considers the similarity between embeddings


def custom_embedding_loss(predicted_embeddings, target_embeddings, target_tokens):
    # Use cosine similarity as a distance metric
    cos_sim = F.cosine_similarity(
        predicted_embeddings, target_embeddings, dim=1)
    # Convert cosine similarity to a positive loss value
    loss = 1 - cos_sim
    # Optionally, use the target tokens to mask out or scale the loss for certain predictions
    # For example, if you want to ignore certain tokens, you could multiply the loss by a mask
    return loss.mean()


# Example usage
num_tokens = 256  # Number of unique block types
embedding_dim = 50  # Size of the block embedding vector
output_dim = 256  # Number of distinct Minecraft block types you want to model
model = MinecraftStructureGenerator(num_tokens, embedding_dim, output_dim)

# Example block tokens
block_tokens = torch.randint(
    0, num_tokens, (1, 64*64*64), dtype=torch.long)  # Random block tokens

# Get predicted block embeddings
predicted_block_embeddings = model(block_tokens)

# Assume we have true block embeddings (from ground truth labels)
# Randomly generated for this example
true_block_embeddings = torch.randn(1, 64*64*64, embedding_dim)

# Compute custom loss
loss = custom_embedding_loss(
    predicted_block_embeddings, true_block_embeddings, block_tokens)
