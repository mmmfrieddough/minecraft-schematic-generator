import torch.nn as nn
import torch.nn.functional as F


class MinecraftStructureLoss(nn.Module):
    def __init__(self, block_embeddings, token_weight=1.0, embedding_weight=1.0):
        super(MinecraftStructureLoss, self).__init__()
        self.block_embeddings = block_embeddings
        self.token_weight = token_weight
        self.embedding_weight = embedding_weight

    def forward(self, predicted_embeddings, predicted_tokens, target_tokens):
        # Print shape and type of inputs
        print('predicted_embeddings:', predicted_embeddings.shape,
              predicted_embeddings.dtype)
        print('predicted_tokens:', predicted_tokens.shape,
              predicted_tokens.dtype)
        print('target_tokens:', target_tokens.shape, target_tokens.dtype)

        # Token loss - standard classification loss
        token_loss = F.cross_entropy(predicted_tokens, target_tokens)

        # Embedding loss - measure the distance between predicted and target embeddings
        target_embeddings = self.block_embeddings(target_tokens)
        embedding_loss = F.mse_loss(predicted_embeddings, target_embeddings)

        # Combine losses with weights
        return self.token_weight * token_loss + self.embedding_weight * embedding_loss
