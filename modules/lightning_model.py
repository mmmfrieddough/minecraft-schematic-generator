import lightning as L
import torch
import torch.nn.functional as F
from torch import optim

from modules.data_module import MinecraftDataModule


# class EmbeddingDiversityLoss(torch.nn.Module):
#     def forward(self, embedding_layer):
#         # Extract embeddings from the embedding layer
#         embeddings = embedding_layer.weight
#         # Normalize embeddings
#         normalized_embeddings = F.normalize(embeddings, p=2, dim=1)
#         # Compute similarity matrix (cosine similarity)
#         similarity_matrix = torch.matmul(
#             normalized_embeddings, normalized_embeddings.T)
#         # Apply ReLU to keep only positive similarities
#         positive_similarities = F.relu(similarity_matrix)
#         # Penalize high similarity values for different classes
#         similarity_loss = torch.triu(positive_similarities, diagonal=1).mean()
#         return similarity_loss


class LightningMinecraftStructureGenerator(L.LightningModule):
    def __init__(self, model, learning_rate=1e-3):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.example_input_array = self._get_example_input()
        self.validation_step_outputs = []

    def _get_example_input(self):
        return torch.rand(1, self.model.prompt_embedding_dim)

    def forward(self, x):
        return self.model(x)

    def loss_function(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets)
        return ce_loss

    # def embeddings_to_labels(self, embeddings):
    #     # Retrieve the embeddings from the embedding layer
    #     # Shape: [num_classes, embedding_size]
    #     class_embeddings = self.embedding_layer.weight.data

    #     # Reshape 'embeddings' to two dimensions: [batch_size * depth * height * width, channels]
    #     batch_size, channels, depth, height, width = embeddings.size()
    #     embeddings_flat = embeddings.view(
    #         batch_size * depth * height * width, channels)

    #     # Calculate the distance between each flattened embedding and the class embeddings
    #     # Shape: [batch_size * d * h * w, num_classes]
    #     distances = torch.cdist(embeddings_flat, class_embeddings)

    #     # Find the index of the nearest embedding for each voxel
    #     nearest_indices = torch.argmin(distances, dim=1)

    #     # Reshape back to the original dimensions (minus channels)
    #     nearest_classes = nearest_indices.view(
    #         batch_size, depth, height, width)

    #     return nearest_classes

    # def labels_to_embeddings(self, labels):
    #     batch_size, d1, d2, d3 = labels.shape

    #     # Flatten the labels while keeping the batch size dimension
    #     labels_flat = labels.view(batch_size, -1)

    #     # Get embeddings for each label
    #     target_embeddings_flat = self.embedding_layer(labels_flat)
    #     embedding_size = target_embeddings_flat.shape[-1]

    #     # Reshape to the desired output shape: [batch_size, embedding_size, d1, d2, d3]
    #     target_embeddings = target_embeddings_flat.view(
    #         batch_size, embedding_size, d1, d2, d3)

    #     return target_embeddings

    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        x, y, _ = batch
        y_hat = self(x)
        loss = self.loss_function(y_hat, y)
        data_module: MinecraftDataModule = self.trainer.datamodule
        generator_type, _ = data_module.val_datasets[dataloader_idx]
        self.log(f'val_loss/{generator_type}',
                 loss, add_dataloader_idx=False)
        self.validation_step_outputs.append(
            {'val_loss': loss, 'num_samples': y.size(0)})

    def on_validation_epoch_end(self):
        val_loss_total = torch.tensor(0.0, device=self.device)
        num_samples_total = 0
        for output in self.validation_step_outputs:
            val_loss_total += output['val_loss'] * output['num_samples']
            num_samples_total += output['num_samples']
        weighted_avg_loss = val_loss_total / num_samples_total
        self.log('val_loss', weighted_avg_loss)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=10, verbose=True),
            'monitor': 'train_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
