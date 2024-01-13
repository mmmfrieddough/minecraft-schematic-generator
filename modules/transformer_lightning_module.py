import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.grads import grad_norm
from torch import optim
from torchmetrics.functional import accuracy

from model import TransformerMinecraftStructureGenerator


class LightningTransformerMinecraftStructureGenerator(L.LightningModule):
    def __init__(self, num_classes, max_sequence_length, embedding_dropout, model_dim, num_heads, num_layers, decoder_dropout, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerMinecraftStructureGenerator(
            num_classes=num_classes,
            max_sequence_length=max_sequence_length,
            embedding_dropout=embedding_dropout,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            decoder_dropout=decoder_dropout
        )
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.validation_step_outputs = []

    def forward(self, structure):
        return self.model(structure)

    def loss_function(self, predictions, targets):
        return torch.nn.functional.cross_entropy(predictions, targets)

    def _forward_and_loss(self, batch):
        masked_structure, full_structure = batch
        masked_structure = masked_structure.view(masked_structure.size(0), -1)
        full_structure = full_structure.view(full_structure.size(0), -1)

        filled_structure = self(masked_structure)

        # Create a mask for positions that were originally zero (masked)
        mask = (masked_structure == 0)

        # Apply the mask to select only the logits for the masked positions
        filled_structure_masked = torch.masked_select(
            filled_structure.transpose(1, 2), mask.unsqueeze(2)).view(-1, self.num_classes)
        full_structure_masked = torch.masked_select(full_structure, mask)

        # Compute loss and accuracy only on the masked positions
        loss = self.loss_function(
            filled_structure_masked, full_structure_masked)
        predictions = torch.argmax(filled_structure_masked, dim=1)
        acc = accuracy(predictions, full_structure_masked,
                       num_classes=self.num_classes, task='multiclass')

        return filled_structure, loss, acc

    def training_step(self, batch, batch_idx):
        _, loss, acc = self._forward_and_loss(batch)
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        predictions, loss, acc = self._forward_and_loss(batch)
        data_module = self.trainer.datamodule
        generator_type, _ = data_module.val_datasets[dataloader_idx]
        self.log(f'val_loss/{generator_type}',
                 loss, add_dataloader_idx=False)
        self.log(f'val_accuracy/{generator_type}',
                 acc, add_dataloader_idx=False)
        self.validation_step_outputs.append(
            {'val_loss': loss, 'val_accuracy': acc, 'num_samples': predictions.size(0)})

    def on_validation_epoch_end(self):
        val_loss_total = torch.tensor(0.0, device=self.device)
        val_accuarcy_total = torch.tensor(0.0, device=self.device)
        num_samples_total = 0
        for output in self.validation_step_outputs:
            val_loss_total += output['val_loss'] * output['num_samples']
            val_accuarcy_total += output['val_accuracy'] * \
                output['num_samples']
            num_samples_total += output['num_samples']
        weighted_avg_loss = val_loss_total / num_samples_total
        weighted_avg_accuracy = val_accuarcy_total / num_samples_total
        self.log('val_loss', weighted_avg_loss)
        self.log('val_accuracy', weighted_avg_accuracy)
        self.validation_step_outputs.clear()

    def generate_neighbor_mask(self, tensor):
        """Generates a mask indicating if an element is 0 and has a neighbor > 1."""
        kernel = torch.ones((1, 1, 3, 3, 3), dtype=tensor.dtype,
                            device=tensor.device)
        kernel[0, 0, 1, 1, 1] = 0  # Ignore the central element

        # Create a mask of elements greater than 1
        greater_than_1 = tensor > 1

        # Convolve to count neighbors that are greater than 1
        neighbors_greater_than_1 = F.conv3d(
            greater_than_1.float(), kernel.float(), padding=1) >= 1

        # Create a mask for elements that are 0
        is_zero = tensor == 0

        # Combine both conditions
        return neighbors_greater_than_1 & is_zero

    def fill_structure(self, structure, temperature=1.0, fill_order='random'):
        self.eval()
        structure = structure.to(self.device)

        # Ensure tensor has batch and channel dimensions
        if structure.dim() == 3:
            structure = structure.unsqueeze(0).unsqueeze(0)

        # Precompute constants
        num_elements = structure.size(2) * structure.size(3)
        structure_view = structure.squeeze(0).view(1, -1)

        with torch.no_grad():
            while True:
                # Generate mask of valid next elements
                mask = self.generate_neighbor_mask(structure)
                if not mask.any():
                    break  # Exit if no more elements to update

                # Get traversal indices based on the specified fill order
                indices = mask.nonzero(as_tuple=False)
                if fill_order == 'bottom_up':
                    indices = indices[indices[:, 3].argsort(descending=False)]
                elif fill_order == 'random':
                    indices = indices[torch.randperm(indices.size(0))]
                else:
                    raise ValueError(f"Unknown fill order: {fill_order}")

                for idx in indices:
                    z, y, x = idx[2], idx[3], idx[4]
                    linear_index = num_elements * z + structure.size(3) * y + x

                    logits = self(structure_view).squeeze(0)
                    logits_for_position = logits[:, linear_index]
                    probabilities = F.softmax(
                        logits_for_position / temperature, dim=-1)
                    predicted_token = torch.multinomial(
                        probabilities, num_samples=1).item()
                    yield predicted_token, z, y, x
                    structure[0, 0, z, y, x] = predicted_token

        self.train()

    def complete_structure(self, structure, temperature=1.0, fill_order='random'):
        for _ in self.fill_structure(structure, temperature, fill_order):
            pass
        return structure

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=25, verbose=True, threshold=1e-5),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
