import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.grads import grad_norm
from torch import optim
from torchmetrics.functional import accuracy

from model import TransformerMinecraftStructureGenerator


class LightningTransformerMinecraftStructureGenerator(L.LightningModule):
    def __init__(self, num_classes, max_sequence_length, embedding_dropout, model_dim, num_heads, num_layers, decoder_dropout, max_learning_rate, warmup_steps):
        super().__init__()
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
        self.max_learning_rate = max_learning_rate
        self.warmup_steps = warmup_steps
        self.validation_step_outputs = []
        self.save_hyperparameters()

        # Precompute center-out order for 11x11x11 structure
        self.center_out_order = self._precompute_center_out_order()
        self.center_out_order = self.center_out_order.to(self.device)

    def _precompute_center_out_order(self):
        shape = (11, 11, 11)
        indices = torch.arange(11*11*11).reshape(shape)
        center = torch.tensor(shape) // 2
        distances = torch.sum((indices.nonzero() - center) ** 2, dim=1)
        return distances.argsort()

    def forward(self, structure):
        return self.model(structure)

    def loss_function(self, predictions, targets):
        return torch.nn.functional.cross_entropy(predictions, targets, ignore_index=0)

    def _forward_and_loss(self, batch: torch.Tensor):
        # Get the structures
        full_structures, masked_structures = batch

        # Create a mask for elements that are zero and adjacent to a value > 1
        masked_structures = masked_structures.unsqueeze(1)
        mask = self.generate_neighbor_mask(masked_structures)
        mask = mask.squeeze(1)

        # Flatten the structures and mask
        batch_size = full_structures.size(0)
        full_structures = full_structures.view(batch_size, -1)
        masked_structures = masked_structures.view(batch_size, -1)
        mask = mask.view(batch_size, -1)

        # Zero out the non-masked elements in full_structures
        full_structures = full_structures * mask

        # Make the predictions
        predicted_structures = self(masked_structures)

        # Compute the loss
        loss = self.loss_function(predicted_structures, full_structures)

        # Compute the accuracy
        predictions = torch.argmax(predicted_structures, dim=1)
        acc = accuracy(predictions, full_structures,
                       num_classes=self.num_classes, task='multiclass', ignore_index=0)

        return predicted_structures, loss, acc

    def training_step(self, batch, batch_idx):
        full_structures, masked_structures = batch
        _, loss, acc = self._forward_and_loss(batch)
        self.log('train_loss', loss)
        self.log('train_accuracy', acc)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        predictions, loss, acc = self._forward_and_loss(batch)
        data_module = self.trainer.datamodule
        dataset_name = data_module.get_val_dataset_name(dataloader_idx)
        self.log(f'val_loss/{dataset_name}', loss, add_dataloader_idx=False)
        self.log(f'val_accuracy/{dataset_name}', acc, add_dataloader_idx=False)
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

    def fill_structure(self, structure, temperature=1.0, fill_order='center_out'):
        self.eval()
        structure = structure.to(self.device)

        # Ensure tensor has batch and channel dimensions
        if structure.dim() == 3:
            structure = structure.unsqueeze(0).unsqueeze(0)

        # Precompute constants
        num_elements = structure.size(2) * structure.size(3)
        structure_view = structure.squeeze(0).view(1, -1)

        self.center_out_order = self.center_out_order.to(structure.device)

        # Pre-calculate distance biases
        all_indices = torch.arange(11*11*11, device=structure.device)
        distance_ranks = torch.searchsorted(self.center_out_order, all_indices)
        distance_biases = (distance_ranks.float() /
                           len(self.center_out_order)) * 1000  # Scale by 10

        with torch.no_grad():
            for i in range(100):
                print(f"Step {i}")
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
                elif fill_order == 'center_out':
                    flat_indices = indices[:, 2] * 121 + \
                        indices[:, 3] * 11 + indices[:, 4]
                    order = torch.argsort(torch.searchsorted(
                        self.center_out_order, flat_indices))
                    indices = indices[order]
                else:
                    raise ValueError(f"Unknown fill order: {fill_order}")

                for idx in indices:
                    z, y, x = idx[2], idx[3], idx[4]
                    linear_index = num_elements * z + structure.size(3) * y + x
                    flat_index = z * 121 + y * 11 + x

                    logits = self(structure_view).squeeze(0)
                    logits_for_position = logits[:, linear_index]
                    # + distance_biases[flat_index]
                    logits_for_position[1] += 2 * i
                    probabilities = F.softmax(
                        logits_for_position / temperature, dim=-1)
                    predicted_token = torch.multinomial(
                        probabilities, num_samples=1).item()
                    # print(distance_biases[flat_index].item())
                    # predicted_token = round(
                    #     distance_biases[flat_index].item()) + 1

                    yield predicted_token, z, y, x
                    structure[0, 0, z, y, x] = predicted_token

            # Find any remaining elements with value 0 and yield them as value 1
            remaining_zeros = (structure == 0).nonzero(as_tuple=False)
            for idx in remaining_zeros:
                z, y, x = idx[2], idx[3], idx[4]
                yield 1, z, y, x
                structure[0, 0, z, y, x] = 1

        self.train()

    def complete_structure(self, masked_structure, temperature=1.0, fill_order='random'):
        for predicted_token, z, y, x in self.fill_structure(masked_structure, temperature, fill_order):
            masked_structure[z, y, x] = predicted_token
        return masked_structure

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.max_learning_rate)

        # Warmup scheduler
        def lr_lambda(step): return min((step + 1) / self.warmup_steps, 1.0)
        warmup_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer,
                lr_lambda=lr_lambda
            ),
            'interval': 'step',
            'frequency': 1
        }

        # Reduce on plateau scheduler
        plateau_scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                factor=0.5,
                patience=10,
                verbose=True,
                threshold=1e-4
            ),
            'monitor': 'val_loss',
            'interval': 'step',
            'frequency': max(1, int(self.trainer.val_check_interval * len(self.trainer.datamodule.train_dataloader())))
        }

        return (
            [optimizer],
            [warmup_scheduler, plateau_scheduler]
        )
