import lightning as L
import torch
from lightning.pytorch.utilities.grads import grad_norm
from torch import optim

from model import TransformerMinecraftStructureGenerator


class LightningTransformerMinecraftStructureGenerator(L.LightningModule):
    def __init__(self, num_classes, max_sequence_length, embedding_dim, freeze_encoder, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerMinecraftStructureGenerator(
            num_classes=num_classes, max_sequence_length=max_sequence_length, embedding_dim=embedding_dim, freeze_encoder=freeze_encoder)
        self.learning_rate = learning_rate
        self.validation_step_outputs = []

    def forward(self, prompt, structure):
        return self.model(prompt, structure)

    def generate(self, prompt: str, autoregressive: bool):
        return self.model.generate_structure(prompt, autoregressive)

    def loss_function(self, predictions, targets):
        return torch.nn.functional.cross_entropy(predictions, targets)

    def _forward_and_loss(self, batch):
        _, structure, prompt = batch
        predictions = self.model(prompt, structure)
        loss = self.loss_function(predictions, structure)
        return predictions, loss

    def training_step(self, batch, batch_idx):
        _, loss = self._forward_and_loss(batch)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        predictions, loss = self._forward_and_loss(batch)
        data_module = self.trainer.datamodule
        generator_type, _ = data_module.val_datasets[dataloader_idx]
        self.log(f'val_loss/{generator_type}',
                 loss, add_dataloader_idx=False)
        self.validation_step_outputs.append(
            {'val_loss': loss, 'num_samples': predictions.size(0)})

    def on_validation_epoch_end(self):
        val_loss_total = torch.tensor(0.0, device=self.device)
        num_samples_total = 0
        for output in self.validation_step_outputs:
            val_loss_total += output['val_loss'] * output['num_samples']
            num_samples_total += output['num_samples']
        weighted_avg_loss = val_loss_total / num_samples_total
        self.log('val_loss', weighted_avg_loss)
        self.validation_step_outputs.clear()

    def on_before_optimizer_step(self, optimizer):
        # Compute the 2-norm for each layer
        # If using mixed precision, the gradients are already unscaled here
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        # return optimizer
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.1, patience=50, verbose=True),
            'monitor': 'train_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
