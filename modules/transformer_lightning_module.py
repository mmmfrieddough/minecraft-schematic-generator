import lightning as L
import torch
from lightning.pytorch.utilities.grads import grad_norm
from torch import optim
from torchmetrics.functional import accuracy

from model import TransformerMinecraftStructureGenerator


class LightningTransformerMinecraftStructureGenerator(L.LightningModule):
    def __init__(self, num_classes, max_sequence_length, embedding_dim, embedding_dropout, num_heads, num_layers, decoder_dropout, freeze_encoder=False, learning_rate=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = TransformerMinecraftStructureGenerator(
            num_classes=num_classes,
            max_sequence_length=max_sequence_length,
            embedding_dim=embedding_dim,
            embedding_dropout=embedding_dropout,
            num_heads=num_heads,
            num_layers=num_layers,
            decoder_dropout=decoder_dropout,
            freeze_encoder=freeze_encoder
        )
        self.num_classes = num_classes
        self.learning_rate = learning_rate
        self.validation_step_outputs = []

    def forward(self, prompt, structure):
        return self.model(prompt, structure)

    def generate(self, prompt: str, temperature: float):
        return self.model.generate_structure(prompt, temperature)

    def loss_function(self, predictions, targets):
        return torch.nn.functional.cross_entropy(predictions, targets)

    def _forward_and_loss(self, batch):
        _, prompt, structure = batch
        predictions = self.model(prompt, structure)
        loss = self.loss_function(predictions, structure)
        acc = accuracy(torch.argmax(predictions, dim=1), structure,
                       task='multiclass', num_classes=self.num_classes)
        return predictions, loss, acc

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

    def on_before_optimizer_step(self, optimizer):
        norms = grad_norm(self.model, norm_type=2)
        self.log_dict(norms)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=10, verbose=True, threshold=1e-5),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
