import random

import lightning as L
import torch
from torch import optim


class VAELightningModule(L.LightningModule):
    def __init__(self, vae, learning_rate=1e-3):
        super().__init__()
        self.vae = vae
        self.learning_rate = learning_rate
        self.validation_step_outputs = []

    def forward(self, x):
        return self.vae(x)

    # def generate(self, prompt: str, temperature: float):
    #     self.eval()
    #     with torch.no_grad():
    #         # Select a random sample from the dataset
    #         _, _, random_sample = self.trainer.datamodule.train_dataloader().dataset[random.randint(
    #             0, len(self.trainer.datamodule.train_dataloader().dataset) - 1)]
    #         random_sample = random_sample.unsqueeze(0)
    #         random_sample = random_sample.to(self.device)
    #         # Run the random sample through the full VAE
    #         sample, _, _ = self.vae(random_sample)
    #         sample = torch.softmax(sample, dim=1)
    #         sample = torch.argmax(sample, dim=1)
    #         sample = sample.squeeze(0)
    #     self.train()
    #     # random_sample = random_sample.squeeze(0)
    #     # return random_sample
    #     return sample

    def generate(self, prompt: str, temperature: float):
        self.eval()
        with torch.no_grad():
            z = torch.randn(1, 4, 2, 2, 2).to(self.device)
            sample = self.vae.decoder(z)
            sample = torch.softmax(sample, dim=1)
            sample = torch.argmax(sample, dim=1)
            sample = sample.squeeze(0)
        self.train()
        return sample

    def loss_function(self, predictions, targets, mu, logvar):
        cross_entropy = torch.nn.functional.cross_entropy(predictions, targets)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return cross_entropy, kl_divergence

    def training_step(self, batch, batch_idx):
        _, _, x = batch
        recon_x, mu, logvar = self.vae(x)
        cross_entropy, kl_divergence = self.loss_function(
            recon_x, x, mu, logvar)
        self.log('train_cross_entropy', cross_entropy)
        self.log('train_kl_divergence', kl_divergence)
        return cross_entropy + kl_divergence

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        _, _, x = batch
        recon_x, mu, logvar = self.vae(x)
        cross_entropy, kl_divergence = self.loss_function(
            recon_x, x, mu, logvar)
        data_module = self.trainer.datamodule
        generator_type, _ = data_module.val_datasets[dataloader_idx]
        self.log(f'val_loss_cross_entropy/{generator_type}',
                 cross_entropy, add_dataloader_idx=False)
        self.validation_step_outputs.append({
            'type': 'val_loss_cross_entropy',
            'loss': cross_entropy,
            'num_samples': recon_x.size(0)
        })
        self.log(f'val_loss_kl_divergence/{generator_type}',
                 kl_divergence, add_dataloader_idx=False)
        self.validation_step_outputs.append({
            'type': 'val_loss_kl_divergence',
            'loss': kl_divergence,
            'num_samples': recon_x.size(0)
        })
        total_loss = cross_entropy + kl_divergence
        self.log(f'val_loss/{generator_type}',
                 total_loss, add_dataloader_idx=False)
        self.validation_step_outputs.append({
            'type': 'val_loss',
            'loss': total_loss,
            'num_samples': recon_x.size(0)
        })

    def on_validation_epoch_end(self):
        loss_totals = {}
        num_samples_totals = {}
        for output in self.validation_step_outputs:
            loss_type = output['type']
            loss_value = output['loss']
            num_samples = output['num_samples']
            if loss_type not in loss_totals:
                loss_totals[loss_type] = torch.tensor(0.0, device=self.device)
                num_samples_totals[loss_type] = 0
            loss_totals[loss_type] += loss_value * num_samples
            num_samples_totals[loss_type] += num_samples
        for loss_type, loss_total in loss_totals.items():
            weighted_avg_loss = loss_total / num_samples_totals[loss_type]
            self.log(loss_type, weighted_avg_loss)
        self.validation_step_outputs.clear()

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        scheduler = {
            'scheduler': torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=0.5, patience=5, verbose=True, threshold=1e-4),
            'monitor': 'val_loss',
            'interval': 'epoch',
            'frequency': 1
        }
        return {'optimizer': optimizer, 'lr_scheduler': scheduler}
