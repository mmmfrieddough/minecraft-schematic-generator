import lightning as L
import torch
from lightning.pytorch.utilities.grads import grad_norm
from torch import optim
from torch.profiler import record_function

from minecraft_schematic_generator.model import TransformerMinecraftStructureGenerator


class LightningTransformerMinecraftStructureGenerator(L.LightningModule):
    def __init__(
        self,
        num_classes: int,
        max_structure_size: int,
        embedding_dropout: float,
        embedding_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        decoder_dropout: float,
        max_learning_rate: float,
        warmup_proportion: float,
    ):
        super().__init__()
        self.model = TransformerMinecraftStructureGenerator(
            num_classes=num_classes,
            max_structure_size=max_structure_size,
            embedding_dropout=embedding_dropout,
            model_dim=model_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            decoder_dropout=decoder_dropout,
            embedding_dim=embedding_dim,
        )
        self.num_classes = num_classes
        self.max_learning_rate = max_learning_rate
        self.warmup_proportion = warmup_proportion
        self.validation_step_outputs = []
        self.save_hyperparameters()

    def forward(self, structure: torch.Tensor) -> torch.Tensor:
        return self.model(structure)

    def loss_function(
        self, predictions: torch.Tensor, targets: torch.Tensor
    ) -> torch.Tensor:
        return torch.nn.functional.cross_entropy(predictions, targets, ignore_index=0)

    def _forward_and_loss(self, batch: torch.Tensor) -> torch.Tensor:
        with record_function("data_prep"):
            # Get the structures
            full_structures, masked_structures = batch

            # Create a mask for elements that are zero and adjacent to a value > 1
            masked_structures = masked_structures.unsqueeze(1)
            mask = self.model.generate_neighbor_mask(masked_structures) & (
                masked_structures == 0
            )
            mask = mask.squeeze(1)

            # Flatten the structures and mask
            batch_size = full_structures.size(0)
            full_structures = full_structures.view(batch_size, -1)
            masked_structures = masked_structures.view(batch_size, -1)
            mask = mask.view(batch_size, -1)

            # Zero out the non-masked elements in full_structures
            full_structures = full_structures * mask

        with record_function("model_forward"):
            # Make the predictions
            predicted_structures = self(masked_structures)

        with record_function("compute_loss"):
            # Compute the loss
            loss = self.loss_function(predicted_structures, full_structures)

            # Compute perplexity
            perplexity = torch.exp(loss)

        return predicted_structures, loss, perplexity

    def training_step(self, batch: torch.Tensor, _) -> torch.Tensor:
        with record_function("training_step_total"):
            with record_function("forward_and_loss"):
                _, loss, perplexity = self._forward_and_loss(batch)

            with record_function("logging"):
                self.log("train_loss", loss)
                self.log("train_perplexity", perplexity)

        return loss

    def validation_step(self, batch: torch.Tensor, _, dataloader_idx: int = 0) -> None:
        predictions, loss, perplexity = self._forward_and_loss(batch)
        data_module = self.trainer.datamodule
        dataset_name = data_module.get_val_dataset_name(dataloader_idx)
        self.log(
            f"val_loss/{dataset_name}", loss, add_dataloader_idx=False, sync_dist=True
        )
        self.log(
            f"val_perplexity/{dataset_name}",
            perplexity,
            add_dataloader_idx=False,
            sync_dist=True,
        )
        self.validation_step_outputs.append(
            {
                "val_loss": loss,
                "val_perplexity": perplexity,
                "num_samples": predictions.size(0),
            }
        )

    def on_validation_epoch_end(self) -> None:
        val_loss_total = torch.tensor(0.0, device=self.device)
        val_perplexity_total = torch.tensor(0.0, device=self.device)
        num_samples_total = 0
        for output in self.validation_step_outputs:
            val_loss_total += output["val_loss"] * output["num_samples"]
            val_perplexity_total += output["val_perplexity"] * output["num_samples"]
            num_samples_total += output["num_samples"]
        weighted_avg_loss = val_loss_total / num_samples_total
        weighted_avg_perplexity = val_perplexity_total / num_samples_total
        self.log("val_loss", weighted_avg_loss, sync_dist=True)
        self.log("val_perplexity", weighted_avg_perplexity, sync_dist=True)
        self.validation_step_outputs.clear()

    def on_before_optimizer_step(self, _) -> None:
        if self.global_step % self.trainer.log_every_n_steps == 0:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def configure_optimizers(self) -> dict:
        optimizer = optim.Adam(self.parameters(), lr=self.max_learning_rate)

        scheduler = {
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=self.max_learning_rate,
                total_steps=self.trainer.estimated_stepping_batches,
                pct_start=self.warmup_proportion,
            ),
            "interval": "step",
            "frequency": 1,
        }

        return {"optimizer": optimizer, "lr_scheduler": scheduler}
