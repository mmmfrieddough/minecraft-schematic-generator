import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.utilities.grads import grad_norm
from torch import optim
from torch.profiler import record_function

from minecraft_schematic_generator.model import TransformerMinecraftStructureGenerator


class LightningTransformerMinecraftStructureGenerator(L.LightningModule):
    def __init__(
        self,
        num_classes,
        max_sequence_length,
        embedding_dropout,
        embedding_dim,
        model_dim,
        num_heads,
        num_layers,
        decoder_dropout,
        max_learning_rate,
        warmup_proportion,
    ):
        super().__init__()
        self.model = TransformerMinecraftStructureGenerator(
            num_classes=num_classes,
            max_sequence_length=max_sequence_length,
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

    def forward(self, structure):
        return self.model(structure)

    def loss_function(self, predictions, targets):
        return torch.nn.functional.cross_entropy(predictions, targets, ignore_index=0)

    def _forward_and_loss(self, batch: torch.Tensor):
        with record_function("data_prep"):
            # Get the structures
            full_structures, masked_structures = batch

            # Create a mask for elements that are zero and adjacent to a value > 1
            masked_structures = masked_structures.unsqueeze(1)
            mask = self.generate_neighbor_mask(masked_structures) & (
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

    def training_step(self, batch, _):
        with record_function("training_step_total"):
            with record_function("data_setup"):
                full_structures, masked_structures = batch
                masked_structures = masked_structures.unsqueeze(1)
                mask = self.generate_neighbor_mask(masked_structures) & (
                    masked_structures == 0
                )
                mask = mask.squeeze(1)

            with record_function("forward_and_loss"):
                _, loss, perplexity = self._forward_and_loss(batch)

            with record_function("logging"):
                self.log("train_loss", loss)
                self.log("train_perplexity", perplexity)

        return loss

    def validation_step(self, batch, _, dataloader_idx=0):
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

    def on_validation_epoch_end(self):
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

    def generate_neighbor_mask(self, tensor):
        """Generates a mask indicating if an element has a neighbor > 1."""
        kernel = torch.ones((1, 1, 3, 3, 3), dtype=tensor.dtype, device=tensor.device)
        kernel[0, 0, 1, 1, 1] = 0  # Ignore the central element

        # Create a mask of elements greater than 1
        greater_than_1 = tensor > 1

        # Convolve to count neighbors that are greater than 1
        neighbors_greater_than_1 = (
            F.conv3d(greater_than_1.float(), kernel.float(), padding=1) >= 1
        )

        # Return the mask
        return neighbors_greater_than_1

    def _get_valid_positions(self, structure, filled_positions):
        """Get ordered list of valid positions to fill, from center outward."""
        # Generate mask of valid next elements
        mask_structure = structure * filled_positions
        mask = self.generate_neighbor_mask(mask_structure) & (structure == 0)

        if not mask.any():
            return None

        # Get positions that need filling
        valid_positions = mask.squeeze().nonzero()

        # Calculate distances from center
        center = torch.tensor([5.0, 5.0, 5.0], device=valid_positions.device)
        distances = torch.norm(valid_positions.float() - center, dim=1)

        # Return positions ordered by distance from center
        return valid_positions[torch.argsort(distances)]

    def _predict_single_position(
        self,
        flattened_structure,
        pos,
        temperature,
        iteration=0,
        air_probability_scaling=0,
    ):
        """Make a prediction for a single position in the structure."""
        with torch.autocast(device_type="cuda"):
            z, y, x = pos
            flat_idx = z * 11 * 11 + y * 11 + x
            logits = self(flattened_structure)
            logits_for_position = logits[0, :, flat_idx]

            # Apply air probability scaling
            logits_for_position[1] += iteration * air_probability_scaling

            # Sample from the distribution
            probabilities = F.softmax(logits_for_position / temperature, dim=-1)
            predicted_token = torch.multinomial(probabilities, num_samples=1).item()

            return (
                predicted_token,
                probabilities[predicted_token].item(),
                probabilities[1].item(),
            )

    def one_shot_inference(self, structure, temperature=1.0, use_greedy=False):
        """Return a new structure with predictions for masked positions.

        Args:
            structure: Input structure tensor
            temperature: Temperature for softmax sampling (ignored if use_greedy=True)
            use_greedy: If True, always select most likely token without sampling
        """
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            # Store original device and dimensionality
            original_device = structure.device
            was_3d = structure.dim() == 3

            # Move to model's device and ensure we have the right shape
            structure = structure.to(self.device)
            if was_3d:
                structure = structure.unsqueeze(0).unsqueeze(0)

            # Flatten the spatial dimensions (depth, height, width) into one dimension
            batch_size, channels, depth, height, width = structure.size()
            flattened = structure.view(batch_size, channels, depth * height * width)
            flattened = flattened.squeeze(1)  # Remove channel dimension

            logits = self(flattened)

            if use_greedy:
                predictions = torch.argmax(logits, dim=1)
            else:
                # Reshape logits to [batch_size * sequence_length, num_classes]
                reshaped_logits = logits.permute(0, 2, 1).reshape(-1, logits.size(1))
                probabilities = F.softmax(reshaped_logits / temperature, dim=1)
                predictions = torch.multinomial(probabilities, num_samples=1).view(
                    batch_size, -1
                )

            predictions = predictions.view(batch_size, depth, height, width)

            result = structure.squeeze(
                1
            ).clone()  # Remove channel dimension from structure
            result[result == 0] = predictions[result == 0]

            # Restore original shape if needed
            if was_3d:
                result = result.squeeze(0)

            return result.to(original_device)

    def fill_structure(
        self,
        structure,
        temperature,
        start_radius,
        max_iterations,
        max_blocks,
        air_probability_iteration_scaling,
    ):
        structure = structure.to(self.device)

        # Ensure tensor has batch and channel dimensions
        if structure.dim() == 3:
            structure = structure.unsqueeze(0).unsqueeze(0)

        flattened_structure = structure.view(1, -1)

        # Initialize mask of valid next elements
        filled_positions = torch.zeros_like(structure, dtype=torch.bool)
        filled_positions[
            0,
            0,
            5 - start_radius : 5 + start_radius + 1,
            5 - start_radius : 5 + start_radius + 1,
            5 - start_radius : 5 + start_radius + 1,
        ] = 1

        with torch.no_grad():
            filled_blocks = 0

            for iteration in range(max_iterations):
                print(f"Iteration {iteration+1}/{max_iterations}")

                valid_positions = self._get_valid_positions(structure, filled_positions)
                if valid_positions is None:
                    print("No more elements to update")
                    break

                # Process each position in center-out order
                for pos in valid_positions:
                    predicted_token, selected_probability, air_probability = (
                        self._predict_single_position(
                            flattened_structure,
                            pos,
                            temperature,
                            iteration,
                            air_probability_iteration_scaling,
                        )
                    )

                    z, y, x = pos
                    print(
                        f"Selected token {predicted_token} with probability {selected_probability*100:.1f}%, air probability {air_probability*100:.1f}%"
                    )

                    yield predicted_token, z, y, x

                    if predicted_token != 1:
                        filled_positions[0, 0, z, y, x] = 1
                        filled_blocks += 1
                        print(f"Filled {filled_blocks}/{max_blocks} solid blocks")
                    if filled_blocks >= max_blocks:
                        break
                    structure[0, 0, z, y, x] = predicted_token

                if filled_blocks >= max_blocks:
                    break

    def complete_structure(self, masked_structure, temperature=1.0):
        for predicted_token, z, y, x in self.fill_structure(
            masked_structure, temperature
        ):
            masked_structure[z, y, x] = predicted_token
        return masked_structure

    def on_before_optimizer_step(self, optimizer):
        if self.global_step % self.trainer.log_every_n_steps == 0:
            norms = grad_norm(self.model, norm_type=2)
            self.log_dict(norms)

    def configure_optimizers(self):
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
