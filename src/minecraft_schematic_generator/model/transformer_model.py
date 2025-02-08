import torch
from huggingface_hub import ModelCard, PyTorchModelHubMixin
from torch import nn
from torch.nn import functional as F

from minecraft_schematic_generator.constants import AIR_BLOCK_ID, MASK_BLOCK_ID

MODEL_CARD_TEMPLATE = """
---
language: en
tags:
- minecraft
- structure-generation
- pytorch
license: mit
repository: https://github.com/mmmfrieddough/minecraft-schematic-generator
---

# Minecraft Structure Generator

This model generates Minecraft structures using a decoder-only transformer architecture.

## Model Details

- Architecture: Decoder-only Transformer
- Vocabulary Size: {{ num_classes }} block types
- Sequence Length: {{ max_sequence_length }}
- Embedding Dimension: {{ embedding.embedding_dim }}
- Attention Heads: {{ attention_heads }}
- Transformer Layers: {{ transformer_layers }}
- Parameters: {{ parameters }}
"""


class TransformerMinecraftStructureGenerator(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        num_classes: int,
        block_str_mapping: dict[str, int],
        max_structure_size: int,
        embedding_dropout: float,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        decoder_dropout: float,
        embedding_dim: int = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.block_str_mapping = block_str_mapping
        self.max_structure_size = max_structure_size
        self.max_sequence_length = max_structure_size**3
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim or model_dim

        # Input
        self.embedding = nn.Embedding(num_classes, self.embedding_dim)
        if self.embedding_dim != model_dim:
            self.embedding_projection = nn.Linear(self.embedding_dim, model_dim)
        self.register_buffer("positions", torch.arange(self.max_sequence_length))
        self.positional_embedding = nn.Embedding(self.max_sequence_length, model_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            model_dim, num_heads, dropout=decoder_dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)

        # Output
        if self.embedding_dim != model_dim:
            self.output_projection = nn.Linear(model_dim, self.embedding_dim)
        self.output_layer = nn.Linear(self.embedding_dim, num_classes)
        self.output_layer.weight = self.embedding.weight

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)

        # Initialize positional embedding
        nn.init.xavier_uniform_(self.positional_embedding.weight)

        # Initialize projection layers if they exist
        if self.embedding_dim != self.model_dim:
            nn.init.xavier_uniform_(self.embedding_projection.weight)
            nn.init.zeros_(self.embedding_projection.bias)
            nn.init.xavier_uniform_(self.output_projection.weight)
            nn.init.zeros_(self.output_projection.bias)

        for layer in self.decoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
            # Initialize biases
            nn.init.zeros_(layer.self_attn.in_proj_bias)
            nn.init.zeros_(layer.self_attn.out_proj.bias)
            nn.init.zeros_(layer.linear1.bias)
            nn.init.zeros_(layer.linear2.bias)
            # Initialize layer norm parameters
            nn.init.ones_(layer.norm1.weight)
            nn.init.zeros_(layer.norm1.bias)
            nn.init.ones_(layer.norm2.weight)
            nn.init.zeros_(layer.norm2.bias)

        if self.output_layer.bias is not None:
            nn.init.zeros_(self.output_layer.bias)

    def forward(self, structure_flat: torch.Tensor) -> torch.Tensor:
        # Embed the sequence
        output_seq = self.embedding(structure_flat)

        # Project the embedding if necessary
        if self.embedding_dim != self.model_dim:
            output_seq = self.embedding_projection(output_seq)

        # Use cached positions tensor, sliced to current sequence length
        output_seq = output_seq + self.positional_embedding(
            self.positions[: structure_flat.size(1)]
        )

        # Add dropout
        output_seq = self.embedding_dropout(output_seq)

        # Reshape the sequence so the batch is the first dimension
        output_seq = output_seq.transpose(0, 1)

        # Run the transformer
        output = self.decoder(tgt=output_seq, memory=output_seq)

        # Reshape back to the batch first format
        output = output.transpose(0, 1)

        # Project back to embedding dimension if necessary
        if self.embedding_dim != self.model_dim:
            output = self.output_projection(output)

        # Run the output layer
        output = self.output_layer(output)

        # Reshape so the class logits are before the sequence dimension
        output = output.transpose(1, 2)

        return output

    def generate_model_card(self) -> ModelCard:
        return ModelCard.from_template(
            card_data=self._hub_mixin_info.model_card_data,
            template_str=MODEL_CARD_TEMPLATE,
            num_classes=self.num_classes,
            max_sequence_length=self.max_sequence_length,
            embedding=self.embedding.embedding_dim,
            attention_heads=self.decoder.layers[0].self_attn.num_heads,
            transformer_layers=len(self.decoder.layers),
            parameters=f"{sum(p.numel() for p in self.parameters()):,}",
        )

    def generate_neighbor_mask(self, tensor: torch.Tensor) -> torch.Tensor:
        """Generates a mask indicating if an element has a neighbor > 1."""
        kernel = torch.ones((1, 1, 3, 3, 3), dtype=tensor.dtype, device=tensor.device)
        kernel[0, 0, 1, 1, 1] = 0  # Ignore the central element

        # Create a mask of elements greater than AIR_BLOCK_ID
        greater_than_1 = tensor > AIR_BLOCK_ID

        # Convolve to count neighbors that are greater than AIR_BLOCK_ID
        neighbors_greater_than_1 = (
            F.conv3d(greater_than_1.float(), kernel.float(), padding=1) >= 1
        )

        # Return the mask
        return neighbors_greater_than_1

    def _get_valid_positions(
        self, structure: torch.Tensor, filled_positions: torch.Tensor
    ) -> torch.Tensor:
        """Get ordered list of valid positions to fill, from center outward."""
        # Generate mask of valid next elements
        mask_structure = structure * filled_positions
        mask = self.generate_neighbor_mask(mask_structure) & (
            structure == MASK_BLOCK_ID
        )

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
        flattened_structure: torch.Tensor,
        pos: torch.Tensor,
        temperature: float,
        iteration: int = 0,
        air_probability_scaling: float = 0.0,
    ):
        """Make a prediction for a single position in the structure."""
        with torch.autocast(device_type="cuda"):
            z, y, x = pos
            flat_idx = z * self.max_structure_size**2 + y * self.max_structure_size + x
            logits = self(flattened_structure)
            logits_for_position = logits[0, :, flat_idx]

            # Apply air probability scaling
            logits_for_position[AIR_BLOCK_ID] += iteration * air_probability_scaling

            # Sample from the distribution
            probabilities = F.softmax(logits_for_position / temperature, dim=-1)
            predicted_token = torch.multinomial(probabilities, num_samples=1).item()

            # print(
            #     f"Selected token {predicted_token} with probability {probabilities[predicted_token].item()*100:.1f}%, air probability {probabilities[1].item()*100:.1f}%"
            # )

            return predicted_token

    def one_shot_inference(
        self,
        structure: torch.Tensor,
        temperature: float = 1.0,
        use_greedy: bool = False,
    ) -> torch.Tensor:
        """Return a new structure with predictions for masked positions.

        Args:
            structure: Input structure tensor
            temperature: Temperature for softmax sampling (ignored if use_greedy=True)
            use_greedy: If True, always select most likely token without sampling
        """
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            # Store dimensionality
            was_3d = structure.dim() == 3

            # Ensure we have the right shape
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

            return result

    def fill_structure(
        self,
        structure: torch.Tensor,
        temperature: float,
        start_radius: int,
        max_iterations: int,
        max_blocks: int,
        air_probability_iteration_scaling: float,
    ):
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
                # print(f"Iteration {iteration+1}/{max_iterations}")

                valid_positions = self._get_valid_positions(structure, filled_positions)
                if valid_positions is None:
                    # print("No more elements to update")
                    break

                # Process each position in center-out order
                for pos in valid_positions:
                    predicted_token = self._predict_single_position(
                        flattened_structure,
                        pos,
                        temperature,
                        iteration,
                        air_probability_iteration_scaling,
                    )

                    z, y, x = pos
                    yield predicted_token, z, y, x

                    if predicted_token != AIR_BLOCK_ID:
                        filled_positions[0, 0, z, y, x] = 1
                        filled_blocks += 1
                        # print(f"Filled {filled_blocks}/{max_blocks} solid blocks")
                    if filled_blocks >= max_blocks:
                        break
                    structure[0, 0, z, y, x] = predicted_token

                if filled_blocks >= max_blocks:
                    break

    def complete_structure(
        self, masked_structure: torch.Tensor, temperature: float = 1.0
    ) -> torch.Tensor:
        for predicted_token, z, y, x in self.fill_structure(
            masked_structure, temperature
        ):
            masked_structure[z, y, x] = predicted_token
        return masked_structure
