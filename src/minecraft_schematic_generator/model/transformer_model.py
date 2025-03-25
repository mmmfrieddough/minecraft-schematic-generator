import torch
from huggingface_hub import ModelCard, PyTorchModelHubMixin
from torch import nn
from torch.nn import functional as F

from minecraft_schematic_generator.constants import AIR_BLOCK_ID, MASK_BLOCK_ID

from .self_attention_decoder import SelfAttentionDecoder, SelfAttentionDecoderLayer

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
- Maximum Sequence Length: {{ max_sequence_length }}
- Embedding Dimension: {{ embedding_dim }}
- Attention Heads: {{ attention_heads }}
- Transformer Layers: {{ transformer_layers }}
- Parameters: {{ parameters }}
"""


class StructureAlternative:
    """Class representing a single alternative structure being generated."""

    def __init__(
        self,
        alternative_id: int,
        structure: torch.Tensor,
        filled_positions: torch.Tensor,
        remaining_iterations: int,
        remaining_blocks: int,
        valid_positions: list[tuple[int, int, int]] | None = None,
    ):
        self.id = alternative_id
        self.structure = structure
        self.filled_positions = filled_positions
        self.remaining_iterations = remaining_iterations
        self.remaining_blocks = remaining_blocks
        self.valid_positions = valid_positions or []

    def get_valid_positions(self):
        """Get valid positions to fill next."""
        if not self.valid_positions:
            self.valid_positions = (
                TransformerMinecraftStructureGenerator._get_valid_positions(
                    self.structure, self.filled_positions
                )
            )
            self.remaining_iterations -= 1

    def place_block(self, token: int, pos: tuple[int, int, int]):
        """Place a block and update state."""
        z, y, x = pos
        if token != AIR_BLOCK_ID:
            self.filled_positions[z, y, x] = 1
            self.remaining_blocks -= 1
        self.structure[z, y, x] = token

    def is_complete(self):
        """Check if this alternative is complete."""
        return self.remaining_blocks <= 0 or self.remaining_iterations <= 0

    def clone_for_alternative(self, new_id: int, new_structure: torch.Tensor):
        """Create a clone of this alternative with a new ID."""
        return StructureAlternative(
            alternative_id=new_id,
            structure=new_structure,
            filled_positions=self.filled_positions.clone(),
            remaining_iterations=self.remaining_iterations,
            remaining_blocks=self.remaining_blocks,
            valid_positions=self.valid_positions.copy(),
        )


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
        feed_forward_dim: int | None = None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.block_str_mapping = block_str_mapping
        self.max_structure_size = max_structure_size
        self.max_sequence_length = max_structure_size**3
        self.model_dim = model_dim
        self.embedding_dim = embedding_dim or model_dim

        feed_forward_dim = feed_forward_dim or model_dim * 4

        # Input
        self.embedding = nn.Embedding(num_classes, self.embedding_dim)
        if self.embedding_dim != model_dim:
            self.embedding_projection = nn.Linear(self.embedding_dim, model_dim)
        self.register_buffer("positions", torch.arange(self.max_sequence_length))
        self.positional_embedding = nn.Embedding(self.max_sequence_length, model_dim)
        self.embedding_dropout = nn.Dropout(embedding_dropout)

        # Transformer
        decoder_layer = SelfAttentionDecoderLayer(
            model_dim,
            num_heads,
            dim_feedforward=feed_forward_dim,
            dropout=decoder_dropout,
            batch_first=True,
        )
        self.decoder = SelfAttentionDecoder(decoder_layer, num_layers)

        # Output
        if self.embedding_dim != model_dim:
            self.output_projection = nn.Linear(model_dim, self.embedding_dim)
        self.output_layer = nn.Linear(self.embedding_dim, num_classes)
        self.output_layer.weight = self.embedding.weight

        self._init_weights()

        self._flattening_orders = self._create_flattening_orders()

    def _create_flattening_order(self, size: int) -> torch.Tensor:
        """
        Return a 1D LongTensor of length size^3 containing the "center-out" ordering
        for a cubic volume of shape [size, size, size].
        The list is sorted by L∞ distance from center, then lexicographically within each shell.
        Each entry is an index in standard (row-major) flattening order.
        """
        c = size // 2
        coords = []

        # Gather all coordinates, calculate distance
        for x in range(size):
            for y in range(size):
                for z in range(size):
                    d = max(abs(x - c), abs(y - c), abs(z - c))
                    coords.append((d, x, y, z))

        # Sort by (distance, x, y, z)
        coords.sort()

        # Convert (x, y, z) → single integer index i = x*(size^2) + y*size + z
        # in row-major ordering
        flatten_idx = [(x * (size * size)) + (y * size) + z for (_, x, y, z) in coords]

        # Make it a LongTensor
        flatten_idx = torch.tensor(flatten_idx, dtype=torch.long)
        return flatten_idx

    def _create_flattening_orders(self) -> dict:
        """Generate ordering tensors for every allowed odd cube size."""
        orders = {}
        # Generate orders for odd sizes from 3 up to max_structure_size
        for size in range(3, self.max_structure_size + 1, 2):
            flatten_idx = self._create_flattening_order(size)
            self.register_buffer(f"flatten_idx_{size}", flatten_idx)
            orders[size] = flatten_idx
        return orders

    def _flatten_structure(self, structure: torch.Tensor) -> torch.Tensor:
        """Flatten a 3D structure tensor to a 1D tensor."""
        batch = structure.size(0)
        size = structure.size(-1)
        flatten_idx = getattr(self, f"flatten_idx_{size}")
        structure_1d = structure.view(batch, -1)
        flatten_idx_batched = flatten_idx.unsqueeze(0).expand(batch, -1)
        structure_flat = structure_1d.gather(1, flatten_idx_batched)
        return structure_flat

    def _unflatten_structure(self, structure_flat: torch.Tensor) -> torch.Tensor:
        """Unflatten a 1D structure tensor to a 3D tensor."""
        batch, channels, length = structure_flat.size()
        size = round(length ** (1 / 3))
        flatten_idx = getattr(self, f"flatten_idx_{size}")
        flatten_idx_batched = flatten_idx.view(1, 1, -1).expand(batch, channels, -1)
        out_2d = structure_flat.new_zeros(batch, channels, size**3)
        out_2d.scatter_(dim=2, index=flatten_idx_batched, src=structure_flat)
        out = out_2d.view(batch, channels, size, size, size)
        return out

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

    def forward(self, structure: torch.Tensor) -> torch.Tensor:
        # Flatten the structure
        structure = self._flatten_structure(structure)

        # Embed the sequence
        output_seq = self.embedding(structure)

        # Project the embedding if necessary
        if self.embedding_dim != self.model_dim:
            output_seq = self.embedding_projection(output_seq)

        # Use cached positions tensor, sliced to current sequence length
        output_seq = output_seq + self.positional_embedding(
            self.positions[: structure.size(1)]
        )

        # Add dropout
        output_seq = self.embedding_dropout(output_seq)

        # Run the transformer
        output = self.decoder(tgt=output_seq)

        # Project back to embedding dimension if necessary
        if self.embedding_dim != self.model_dim:
            output = self.output_projection(output)

        # Run the output layer
        output = self.output_layer(output)

        # Reshape so the class logits are before the sequence dimension
        output = output.transpose(1, 2)

        # Unflatten the structure
        structure = self._unflatten_structure(output)

        return structure

    def generate_model_card(self) -> ModelCard:
        return ModelCard.from_template(
            card_data=self._hub_mixin_info.model_card_data,
            template_str=MODEL_CARD_TEMPLATE,
            num_classes=self.num_classes,
            max_sequence_length=self.max_sequence_length,
            embedding_dim=self.embedding.embedding_dim,
            attention_heads=self.decoder.layers[0].self_attn.num_heads,
            transformer_layers=len(self.decoder.layers),
            parameters=f"{sum(p.numel() for p in self.parameters()):,}",
        )

    @staticmethod
    def generate_neighbor_mask(tensor: torch.Tensor) -> torch.Tensor:
        """Generates a mask indicating if an element has a neighbor that is not air."""
        # Add a temporary channel dimension
        tensor = tensor.unsqueeze(1)

        kernel = torch.ones((1, 1, 3, 3, 3), dtype=tensor.dtype, device=tensor.device)
        kernel[0, 0, 1, 1, 1] = 0  # Ignore the central element

        # Create a mask of elements greater than AIR_BLOCK_ID
        non_air = tensor > AIR_BLOCK_ID

        # Convolve to count neighbors that are greater than AIR_BLOCK_ID
        has_non_air_neighbors = (
            F.conv3d(non_air.float(), kernel.float(), padding=1) >= 1
        )

        # Return the mask
        return has_non_air_neighbors.squeeze(1)

    @staticmethod
    def _get_valid_positions(
        structure: torch.Tensor, filled_positions: torch.Tensor
    ) -> list[tuple[int, int, int]]:
        """Get ordered list of valid positions to fill, from center outward."""
        # Limit structure to positions that have been filled by the model
        mask_structure = structure * filled_positions
        # Generate mask of valid next elements
        mask_structure = mask_structure.unsqueeze(0)
        mask = TransformerMinecraftStructureGenerator.generate_neighbor_mask(
            mask_structure
        )
        mask_structure = mask_structure.squeeze(0)
        mask = mask.squeeze(0)
        # Limit to positions that were originally masked
        mask = mask & (structure == MASK_BLOCK_ID)

        if not mask.any():
            return []

        # Get positions that need filling
        valid_positions = mask.nonzero()

        # Calculate center coordinates based on structure dimensions
        center = torch.tensor(
            [d // 2 for d in structure.shape],
            device=valid_positions.device,
            dtype=torch.float,
        )

        # Calculate distances from center
        distances = torch.norm(valid_positions.float() - center, dim=1)

        # Return positions ordered by distance from center
        return valid_positions[torch.argsort(distances)].tolist()

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
            # Add batch dimension if necessary
            if structure.dim() == 3:
                structure = structure.unsqueeze(0)

            # Run the model
            logits = self.forward(structure)

            # Select most likely token or sample from distribution
            if use_greedy:
                predictions = torch.argmax(logits, dim=1)
            else:
                probabilities = F.softmax(logits / temperature, dim=1)
                predictions = torch.multinomial(probabilities, num_samples=1)

            # Copy original structure and replace masked positions with predictions
            result = structure.clone()
            result[result == 0] = predictions[result == 0]

            return result

    def fill_structure(
        self,
        structure: torch.Tensor,
        temperature: float,
        start_radius: int,
        max_iterations: int,
        max_blocks: int,
        max_alternatives: int,
        min_alternative_probability: float,
    ):
        """Fill a structure with multiple alternatives."""
        assert structure.dim() == 3, "Structure must have 3 dimensions"
        assert structure.size(0) == structure.size(1) == structure.size(2), (
            f"Structure must be cubic, got shape {structure.shape}"
        )
        assert structure.size(0) <= self.max_structure_size, (
            f"Structure size {structure.size(0)} exceeds maximum size {self.max_structure_size}"
        )

        # Initialize tensor to track filled positions
        filled_positions = torch.zeros_like(structure, dtype=torch.bool)

        # Fill the center up to the start radius
        z_mid, y_mid, x_mid = (
            structure.shape[0] // 2,
            structure.shape[1] // 2,
            structure.shape[2] // 2,
        )
        filled_positions[
            z_mid - start_radius : z_mid + start_radius + 1,
            y_mid - start_radius : y_mid + start_radius + 1,
            x_mid - start_radius : x_mid + start_radius + 1,
        ] = 1

        # Create batch of structures for all possible alternatives
        batched_structure = (
            structure.unsqueeze(0).expand(max_alternatives, -1, -1, -1).clone()
        )

        # Create the first alternative
        alternatives = [
            StructureAlternative(
                alternative_id=0,
                structure=batched_structure[0],
                filled_positions=filled_positions,
                remaining_iterations=max_iterations,
                remaining_blocks=max_blocks,
            )
        ]

        # Track the next available alternative ID
        next_alternative_id = 1

        # Process alternatives until none remain
        with torch.no_grad(), torch.autocast(device_type="cuda"):
            while alternatives:
                next_alternatives = []

                # Forward pass for all current alternatives at once
                logits = self.forward(batched_structure)

                # Process each alternative in the current list
                for alternative in alternatives:
                    # Skip completed alternatives
                    if alternative.is_complete():
                        continue

                    # Get the next valid positions if needed
                    alternative.get_valid_positions()
                    if not alternative.valid_positions:
                        continue

                    next_alternatives.append(alternative)

                    # Process the next position
                    pos = alternative.valid_positions.pop(0)
                    z, y, x = pos

                    # Get logits for this position
                    logits_for_position = logits[alternative.id, :, z, y, x]

                    # Sample from the distribution
                    probabilities = F.softmax(logits_for_position / temperature, dim=-1)
                    predicted_token = torch.multinomial(
                        probabilities, num_samples=1
                    ).item()

                    if next_alternative_id < max_alternatives:
                        # Find potential alternatives
                        probabilities[predicted_token] = 0
                        top_probs, top_indices = torch.topk(
                            probabilities, max_alternatives - next_alternative_id
                        )

                        for prob, token in zip(top_probs, top_indices):
                            if (
                                prob < min_alternative_probability
                                or next_alternative_id >= max_alternatives
                            ):
                                break

                            # Create a new alternative with this token
                            batched_structure[next_alternative_id] = (
                                alternative.structure.clone()
                            )
                            new_alternative = alternative.clone_for_alternative(
                                next_alternative_id,
                                batched_structure[next_alternative_id],
                            )
                            alternative_token = token.item()

                            # Yield the result
                            yield (
                                new_alternative.id,
                                alternative.id,
                                alternative_token,
                                z,
                                y,
                                x,
                            )

                            # Place the alternative block
                            new_alternative.place_block(alternative_token, pos)

                            # Add the new alternative to the processing list
                            next_alternatives.append(new_alternative)
                            next_alternative_id += 1

                    # Yield the result
                    yield alternative.id, 0, predicted_token, z, y, x

                    # Place the block in the current alternative
                    alternative.place_block(predicted_token, pos)

                # Update the alternatives list for the next round
                alternatives = next_alternatives
