import numpy as np
import torch
from huggingface_hub import ModelCard, PyTorchModelHubMixin
from torch import nn

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


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model

        # Calculate the dimensions of the cubic grid
        cube_side = round(max_len ** (1 / 3))

        # Ensure the sequence can form a perfect cube
        assert cube_side**3 == max_len, "max_len must be a perfect cube"

        # Create the positional embedding layer
        self.positional_embedding = nn.Embedding(max_len, d_model)

        # Initialize the positional embeddings with 3D spatial encoding
        self.positional_embedding.weight.data.copy_(
            self._3d_spatial_encoding(cube_side)
        )

        self.register_buffer("positions", torch.arange(max_len).expand((1, max_len)))

    def _3d_spatial_encoding(self, cube_side):
        # Generate a 3D grid
        z_grid, y_grid, x_grid = torch.meshgrid(
            torch.linspace(0, 1, cube_side),
            torch.linspace(0, 1, cube_side),
            torch.linspace(0, 1, cube_side),
            indexing="ij",
        )

        # Flatten the grid
        grid = torch.stack((z_grid, y_grid, x_grid), dim=-1).view(-1, 3)

        # Encode each dimension into higher dimensions
        div_term = torch.exp(
            torch.arange(0, self.d_model // 3, 2)
            * -(np.log(10000.0) / (self.d_model // 3))
        )
        pos_encoding = torch.zeros((grid.shape[0], self.d_model))
        for i in range(3):  # For each of the z, y, x dimensions
            pos_encoding[:, i::6] = torch.sin(
                grid[:, i : i + 1] * div_term
            )  # Sine for even indices
            pos_encoding[:, (i + 1) :: 6] = torch.cos(
                grid[:, i : i + 1] * div_term
            )  # Cosine for odd indices

        return pos_encoding

    def forward(self, x):
        x = x + self.positional_embedding(self.positions[:, : x.size(1)])
        return x


class TransformerMinecraftStructureGenerator(nn.Module, PyTorchModelHubMixin):
    def __init__(
        self,
        num_classes,
        max_sequence_length,
        embedding_dropout,
        model_dim,
        num_heads,
        num_layers,
        decoder_dropout,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length

        self.embedding = nn.Embedding(num_classes, model_dim)
        self.positional_encoding = PositionalEncoding(model_dim, max_sequence_length)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            model_dim, num_heads, dropout=decoder_dropout
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_layer = nn.Linear(model_dim, num_classes)

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.embedding.weight)
        for layer in self.decoder.layers:
            nn.init.xavier_uniform_(layer.self_attn.in_proj_weight)
            nn.init.xavier_uniform_(layer.self_attn.out_proj.weight)
            nn.init.xavier_uniform_(layer.linear1.weight)
            nn.init.xavier_uniform_(layer.linear2.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, structure_flat: torch.Tensor) -> torch.Tensor:
        # Embed the sequence
        output_seq = self.embedding(structure_flat)

        # Add positional encoding
        output_seq = self.positional_encoding(output_seq)

        # Add dropout
        output_seq = self.embedding_dropout(output_seq)

        # Reshape the sequence so the batch is the first dimension
        output_seq = output_seq.transpose(0, 1)

        # Run the transformer
        output = self.decoder(tgt=output_seq, memory=output_seq)

        # Reshape back to the batch first format
        output = output.transpose(0, 1)

        # Run the output layer
        output = self.output_layer(output)

        # Reshape so the class logits are before the sequence dimension
        output = output.transpose(1, 2)

        return output

    def generate_model_card(self):
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
