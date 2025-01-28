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
        embedding_dim=None,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length
        self.embedding_dim = embedding_dim or model_dim
        self.model_dim = model_dim

        # Input
        self.embedding = nn.Embedding(num_classes, self.embedding_dim)
        if self.embedding_dim != model_dim:
            self.embedding_projection = nn.Linear(self.embedding_dim, model_dim)
        self.register_buffer("positions", torch.arange(max_sequence_length))
        self.positional_embedding = nn.Embedding(max_sequence_length, model_dim)
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
