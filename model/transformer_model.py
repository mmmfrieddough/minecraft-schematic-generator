import torch
import torch.nn.functional as F
from torch import nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super(PositionalEncoding, self).__init__()
        self.positional_embedding = nn.Embedding(max_len, d_model)
        self.register_buffer('positions', torch.arange(
            max_len).expand((1, max_len)))

    def forward(self, x):
        x = x + self.positional_embedding(self.positions[:, :x.size(1)])
        return x


class TransformerMinecraftStructureGenerator(nn.Module):
    def __init__(self, num_classes, max_sequence_length, embedding_dropout, model_dim, num_heads, num_layers, decoder_dropout):
        super().__init__()
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length

        self.embedding = nn.Embedding(num_classes, model_dim)
        self.positional_encoding = PositionalEncoding(
            model_dim, max_sequence_length)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        decoder_layer = nn.TransformerDecoderLayer(
            model_dim, num_heads, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.output_layer = nn.Linear(model_dim, num_classes)

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
