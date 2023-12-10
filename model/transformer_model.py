import torch
import torch.nn.functional as F
from torch import nn
from transformers import DistilBertModel, DistilBertTokenizer


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
    def __init__(self, num_classes, max_sequence_length, embedding_dim, embedding_dropout, num_heads, num_layers, decoder_dropout, freeze_encoder=False):
        super().__init__()
        self.num_classes = num_classes
        self.max_sequence_length = max_sequence_length

        self.tokenizer = DistilBertTokenizer.from_pretrained(
            'distilbert-base-uncased')
        self.encoder = DistilBertModel.from_pretrained(
            'distilbert-base-uncased')

        # Freeze DistilBERT's weights if requested
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.embedding = nn.Embedding(num_classes, embedding_dim)
        self.positional_encoding = PositionalEncoding(
            embedding_dim, max_sequence_length)
        self.embedding_dropout = nn.Dropout(embedding_dropout)
        self.embedding_upsample = nn.Linear(
            embedding_dim, self.encoder.config.dim)
        decoder_layer = nn.TransformerDecoderLayer(
            self.encoder.config.dim, num_heads, dropout=decoder_dropout)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers)
        self.embedding_downsample = nn.Linear(
            self.encoder.config.dim, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, num_classes)

    def encode_prompt(self, prompt_tokens: torch.Tensor) -> torch.Tensor:
        prompt_encodings = self.encoder(**prompt_tokens)[0]
        return prompt_encodings

    def predict_next_blocks(self, prompt_encodings: torch.Tensor, sequence: torch.Tensor) -> torch.Tensor:
        # Embed the sequence
        output_seq = self.embedding(sequence)

        # Add positional encoding
        output_seq = self.positional_encoding(output_seq)

        # Add dropout
        output_seq = self.embedding_dropout(output_seq)

        # Upsample the embedding
        output_seq = self.embedding_upsample(output_seq)

        # Reshape the prompt and sequence so the sequence is the first dimension
        prompt_encodings = prompt_encodings.transpose(0, 1)
        output_seq = output_seq.transpose(0, 1)

        # Run the transformer
        mask = nn.Transformer.generate_square_subsequent_mask(
            output_seq.size(0)).to(output_seq.device)
        output = self.decoder(
            tgt=output_seq, memory=prompt_encodings, tgt_mask=mask)

        # Reshape back to the batch first format
        output = output.transpose(0, 1)

        # Downsample the embedding
        output = self.embedding_downsample(output)

        # Run the output layer
        output = self.output_layer(output)

        # Reshape so the class logits are before the sequence dimension
        output = output.transpose(1, 2)

        return output

    def forward(self, prompt: str, structure: torch.Tensor) -> torch.Tensor:
        # Encode the prompt
        prompt_tokens = self.tokenizer(
            prompt, return_tensors='pt', padding=True, truncation=True).to(self.encoder.device)
        prompt_encodings = self.encode_prompt(prompt_tokens)

        # Flatten the 3D structure for processing
        batch_size, depth, height, width = structure.size()
        structure_flat = structure.reshape(batch_size, -1)
        # Add the start token to the beginning of the sequence and remove the last token
        structure_flat = torch.cat((torch.zeros(
            batch_size, 1, dtype=structure.dtype, device=structure.device), structure_flat[:, :-1]), dim=1)

        # Predict the next blocks
        output = self.predict_next_blocks(prompt_encodings, structure_flat)

        # Reshape the output back to the original 3D structure
        output = output.view(batch_size, self.num_classes,
                             depth, height, width)

        return output

    def generate_structure(self, prompt: str, temperature: float) -> torch.Tensor:
        self.eval()
        # Initialize the current sequence tensor with the appropriate shape and device
        current_sequence = torch.full(
            (1, self.max_sequence_length + 1), 0, dtype=torch.long, device=self.encoder.device)

        with torch.no_grad():
            prompt_tokens = self.tokenizer(
                prompt, return_tensors='pt', padding=True, truncation=True).to(self.encoder.device)
            prompt_encodings = self.encode_prompt(prompt_tokens)

            # Loop through each position
            for i in range(1, self.max_sequence_length + 1):
                # Run the model on the current structure
                current_subsequence = current_sequence[:, :i]
                output = self.predict_next_blocks(
                    prompt_encodings, current_subsequence)

                # Apply temperature scaling to logits
                scaled_logits = output[0, :, -1] / temperature
                probabilities = F.softmax(scaled_logits, dim=-1)

                # Sample from the probability distribution
                block = torch.multinomial(probabilities, num_samples=1).item()

                # Add the block to the structure
                current_sequence[0, i] = block

        current_sequence = current_sequence[0, 1:]
        structure = current_sequence.view(8, 8, 8)
        return structure
