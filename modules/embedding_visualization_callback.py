import numpy as np
import pandas as pd
import plotly.express as px
import torch
from lightning.pytorch.callbacks import Callback
from sklearn.manifold import TSNE

from converter import BlockTokenMapper


class EmbeddingVisualizationCallback(Callback):
    def __init__(self, data_module, frequency=1):
        """
        Args:
            frequency (int): Frequency (in epochs) to perform visualization.
        """
        self.frequency = frequency
        self.mapper = BlockTokenMapper()
        self.data_module = data_module
        self.train_tokens = set()

    def setup(self, trainer, pl_module, stage) -> None:
        # Get the train dataloader
        train_dataloader = self.data_module.train_dataloader()

        # Get the train tokens
        for batch in train_dataloader:
            _, labels, _ = batch
            labels = labels.view(-1).tolist()
            self.train_tokens.update(labels)

    def on_train_epoch_end(self, trainer, pl_module):
        # Perform visualization at the end of an epoch
        if (trainer.current_epoch + 1) % self.frequency == 0:
            self.visualize_embeddings(pl_module)

    def generate_samples(self, pl_module):
        # Select a sample
        dataloader = self.data_module.train_dataloader()
        samples = next(iter(dataloader))
        features, labels, _ = samples

        # Move the sample to the same device as the model
        features = features.to(pl_module.device)

        # Generate a sample using the model
        predictions = pl_module(features)

        # Pick the first sample from the batch
        predictions = predictions[0]
        labels = labels[0]

        # Pick 10 random samples from the 64x64x64 3D space for predictions and labels
        num_samples = 50
        z_indices = torch.randint(0, 64, (num_samples,))
        y_indices = torch.randint(0, 64, (num_samples,))
        x_indices = torch.randint(0, 64, (num_samples,))

        # Gather the samples using the 3D indices
        predictions_samples = predictions[:, z_indices, y_indices, x_indices]
        labels_samples = labels[z_indices, y_indices, x_indices]

        # Convert the tensors to lists
        list_of_tensors = torch.chunk(
            predictions_samples, chunks=num_samples, dim=1)
        predictions = [t.squeeze().cpu().detach().numpy()
                       for t in list_of_tensors]
        labels = labels_samples.tolist()

        return predictions, labels

    def visualize_embeddings(self, pl_module):
        predictions, labels = self.generate_samples(pl_module)
        num_embeddings = len(self.mapper.token_to_block_id_map)
        embedding_layer = pl_module.embedding_layer
        embeddings = [embedding_layer(torch.tensor([i]).to(pl_module.device)).cpu(
        ).detach().numpy().squeeze() for i in range(1, num_embeddings)]
        embeddings += predictions
        classes = [self.mapper.token_to_block_id_map[i]
                   for i in range(1, num_embeddings)]
        classes += [self.mapper.token_to_block_id_map[i] for i in labels]
        included = [i in self.train_tokens if self.mapper.token_to_block_id_map[i]
                    != 'minecraft:air' else 'air' for i in range(1, num_embeddings)]
        included += ['sample air' if self.mapper.token_to_block_id_map[i]
                     == 'minecraft:air' else 'sample other' for i in labels]
        embeddings_array = np.array(embeddings)

        perplexity_value = min(30, len(embeddings_array) - 1)

        # Use t-SNE to reduce the dimensionality
        tsne = TSNE(n_components=2, random_state=0,
                    perplexity=perplexity_value)
        reduced_embeddings = tsne.fit_transform(embeddings_array)

        # Create a DataFrame for Plotly
        df = pd.DataFrame(reduced_embeddings, columns=['x', 'y'])
        df['class'] = classes
        df['In training set'] = included

        # Create an interactive scatter plot
        fig = px.scatter(df, x='x', y='y', hover_data=[
                         'class'], color='In training set')
        fig.update_layout(title='t-SNE visualization of block embeddings',
                          xaxis_title='t-SNE dimension 1', yaxis_title='t-SNE dimension 2')

        # Update the layout for dark mode
        fig.update_layout(
            template="plotly_dark",  # Use the built-in dark mode template
            plot_bgcolor='rgba(0,0,0,0)',  # Set plot background to black
            # Set the overall background to black
            paper_bgcolor='rgba(0,0,0,0)',
            font_color="white",  # Set text color to white
        )

        fig.show()
