import os

import h5py
import numpy as np

from common.file_paths import (EMBEDDINGS_DIR, TARGET_TENSORS_DIR,
                               TRAINING_DATA_DIR)


def prepare_data():
    """
    Bundles together all the embeddings and schematics into a single file to be used for training.
    """
    os.makedirs(TRAINING_DATA_DIR, exist_ok=True)

    # Create an HDF5 file
    with h5py.File(os.path.join(TRAINING_DATA_DIR, 'data.h5'), 'w') as hf:
        for i, file in enumerate(os.listdir(EMBEDDINGS_DIR)):
            # Load the input embedding
            input_embedding = np.load(os.path.join(EMBEDDINGS_DIR, file))

            # Load the corresponding target tensor
            target_tensor = np.load(os.path.join(TARGET_TENSORS_DIR, file))

            # Create a group for each sample
            group = hf.create_group(f'sample_{i}')

            # Create a dataset for input embeddings within the group
            group.create_dataset('input_embedding', data=input_embedding)

            # Create a dataset for target tensor within the group
            group.create_dataset('target_tensor', data=target_tensor)

            # Break early for debugging
            if i == 10:
                break
