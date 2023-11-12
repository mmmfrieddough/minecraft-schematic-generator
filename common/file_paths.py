import os

# Define the base directory for your data
BASE_DIR = 'data/'

# Define the directories for each type of data
METADATA_DIR = os.path.join(BASE_DIR, 'metadata/')
SCHEMATICS_DIR = os.path.join(BASE_DIR, 'schematics/')
DESCRIPTIONS_DIR = os.path.join(BASE_DIR, 'descriptions/')
EMBEDDINGS_DIR = os.path.join(BASE_DIR, 'embeddings/')
TARGET_TENSORS_DIR = os.path.join(BASE_DIR, 'target_tensors/')
TRAINING_DATA_DIR = os.path.join(BASE_DIR, 'training_data/')
