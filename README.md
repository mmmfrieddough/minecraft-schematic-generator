# Minecraft Structure Generator

[![Build and Release](https://github.com/mmmfrieddough/minecraft-schematic-generator/actions/workflows/build-and-release.yml/badge.svg)](https://github.com/mmmfrieddough/minecraft-schematic-generator/actions/workflows/build-and-release.yml)
[![Lint and Format](https://github.com/mmmfrieddough/minecraft-schematic-generator/actions/workflows/lint-and-format.yml/badge.svg)](https://github.com/mmmfrieddough/minecraft-schematic-generator/actions/workflows/lint-and-format.yml)
[![Model](https://img.shields.io/badge/🤗_Model-Hugging_Face-yellow)](https://huggingface.co/mmmfrieddough/minecraft-schematic-generator)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

An ML-powered structure generator for Minecraft that uses a transformer architecture to complete partially built structures. This model serves as the backbone for [Craftpilot](https://github.com/mmmfrieddough/craftpilot), enabling intelligent structure completion in-game.

## Overview

This project uses a decoder-only transformer model to generate Minecraft structures by predicting missing blocks in a given schematic. The model understands spatial relationships and building patterns from millions of player-created structures, allowing it to complete partial builds in a way that maintains architectural coherence and style.

### Key Features

- Processes 11x11x11 3D structure inputs
- Handles ~14,000 unique block types and states
- Trained on 8M+ samples (10B+ tokens) from real player builds
- Autoregressive generation starting from existing blocks
- Built with PyTorch + Lightning for scalable training

## How It Works

### Input Processing

1. Takes a 3D Minecraft schematic (11x11x11)
2. Converts blocks to tokens (~14K vocabulary)
3. Marks areas to be filled with token 0
4. Flattens the 3D structure for transformer processing

### Model Architecture

- Decoder-only transformer (~80M parameters)
- Learned positional encodings for 3D spatial awareness
- Token embeddings for block representation
- Autoregressive generation focusing on blocks adjacent to existing structures

### Generation Process

1. Identifies empty spaces (token 0) adjacent to existing blocks
2. Predicts one block at a time, building outward from existing structure
3. Continues until all relevant empty spaces are filled
4. Maintains structural integrity and style consistency

## Data Collection Pipeline

Due to the large size of Minecraft worlds and the often sparse nature of player-built structures, collecting samples is a challenging task. The process must be executed efficiently to gather a large number of samples and is divided into three stages:

### 1. Chunk Analysis

Directly examining every position for a sample is impractical. The process leverages the way Minecraft world files are saved. The world is divided into 16x16 chunks, each containing a "palette" of unique block types. The system compiles a list of chunks that include block types of interest, typically those not found in a naturally generated Minecraft world. The presence of these blocks indicates player activity and potential structures of interest. For custom maps, this list may need manual updates.

### 2. Sample Selection

The process examines previously selected chunks for sample positions. From each starting position, it assesses the area covered by a potential sample. A broader set of block types is considered at this stage - including blocks that might appear in natural structures (like villages) - since the analysis is already focused on likely player-built areas. The system ensures a sufficient number of these blocks are present, along with a variety of unique ones. While some overlap between samples is acceptable, they should not be adjacent, so a minimum distance threshold is enforced.

### 3. Data Processing

Finally, the system processes all identified positions and saves each sample as a schematic file.

## Usage

### Pre-built Binaries

Ready-to-use executables are available in the releases section, providing an easy way to run the model server.

### Running the Server

The server provides a FastAPI interface for structure generation:

```bash
./minecraft-schematic-generator [options]
```

Server will be available at http://localhost:8000

## Installation

Before installing the project dependencies, ensure you have the correct version of PyTorch installed for your system. Visit the [PyTorch installation page](https://pytorch.org/get-started/locally/) to select the appropriate version based on your operating system, package manager, and CUDA version.

Once PyTorch is installed, you can install the remaining dependencies with:

```bash
pip install -e .[dev,test,docs]
```

### Available Scripts

- `train.py` - Main training utility
- `sample.py` - World sampling utility
- `prepare_data.py` - Dataset compilation
- `inference.ipynb` - Interactive model testing

### Training Workflow

1. Assemble a selection of worlds to get samples from.
2. Collect samples from the worlds.
3. Compile the sample schematics into a dataset.
4. Run training script.

## Resources

- [Model on Huggingface](https://huggingface.co/mmmfrieddough/minecraft-schematic-generator)
- [Craftpilot Mod](https://github.com/mmmfrieddough/craftpilot)

## Contributing

Contributions are welcome! Please feel free to submit a pull request or open an issue if you have any suggestions or improvements.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
