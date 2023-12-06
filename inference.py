from datetime import datetime
from pathlib import Path

import torch

from converter import SchematicArrayConverter
from modules import LightningTransformerMinecraftStructureGenerator

model_version = 15
output_dir = 'schematic_viewer/public/schematics/'
checkpoint_path = f'lightning_logs/minecraft_structure_generator/version_{model_version}/checkpoints/last.ckpt'
# checkpoint_path = f'version_{model_version}/checkpoints/last.ckpt'
model = LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
    checkpoint_path)
model.eval()

converter = SchematicArrayConverter()

# Loop to take user input and perform inference
while True:
    prompt = input("Enter your text input (or type 'exit' to stop): ")
    if prompt.lower() == 'exit':
        break

    # Perform inference
    with torch.no_grad():
        print("Performing inference...")
        output = model.generate(prompt, True)
        output = output.cpu().numpy()

    print("Converting output array to schematic...")
    schematic = converter.array_to_schematic(output)
    schematic.name = prompt
    print("Saving schematic to file...")
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    unique_filename = f'inference_{timestamp}.schem'
    path = Path(output_dir) / unique_filename
    schematic.save_to_file(path)
