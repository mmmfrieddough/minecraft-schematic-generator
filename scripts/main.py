from pathlib import Path
import traceback
from typing import List, Tuple

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from schempy.components import BlockPalette
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from minecraft_schematic_generator.converter import BlockTokenMapper
from minecraft_schematic_generator.converter.converter import SchematicArrayConverter
from minecraft_schematic_generator.data_preparer import clean_block_properties
from minecraft_schematic_generator.modules import LightningTransformerMinecraftStructureGenerator


class Request(BaseModel):
    platform: str
    version_number: int
    temperature: float
    start_radius: int
    max_iterations: int
    max_blocks: int
    air_probability_iteration_scaling: float
    structure: List[List[List[str]]]


class Block(BaseModel):
    block_state: str
    z: int
    y: int
    x: int


block_token_mapper = BlockTokenMapper()

model_version = 12
output_dir = 'schematic_viewer/public/schematics/'
checkpoint_path = f'lightning_logs/center_data/version_{model_version}/checkpoints/last.ckpt'
model = LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
    checkpoint_path)
model.eval()

app = FastAPI()


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    print(f"Validation error details: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()}
    )


@app.post("/complete-structure/")
async def complete_structure(input: Request):
    try:
        test = []

        def convert(block_str: str):
            # Convert to schempy format
            block = BlockPalette._parse_block_str(block_str)

            # Clean block properties
            clean_block_properties(block)

            # Convert to token
            token = block_token_mapper.block_to_token(block)

            if token not in test:
                test.append(token)
                print(f"Token: {token}, Block: {block_str}")

            return token

        # Convert the string values to IDs
        input_structure_ids = [
            [[convert(block_str) for block_str in y] for y in z] for z in input.structure]

        # Convert the input data to a torch tensor
        input_tensor = torch.tensor(input_structure_ids)

        # Save as a schematic for debugging
        # try:
        #     schematic_array_converter = SchematicArrayConverter()
        #     schematic = schematic_array_converter.array_to_schematic(
        #         input_tensor)
        #     schematic.name = 'Test'
        #     schematic.save_to_file(Path('debug.schem'), 2)
        # except Exception as e:
        #     print(f"Failed to save schematic: {e}")

        # Mask out the air blocks
        input_tensor[input_tensor == 1] = 0

        # Generate the structure
        def generate():
            for block, z, y, x in model.fill_structure(input_tensor, input.temperature, input.start_radius, input.max_iterations, input.max_blocks, input.air_probability_iteration_scaling):
                # Convert the token back to a block
                block = block_token_mapper.token_to_block(block)

                response = Block(block_state=str(block), z=z, y=y, x=x)

                yield response.model_dump_json() + "\n"

        # Return the response as a stream
        return StreamingResponse(generate(), media_type="application/x-ndjson")

    except Exception as e:
        traceback_str = traceback.format_exc()
        print(traceback_str)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=5)
