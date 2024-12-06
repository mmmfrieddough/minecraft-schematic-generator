import time
import traceback
from typing import List, Tuple

import amulet.api.block
import PyMCTranslate
import schempy
import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from converter import BlockTokenMapper
from data_preparer import clean_block_properties
from modules import LightningTransformerMinecraftStructureGenerator
from schempy.components import BlockPalette


class Request(BaseModel):
    platform: str
    version_number: int
    temperature: float
    structure: List[List[List[str]]]


class Block(BaseModel):
    value: str
    position: Tuple[int, int, int]


block_token_mapper = BlockTokenMapper()

model_version = 12
output_dir = 'schematic_viewer/public/schematics/'
checkpoint_path = f'lightning_logs/center_data/version_{model_version}/checkpoints/last.ckpt'
model = LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
    checkpoint_path)
model.eval()

app = FastAPI()


@app.post("/complete-structure/")
async def complete_structure(input: Request):
    print(input)
    try:
        translation_manager = PyMCTranslate.new_translation_manager()
        source_version = translation_manager.get_version(
            input.platform, input.version_number)
        internal_version = translation_manager.get_version('java', 3578)

        test = []

        def convert(block_str: str):
            val1 = block_str

            # Create a block object
            block = amulet.api.block.Block.from_string_blockstate(block_str)
            val2 = block

            # Convert to universal format
            block, _, _ = source_version.block.to_universal(block)
            val3 = block

            # Convert to target format
            block, _, _ = internal_version.block.from_universal(block)
            val4 = block

            # Convert to schempy format
            block = BlockPalette._parse_block_str(block.blockstate)
            val5 = block

            # Clean block properties
            clean_block_properties(block)
            val6 = block

            # Convert to token
            val7 = block_token_mapper.block_to_token(block)

            return block_token_mapper.block_to_token(block)

        # Convert the string values to IDs
        input_structure_ids = [
            [[convert(block_str) for block_str in y] for y in z] for z in input.structure]

        # Convert the input data to a torch tensor
        input_tensor = torch.tensor(input_structure_ids)

        # Mask out the air blocks
        input_tensor[input_tensor == 1] = 0

        test = []

        # Generate the structure
        def generate():
            for block, z, y, x in model.fill_structure(input_tensor, input.temperature):
                test0 = block
                # Convert the token back to a block
                block = block_token_mapper.token_to_block(block)
                test1 = block

                # Convert to Amulet format
                block = amulet.api.block.Block.from_string_blockstate(
                    str(block))
                test2 = block

                # Convert to universal format
                block, _, _ = internal_version.block.to_universal(block)
                test3 = block

                # Convert to source format
                block, _, _ = source_version.block.from_universal(block)
                test4 = block

                if 'stairs' in block.blockstate:
                    print(test0)
                    print(test1)
                    print(test2)
                    print(test3)
                    print(test4)
                    print('-----------------')

                id = block.blockstate
                response = Block(value=id, position=(z, y, x))
                if id not in test:
                    test.append(id)
                    print(id)
                    print(response.model_dump_json())
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
