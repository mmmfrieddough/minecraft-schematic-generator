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
                # print('-------------------')
                # print(str(block), token)

            return token

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
                # Convert the token back to a block
                block = block_token_mapper.token_to_block(block)

                response = Block(value=str(block), position=(z, y, x))
                if id not in test:
                    test.append(id)
                    # print('-------------------')
                    # print(test0, test1, test2, test3, test4)

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
