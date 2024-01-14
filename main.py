import time
from typing import List, Tuple

import torch
from fastapi import FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel

from modules import LightningTransformerMinecraftStructureGenerator
import schempy
from converter import BlockTokenMapper


class Request(BaseModel):
    temperature: float = 0.7
    structure: List[List[List[str]]]


class Block(BaseModel):
    value: str
    position: Tuple[int, int, int]


block_token_mapper = BlockTokenMapper()

model_version = 67
output_dir = 'schematic_viewer/public/schematics/'
checkpoint_path = f'lightning_logs/version_{model_version}/checkpoints/last.ckpt'
model = LightningTransformerMinecraftStructureGenerator.load_from_checkpoint(
    checkpoint_path)
model.eval()

app = FastAPI()


@app.post("/complete-structure/")
async def complete_structure(input: Request):
    print(input)
    try:
        [[[print(block_str) for block_str in y]
          for y in z] for z in input.structure]
        # Convert the string values to IDs
        input_structure_ids = [[[block_token_mapper.block_to_token(schempy.Block(
            block_str)) for block_str in y] for y in z] for z in input.structure]

        # Convert the input data to a torch tensor
        input_tensor = torch.tensor(input_structure_ids)

        # Mask out the air blocks
        input_tensor[input_tensor == 1] = 0

        # Generate the structure
        def generate():
            for block, z, y, x in model.fill_structure(input_tensor, input.temperature):
                block = block_token_mapper.token_to_block(block)
                id = str(block)
                response = Block(value=id, position=(z, y, x))
                print(response.model_dump_json())
                yield response.model_dump_json() + "\n"

        # Return the response as a stream
        return StreamingResponse(generate(), media_type="application/x-ndjson")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, timeout_keep_alive=5)
