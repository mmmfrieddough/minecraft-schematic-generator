from typing import List

from pydantic import BaseModel


class StructureRequest(BaseModel):
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
