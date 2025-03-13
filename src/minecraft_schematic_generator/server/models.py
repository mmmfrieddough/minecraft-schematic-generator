from typing import List

from pydantic import BaseModel


class StructureRequest(BaseModel):
    platform: str
    version_number: int
    temperature: float
    start_radius: int
    max_iterations: int
    max_blocks: int
    max_alternatives: int
    palette: dict[int, str]
    structure: List[List[List[int]]]


class Block(BaseModel):
    alternative_num: int
    previous_alternative_num: int
    block_state: str
    z: int
    y: int
    x: int
