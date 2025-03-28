from typing import List

from pydantic import BaseModel


class StructureRequest(BaseModel):
    model_type: str = "default"
    model_version: str | None = None
    inference_device: str | None = None
    platform: str = "java"
    version_number: int
    temperature: float = 1.0
    start_radius: int = 1
    max_iterations: int = 5
    max_blocks: int = 20
    max_alternatives: int = 1
    min_alternative_probability: float = 0.3
    ignore_replaceable_blocks: bool = False
    palette: dict[int, str]
    structure: List[List[List[int]]]


class StructureResponse(BaseModel):
    type: str
    alternative_num: int
    previous_alternative_num: int
    block_state: str
    z: int
    y: int
    x: int
