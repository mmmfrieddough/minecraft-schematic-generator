from typing import Any

import torch
from torch.utils.data import DataLoader


class ResumableDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._sampler_rng_state = None

    def state_dict(self) -> dict[str, Any]:
        """Save dataloader state."""
        return {
            "sampler_rng_state": torch.get_rng_state(),
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load dataloader state."""
        self._sampler_rng_state = state_dict.get("sampler_rng_state")
        if self._sampler_rng_state is not None:
            torch.set_rng_state(self._sampler_rng_state)
