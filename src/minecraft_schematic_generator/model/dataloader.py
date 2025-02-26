from typing import Any

from torch.utils.data import DataLoader


class ResumableDataLoader(DataLoader):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def state_dict(self) -> dict[str, Any]:
        """Save dataloader state."""
        return {
            "epoch": self.sampler.epoch,
            "seed": self.sampler.seed,
        }

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        """Load dataloader state."""
        self.sampler.epoch = state_dict["epoch"]
        self.sampler.seed = state_dict["seed"]
