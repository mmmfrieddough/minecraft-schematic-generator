from .dataloader import ResumableDataLoader


class CombinedDataLoader:
    def __init__(self, dataloaders: list[ResumableDataLoader]):
        self._dataloaders = dataloaders
        self._lengths = [len(dl) for dl in self._dataloaders]
        self._active_indices = None
        self._consumed = None

    def __iter__(self):
        if self._active_indices is None or self._consumed is None:
            self._active_indices = set(
                [i for i, length in enumerate(self._lengths) if length > 0]
            )
            self._consumed = [0] * len(self._dataloaders)

        # Create one iterator per dataloader
        iterators = [iter(dl) for dl in self._dataloaders]

        while self._active_indices:
            # Pick loader i in 'active' with smallest ratio consumed[i] / length[i]
            i = min(
                self._active_indices,
                key=lambda idx: self._consumed[idx] / self._lengths[idx],
            )
            try:
                yield next(iterators[i])
                self._consumed[i] += 1
            except StopIteration:
                # This dataloader is exhausted
                self._active_indices.remove(i)

    def __len__(self) -> int:
        return sum(self._lengths)

    def state_dict(self) -> dict:
        # Save states of individual dataloaders
        dataloader_states = []
        for dataloader in self._dataloaders:
            dataloader_states.append(dataloader.state_dict())

        return {
            "active_indices": list(self._active_indices)
            if self._active_indices
            else None,
            "consumed": self._consumed,
            "lengths": self._lengths,
            "dataloader_states": dataloader_states,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        if state_dict["lengths"] != self._lengths:
            raise ValueError(
                f"Loaded state lengths {state_dict['lengths']} do not match current lengths {self._lengths}"
            )

        self._active_indices = (
            set(state_dict["active_indices"]) if state_dict["active_indices"] else None
        )
        self._consumed = state_dict["consumed"]

        # Load states of individual dataloaders
        for dataloader, dataloader_state in zip(
            self._dataloaders, state_dict["dataloader_states"]
        ):
            dataloader.load_state_dict(dataloader_state)
