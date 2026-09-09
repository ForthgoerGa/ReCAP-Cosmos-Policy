"""Dense causal video candidates, independent of the original state pool."""
from ..retrieval import PushTRetrieval

class VideoRetrievalPool(PushTRetrieval):
    STRIDE = 1

    def window_metadata(self):
        lengths = []
        for key, item in self._base_data.items():
            ids = self._demo_indices[key]
            ends = [self._subframes[i]["t_last"] for i in ids]
            if ends != list(range(len(item["states"]))):
                raise ValueError(f"Non-dense video endpoints for {key}")
        for sf in self._subframes:
            length = min(self.WINDOW_SIZE, sf["t_last"] + 1)
            if sf["start"] != sf["t_last"] + 1 - length:
                raise ValueError("Non-causal video window")
            lengths.append(length)
        return {"window_length": lengths,
                "window_type": ["partial_window" if n < self.WINDOW_SIZE else "full_window" for n in lengths]}
