"""Exact mapping between Qwen image placeholders and merged visual tokens."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import torch


MERGE_SIZE = 2


@dataclass(frozen=True)
class ImageTokenGeometry:
    """Geometry for one still image in a batch-one Qwen input."""

    image_grid_thw: tuple[int, int, int]
    merged_grid_thw: tuple[int, int, int]
    image_token_sequence_indices: tuple[int, ...]
    patch_token_coordinates_thw: tuple[tuple[int, int, int], ...]
    token_coordinates_thw: tuple[tuple[int, int, int], ...]

    def metadata(self) -> dict[str, object]:
        return asdict(self)


def build_image_token_geometry(
    image_grid_thw: torch.Tensor | Sequence[Sequence[int]],
    image_token_sequence_indices: torch.Tensor | Sequence[int],
) -> ImageTokenGeometry:
    """Build validated row-major geometry from processor output.

    Phase 0B empirically validated the row-major order below for Qwen3-VL's
    post-2x2-merge grid.  This function deliberately handles exactly one still
    image, which is the synthetic instrumentation extraction contract.
    """
    grid = torch.as_tensor(image_grid_thw, device="cpu").tolist()
    if len(grid) != 1 or len(grid[0]) != 3:
        raise ValueError("synthetic instrumentation requires image_grid_thw for exactly one image")
    temporal, patch_rows, patch_cols = (int(value) for value in grid[0])
    if patch_rows % MERGE_SIZE or patch_cols % MERGE_SIZE:
        raise ValueError("patch grid must be divisible by Qwen's spatial merge size")
    merged = (temporal, patch_rows // MERGE_SIZE, patch_cols // MERGE_SIZE)
    indices = tuple(int(value) for value in torch.as_tensor(image_token_sequence_indices, device="cpu").tolist())
    expected = merged[0] * merged[1] * merged[2]
    if len(indices) != expected:
        raise ValueError(f"image placeholder count {len(indices)} does not match merged grid token count {expected}")
    coordinates = tuple(
        (time, row, col)
        for time in range(merged[0])
        for row in range(merged[1])
        for col in range(merged[2])
    )
    patch_coordinates = tuple(
        (time, row, col)
        for time in range(temporal)
        for row in range(patch_rows)
        for col in range(patch_cols)
    )
    return ImageTokenGeometry((temporal, patch_rows, patch_cols), merged, indices, patch_coordinates, coordinates)
