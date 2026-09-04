"""Deterministic synthetic images for instrumentation validation only."""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Callable

import numpy as np
from PIL import Image


@dataclass(frozen=True)
class SyntheticImageSpec:
    """A deterministic image and the information needed to reproduce it."""

    identifier: str
    description: str
    height: int
    width: int
    seed: int
    pattern: str

    def metadata(self) -> dict[str, object]:
        return asdict(self)


SYNTHETIC_IMAGES: tuple[SyntheticImageSpec, ...] = (
    SyntheticImageSpec(
        "asymmetric_texture_256x384",
        "Asymmetric 8-by-12 grid of independently seeded 32-pixel RGB cells.",
        256, 384, 7_310_941, "cell_texture",
    ),
    SyntheticImageSpec(
        "asymmetric_texture_384x256",
        "Asymmetric 12-by-8 grid of independently seeded 32-pixel RGB cells.",
        384, 256, 7_310_942, "cell_texture",
    ),
    SyntheticImageSpec(
        "spatial_regions_256x256",
        "Four distinct coloured quadrants with a diagonal stripe and centre marker.",
        256, 256, 7_310_943, "regions",
    ),
    SyntheticImageSpec(
        "uniform_control_256x256",
        "Uniform mid-grey control image with no designed spatial variation.",
        256, 256, 7_310_944, "uniform",
    ),
)


def _cell_texture(spec: SyntheticImageSpec) -> np.ndarray:
    rng = np.random.default_rng(spec.seed)
    return rng.integers(0, 256, (spec.height, spec.width, 3), dtype=np.uint8)


def _regions(spec: SyntheticImageSpec) -> np.ndarray:
    pixels = np.zeros((spec.height, spec.width, 3), dtype=np.uint8)
    half_h, half_w = spec.height // 2, spec.width // 2
    pixels[:half_h, :half_w] = (220, 45, 45)
    pixels[:half_h, half_w:] = (45, 180, 65)
    pixels[half_h:, :half_w] = (45, 85, 220)
    pixels[half_h:, half_w:] = (225, 185, 45)
    row, col = np.indices((spec.height, spec.width))
    pixels[np.abs(row - col) < 5] = (255, 255, 255)
    centre = (np.abs(row - half_h) < 12) & (np.abs(col - half_w) < 12)
    pixels[centre] = (0, 0, 0)
    return pixels


def _uniform(spec: SyntheticImageSpec) -> np.ndarray:
    return np.full((spec.height, spec.width, 3), 127, dtype=np.uint8)


_GENERATORS: dict[str, Callable[[SyntheticImageSpec], np.ndarray]] = {
    "cell_texture": _cell_texture,
    "regions": _regions,
    "uniform": _uniform,
}


def render_synthetic_image(spec: SyntheticImageSpec) -> Image.Image:
    """Render an RGB image without touching disk or global RNG state."""
    try:
        pixels = _GENERATORS[spec.pattern](spec)
    except KeyError as exc:
        raise ValueError(f"unknown synthetic pattern: {spec.pattern}") from exc
    return Image.fromarray(pixels, mode="RGB")
