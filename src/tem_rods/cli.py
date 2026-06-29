"""
Command-Line Interface — the `tem-rods` terminal command
=========================================================

This file is what you run from the terminal (`tem-rods analyze ...`). It reads
your options (image path, scale bar, output folder), builds the analysis settings,
and hands everything off to the main pipeline.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

from tem_rods.calibrate import nm_per_pixel_from_scale_bar
from tem_rods.models import AnalysisConfig
from tem_rods.pipeline import analyze_image, print_summary

app = typer.Typer(
    name="tem-rods",
    help="Analyze CdSe/CdS TEM images: measure rod/dot size and classify shape.",
    no_args_is_help=True,
)
console = Console()


@app.callback()
def main() -> None:
    """CdSe/CdS TEM rod and dot analysis."""


@app.command("analyze")
def analyze(
    image: Path = typer.Option(..., "--image", "-i", help="Path to TEM image (png/tif/jpg)."),
    nm_per_pixel: Optional[float] = typer.Option(
        None,
        "--nm-per-pixel",
        help="Nanometers per pixel (required unless scale bar args provided).",
    ),
    scale_bar_nm: Optional[float] = typer.Option(
        None,
        "--scale-bar-nm",
        help="Scale bar length in nm (use with --scale-bar-pixels).",
    ),
    scale_bar_pixels: Optional[float] = typer.Option(
        None,
        "--scale-bar-pixels",
        help="Scale bar length in pixels.",
    ),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o"),
    min_area: int = typer.Option(150, help="Minimum particle area in pixels (150 recommended for SI rods)."),
    min_eccentricity_rod: float = typer.Option(0.85, help="Min eccentricity to call a particle a rod."),
    min_aspect_ratio_rod: float = typer.Option(1.5, help="Min aspect ratio to call a particle a rod."),
    no_watershed: bool = typer.Option(True, help="Disable watershed (recommended for nanorods)."),
) -> None:
    """Segment particles, classify rods vs dots, and export measurements."""
    if nm_per_pixel is None:
        if scale_bar_nm is None or scale_bar_pixels is None:
            raise typer.BadParameter(
                "Provide --nm-per-pixel OR both --scale-bar-nm and --scale-bar-pixels."
            )
        nm_per_pixel = nm_per_pixel_from_scale_bar(scale_bar_nm, scale_bar_pixels)

    config = AnalysisConfig(
        min_particle_area_px=min_area,
        min_eccentricity_rod=min_eccentricity_rod,
        min_aspect_ratio_rod=min_aspect_ratio_rod,
        use_watershed=not no_watershed,
    )

    result = analyze_image(
        image,
        nm_per_pixel,
        output_dir=output_dir,
        config=config,
    )
    print_summary(result)


if __name__ == "__main__":
    app()
