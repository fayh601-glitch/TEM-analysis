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
from tem_rods.scale_bar import detect_scale_bar_pixels

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
    auto_scale_bar: bool = typer.Option(
        False,
        "--auto-scale-bar",
        help="Detect the scale bar automatically (uses --scale-bar-nm, default 20).",
    ),
    output_dir: Path = typer.Option(Path("outputs"), "--output-dir", "-o"),
    min_area: int = typer.Option(150, help="Minimum particle area in pixels (150 recommended for SI rods)."),
    min_eccentricity_rod: float = typer.Option(0.85, help="Min eccentricity to call a particle a rod."),
    min_aspect_ratio_rod: float = typer.Option(1.5, help="Min aspect ratio to call a particle a rod."),
    no_watershed: bool = typer.Option(True, help="Disable global watershed (usually keep off)."),
    no_split_touching: bool = typer.Option(
        False,
        help="Disable splitting of large merged blobs into separate rods.",
    ),
    split_min_area: int = typer.Option(
        500,
        help="Split blobs larger than this area (px) that likely contain 2+ rods.",
    ),
) -> None:
    """Segment particles, classify rods vs dots, and export measurements."""
    if nm_per_pixel is None:
        if auto_scale_bar:
            bar_nm = scale_bar_nm if scale_bar_nm is not None else 20.0
            bar_px, nm_per_pixel = detect_scale_bar_pixels(image, scale_bar_nm=bar_nm)
            console.print(
                f"Auto scale bar: {bar_px:.1f} px for {bar_nm:g} nm "
                f"→ {nm_per_pixel:.4f} nm/pixel"
            )
        elif scale_bar_nm is None or scale_bar_pixels is None:
            raise typer.BadParameter(
                "Provide --nm-per-pixel, --auto-scale-bar, "
                "or both --scale-bar-nm and --scale-bar-pixels."
            )
        else:
            nm_per_pixel = nm_per_pixel_from_scale_bar(scale_bar_nm, scale_bar_pixels)

    config = AnalysisConfig(
        min_particle_area_px=min_area,
        min_eccentricity_rod=min_eccentricity_rod,
        min_aspect_ratio_rod=min_aspect_ratio_rod,
        use_watershed=not no_watershed,
        split_touching_particles=not no_split_touching,
        split_min_area_px=split_min_area,
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
