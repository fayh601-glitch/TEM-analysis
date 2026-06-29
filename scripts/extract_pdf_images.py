#!/usr/bin/env python3
"""
Extract PDF Images — pull embedded figures out of a scientific PDF
===================================================================

Many papers publish supplementary TEM images inside a PDF file. This script
opens the PDF and saves each embedded image as its own PNG file so you can
analyze them with the main pipeline.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def extract_images(pdf_path: Path, output_dir: Path) -> int:
    try:
        import fitz  # pymupdf
    except ImportError as exc:
        raise SystemExit(
            "pymupdf is required. Install with: pip install pymupdf"
        ) from exc

    output_dir.mkdir(parents=True, exist_ok=True)
    doc = fitz.open(pdf_path)
    count = 0

    for page_index in range(len(doc)):
        page = doc[page_index]
        for img_index, img in enumerate(page.get_images(full=True)):
            xref = img[0]
            base = doc.extract_image(xref)
            ext = base["ext"]
            data = base["image"]
            out_name = f"{pdf_path.stem}_p{page_index + 1:02d}_img{img_index + 1:02d}.{ext}"
            out_path = output_dir / out_name
            out_path.write_bytes(data)
            count += 1
            print(f"Wrote {out_path}")

    doc.close()
    return count


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Extract embedded images from a PDF into data/raw/."
    )
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Output directory (default: data/raw)",
    )
    args = parser.parse_args()
    n = extract_images(args.pdf, args.output)
    print(f"Extracted {n} images to {args.output}")


if __name__ == "__main__":
    main()
