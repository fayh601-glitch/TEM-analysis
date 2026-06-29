#!/usr/bin/env python3
"""
Extract embedded images from the Cossairt 2018 SI PDF (or any PDF).

Usage:
    python scripts/extract_pdf_images.py /path/to/c8qm00056e1.pdf --output data/raw

Requires: pip install pymupdf
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
    parser = argparse.ArgumentParser(description="Extract images from a PDF.")
    parser.add_argument("pdf", type=Path, help="Path to PDF file")
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for extracted images",
    )
    args = parser.parse_args()
    n = extract_images(args.pdf, args.output)
    print(f"\nExtracted {n} image(s) to {args.output}")
    print("Next: crop individual TEM panels and fill data/calibration.csv with scale bars.")


if __name__ == "__main__":
    main()
