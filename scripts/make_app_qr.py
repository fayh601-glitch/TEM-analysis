#!/usr/bin/env python3
"""Generate a QR code PNG that opens the public TEM analyzer URL."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url", help="Public Streamlit app URL")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output PNG path (default: docs/tem_analyzer_qr.png)",
    )
    args = parser.parse_args()

    try:
        import qrcode
    except ImportError:
        print("Install qrcode first: pip install 'qrcode[pil]'", file=sys.stderr)
        raise SystemExit(1)

    repo = Path(__file__).resolve().parents[1]
    out = args.output or (repo / "docs" / "tem_analyzer_qr.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    img = qrcode.make(args.url)
    img.save(out)
    print(f"QR code written to {out}")
    print(f"Scans to: {args.url}")


if __name__ == "__main__":
    main()
