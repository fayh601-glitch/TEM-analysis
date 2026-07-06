#!/usr/bin/env python3
"""
Build docs/TEM-analysis-Manual.pdf from docs/MANUAL.md.

Requires: pip install fpdf2  (listed in requirements-optional.txt)

Usage:
    python scripts/build_manual_pdf.py
"""

from __future__ import annotations

import re
from pathlib import Path

from fpdf import FPDF

ROOT = Path(__file__).resolve().parents[1]
MD_PATH = ROOT / "docs" / "MANUAL.md"
PDF_PATH = ROOT / "docs" / "TEM-analysis-Manual.pdf"

SKIP_CHARS = set("┌┐└┘│▼─→")


class ManualPDF(FPDF):
    def header(self) -> None:
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, "TEM-Analysis Manual", align="R", new_x="LMARGIN", new_y="NEXT")
        self.ln(2)

    def footer(self) -> None:
        self.set_y(-12)
        self.set_font("Helvetica", "I", 8)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, f"Page {self.page_no()}", align="C")


def _strip_md(text: str) -> str:
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    text = text.replace("**", "").replace("`", "")
    text = text.replace("→", "->").replace("·", "-").replace("±", "+/-")
    text = text.replace("—", "-").replace("–", "-").replace("•", "-")
    return text.encode("latin-1", errors="replace").decode("latin-1")


def _write(pdf: ManualPDF, text: str, *, h: float = 5, style: str = "", size: int = 11) -> None:
    clean = _strip_md(text).strip()
    if not clean:
        return
    pdf.set_x(pdf.l_margin)
    pdf.set_font("Helvetica", style=style, size=size)
    pdf.set_text_color(0, 0, 0)
    pdf.multi_cell(pdf.epw, h, clean)


def build_pdf() -> None:
    if not MD_PATH.exists():
        raise FileNotFoundError(MD_PATH)

    pdf = ManualPDF()
    pdf.set_margins(18, 18, 18)
    pdf.set_auto_page_break(auto=True, margin=18)
    pdf.add_page()

    in_code = False
    for raw_line in MD_PATH.read_text(encoding="utf-8").splitlines():
        line = raw_line.rstrip()

        if not in_code and any(ch in line for ch in SKIP_CHARS):
            if not line.startswith("#"):
                continue

        if line.startswith("```"):
            in_code = not in_code
            continue

        if in_code:
            pdf.set_x(pdf.l_margin)
            pdf.set_font("Courier", size=9)
            pdf.set_text_color(40, 40, 40)
            pdf.multi_cell(pdf.epw, 5, _strip_md(line) if line else " ")
            pdf.set_text_color(0, 0, 0)
            continue

        if not line.strip():
            pdf.ln(3)
            continue

        if line.startswith("# "):
            pdf.ln(4)
            _write(pdf, line[2:], h=8, style="B", size=16)
            continue

        if line.startswith("## "):
            pdf.ln(3)
            _write(pdf, line[3:], h=7, style="B", size=13)
            continue

        if line.startswith("### "):
            pdf.ln(2)
            _write(pdf, line[4:], h=6, style="B", size=11)
            continue

        if line.startswith("|") and "---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]
            row = " | ".join(_strip_md(c) for c in cells if c)
            _write(pdf, row, size=9)
            continue

        if line.startswith("- ") or line.startswith("* "):
            _write(pdf, f"- {_strip_md(line[2:])}")
            continue

        if line.startswith("---"):
            pdf.ln(2)
            continue

        _write(pdf, line)

    PDF_PATH.parent.mkdir(parents=True, exist_ok=True)
    pdf.output(str(PDF_PATH))
    print(f"Wrote {PDF_PATH}")


if __name__ == "__main__":
    build_pdf()
