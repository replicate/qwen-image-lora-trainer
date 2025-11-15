#!/usr/bin/env python3
"""Sync root requirements.txt from ai-toolkit/requirements.txt plus extra pins."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TOOLKIT_REQ = ROOT / "ai-toolkit" / "requirements.txt"
ROOT_REQ = ROOT / "requirements.txt"
EXTRA_LINES = ["pydantic<2.12.0"]
EXCLUDE = {"gradio", "pydantic"}

def normalise(lines):
    seen = set()
    ordered = []
    for raw in lines:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line in seen or line in EXCLUDE:
            continue
        seen.add(line)
        ordered.append(line)
    return ordered


def main() -> None:
    toolkit_lines = TOOLKIT_REQ.read_text().splitlines()
    merged = normalise(toolkit_lines)
    for line in EXTRA_LINES:
        if line not in merged:
            merged.append(line)
    ROOT_REQ.write_text("\n".join(merged) + "\n")
    print(f"Wrote {ROOT_REQ} with {len(merged)} entries")


if __name__ == "__main__":
    main()
