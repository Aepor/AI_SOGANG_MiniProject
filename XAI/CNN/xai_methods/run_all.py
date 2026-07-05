"""Run the current CNN XAI export pipeline.

Only the four kept CNN XAI methods are exposed:

- unigram_occlusion
- ngram_occlusion
- filter_activation
- integrated_gradients

The old CSV/figure batch pipeline used to live here. The project now uses
`XAI/CNN/export_word_json.py` as the single readable entrypoint, and this file
keeps the older command path working.
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from XAI.CNN.export_word_json import main as export_word_json_main  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """Delegate to the eojeol-level JSON exporter."""
    return export_word_json_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
