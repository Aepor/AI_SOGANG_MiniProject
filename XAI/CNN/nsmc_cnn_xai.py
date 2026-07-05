"""Compatibility wrapper for the current CNN XAI pipeline.

The current implementation exports eojeol-level JSON for the four kept methods:
unigram occlusion, n-gram occlusion, filter activation, and Integrated
Gradients. This file keeps the old command working:

    python XAI/CNN/nsmc_cnn_xai.py
"""

from __future__ import annotations

import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from XAI.CNN.xai_methods.run_all import main


if __name__ == "__main__":
    raise SystemExit(main())
