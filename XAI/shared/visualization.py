"""Small plotting helpers for notebook and script outputs.

Script 실행 환경은 화면이 없는 터미널일 수 있다. 그래서 matplotlib backend를 `Agg`로
고정해 PNG 파일 저장만 하도록 만들었다. Notebook에서는 별도 표시용 plot을 직접 그려도
되지만, CLI 산출물은 이 helper를 사용한다.
"""

from __future__ import annotations

import re
from pathlib import Path

try:
    import matplotlib

    # GUI 창을 띄우지 않는 backend. 서버/터미널 실행에서도 PNG 저장이 가능하다.
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except Exception:
    HAS_MATPLOTLIB = False
    plt = None


def safe_filename(value: str) -> str:
    """Convert sample ids/text into a filesystem-safe short filename."""
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return value[:120] or "sample"


def configure_matplotlib() -> None:
    """Configure Korean-friendly plotting defaults when matplotlib is available."""
    if not HAS_MATPLOTLIB or plt is None:
        return

    # Windows 기본 한글 폰트인 Malgun Gothic을 우선 사용하고, 없으면 DejaVu Sans로 fallback.
    plt.rcParams["axes.unicode_minus"] = False
    plt.rcParams["font.family"] = ["Malgun Gothic", "DejaVu Sans"]


def plot_bar(
    labels: list[str],
    values: list[float],
    title: str,
    xlabel: str,
    path: Path,
    color: str = "#4C78A8",
) -> None:
    """Save a horizontal bar chart for top-k XAI evidence rows."""
    if not HAS_MATPLOTLIB or plt is None or not labels:
        return
    configure_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)

    # label 개수에 따라 높이를 조절해서 긴 token/ngram 목록이 겹치지 않게 한다.
    height = max(3.0, min(8.0, 0.45 * len(labels) + 1.5))
    plt.figure(figsize=(10, height))
    y = list(range(len(labels)))
    plt.barh(y, values, color=color)
    plt.yticks(y, labels)
    plt.gca().invert_yaxis()
    plt.xlabel(xlabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
