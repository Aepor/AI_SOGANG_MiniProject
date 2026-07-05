"""Path helpers for the AI_SOGANG XAI workspace.

여러 스크립트를 repo root, `XAI/CNN`, 또는 notebook 폴더에서 실행할 수 있기 때문에
상대 경로를 매번 직접 쓰면 쉽게 깨진다. 이 파일은 "현재 어디서 실행하든 같은 파일을
찾기" 위한 작은 경로 규칙들을 모아 둔다.
"""

from __future__ import annotations

from pathlib import Path


def repo_root() -> Path:
    """Return the repository root based on this file location.

    `paths.py`는 `XAI/CNN/shared/paths.py`에 있으므로 `parents[3]`이 repo root이다.
    cwd에 의존하지 않아서 VS Code/Jupyter 실행 위치가 달라도 안정적이다.
    """
    return Path(__file__).resolve().parents[3]


def xai_root() -> Path:
    """Return the `XAI` directory inside the repo."""
    return repo_root() / "XAI"


def cnn_dir() -> Path:
    """Return the CNN-specific XAI directory."""
    return xai_root() / "CNN"


def ensure_dir(path: Path) -> Path:
    """Create a directory if needed and return it for fluent path setup."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def default_cnn_model_path() -> Path:
    """Default trained CNN checkpoint path."""
    return cnn_dir() / "best_cnn_model.pt"


def default_cnn_cache_path() -> Path:
    """Default cache for the reconstructed CNN vocabulary."""
    return cnn_dir() / "cnn_preprocess_cache.pkl"


def default_cnn_output_dir() -> Path:
    """Default directory for generated CNN XAI outputs."""
    return cnn_dir() / "outputs"


def find_data_file(filename: str) -> Path:
    """Find an NSMC data file from the known project locations.

    The repo has had data copies under `Data/NSMC`, `XAI/Data/NSMC`, and
    `model_constructed/Data/NSMC`. Searching in a fixed priority order keeps
    scripts usable without requiring the user to pass paths every time.
    """
    root = repo_root()
    candidates = [
        xai_root() / "Data" / "NSMC" / filename,
        root / "Data" / "NSMC" / filename,
        root / "model_constructed" / "Data" / "NSMC" / filename,
        cnn_dir() / filename,
        root / filename,
    ]
    for path in candidates:
        if path.exists():
            return path
    tried = "\n".join(str(path) for path in candidates)
    raise FileNotFoundError(f"Could not find {filename}. Tried:\n{tried}")
