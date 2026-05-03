"""Build audit, verification, and canonical coursework artifacts."""

from __future__ import annotations

import json
import sys
from pathlib import Path


def find_project_root() -> Path:
    current = Path(__file__).resolve()
    for candidate in [current.parent, *current.parents]:
        if (candidate / "pyproject.toml").exists() and (candidate / "modules").exists():
            return candidate
    raise RuntimeError("Could not locate project root containing pyproject.toml and modules/.")


def main() -> None:
    project_root = find_project_root()
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))
    from modules.coursework_artifacts import build_all_coursework_artifacts

    summary = build_all_coursework_artifacts(project_root)
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
