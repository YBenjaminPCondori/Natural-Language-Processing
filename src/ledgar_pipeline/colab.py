"""Google Colab setup helpers for the LEDGAR pipeline."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Iterable


DEFAULT_COLAB_PROJECT_CANDIDATES = (
    Path("/content/Natural-Language-Processing"),
    Path("/content/drive/MyDrive/Natural-Language-Processing"),
)


def running_in_colab() -> bool:
    """Return True when running inside a Google Colab runtime."""
    return bool(os.getenv("COLAB_RELEASE_TAG")) or importlib.util.find_spec("google.colab") is not None


def add_import_paths(project_root: Path | str) -> tuple[Path, Path]:
    """Add the project root and src directory to sys.path."""
    root = Path(project_root).expanduser().resolve()
    src = root / "src"
    for path in (root, src):
        path_str = str(path)
        if path_str not in sys.path:
            sys.path.insert(0, path_str)
    return root, src


def is_project_root(path: Path | str) -> bool:
    """Check for the expected coursework repository markers."""
    candidate = Path(path).expanduser()
    return (
        (candidate / "pyproject.toml").exists()
        and (candidate / "src").exists()
        and (candidate / "modules").exists()
    )


def find_project_root(
    start: Path | str | None = None,
    *,
    override: Path | str | None = None,
    extra_candidates: Iterable[Path | str] = (),
) -> Path:
    """Find the LEDGAR project root in local Jupyter or Colab."""
    candidates: list[Path] = []
    env_override = os.getenv("LEDGAR_PROJECT_ROOT")
    for value in (override, env_override):
        if value:
            candidates.append(Path(value))

    start_path = Path(start or os.getcwd()).expanduser()
    candidates.extend([start_path, *start_path.parents])
    candidates.extend(Path(path) for path in extra_candidates)

    if running_in_colab():
        candidates.extend(DEFAULT_COLAB_PROJECT_CANDIDATES)

    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if is_project_root(resolved):
            return resolved

    raise FileNotFoundError(
        "Could not find the LEDGAR project root. In Colab, clone/upload the repo and "
        "set LEDGAR_PROJECT_ROOT or PROJECT_ROOT_OVERRIDE to the folder containing "
        "pyproject.toml, src/, and modules/."
    )


def mount_google_drive(mount_point: Path | str = "/content/drive") -> Path:
    """Mount Google Drive when running in Colab."""
    if not running_in_colab():
        return Path(mount_point)

    from google.colab import drive

    drive.mount(str(mount_point))
    return Path(mount_point)


def configure_notebook_paths(
    project_root: Path | str | None = None,
    *,
    mount_drive: bool = False,
    drive_mount_point: Path | str = "/content/drive",
) -> tuple[Path, Path]:
    """Optionally mount Drive, find the repo root, and configure imports."""
    if mount_drive:
        mount_google_drive(drive_mount_point)
    root = find_project_root(override=project_root)
    os.environ["LEDGAR_PROJECT_ROOT"] = str(root)
    return add_import_paths(root)
