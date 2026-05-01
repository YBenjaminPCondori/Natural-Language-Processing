"""Coursework-facing modules for legal clause classification."""

from .data_setup import (
    ProjectPaths,
    build_project_paths,
    colab_project_candidates,
    find_project_root,
    looks_like_project_root,
    running_in_colab,
    seed_everything,
)

__all__ = [
    "ProjectPaths",
    "build_project_paths",
    "colab_project_candidates",
    "find_project_root",
    "looks_like_project_root",
    "running_in_colab",
    "seed_everything",
]
