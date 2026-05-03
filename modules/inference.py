"""Inference module for saved LEDGAR models."""

try:
    from ledgar_pipeline.inference import *  # noqa: F401,F403
except ModuleNotFoundError:
    __all__: list[str] = []
