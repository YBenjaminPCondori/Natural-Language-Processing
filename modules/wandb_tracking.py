"""Legacy optional Weights & Biases compatibility wrapper.

The coursework notebook does not import or require this module.
"""

try:
    from ledgar_pipeline.wandb_tracking import *  # noqa: F401,F403
except ModuleNotFoundError:
    __all__: list[str] = []
