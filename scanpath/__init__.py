"""scanpath: path generator -> instrument model -> evaluator -> optimiser.

One file per stage: paths.py (1), instruments.py (2), metrics.py (3), optimisation.py (4);
core.py holds the shared records, registries, image formation and pipeline calls.

Public API (everything a user needs):

    Budget, Trajectory, Sample          records passed between stages
    parameters(name), describe()        the explicit parameters of any component
    make_path                           stage 1: commanded path
    simulate                            stage 2, case A: path -> path
    make_image                          stage 2, case B: image -> image
    image_as_acquired                   the image a user sees after a case-A model
    evaluate                            stage 3: named scores
    run                                 stages 1-3 in one call (either case of stage 2)
    grid_search, random_search,         stage 4: interchangeable searches, same inputs and
    pattern_search, de_search,                   outputs (SEARCHES maps names to them)
    gp_search
    limits, pareto_front                constraints and competing objectives
    register_path, register_instrument, register_image_model, register_metric
                                        add a plug-in without editing the package
"""
from .core import (Budget, Result, Sample, Trajectory, describe, evaluate, image_as_acquired,
                   make_image, make_path, parameters, reconstruct, register_image_model,
                   register_instrument, register_metric, register_path, run, simulate,
                   IMAGE_MODELS, INSTRUMENTS, METRICS, PATHS)
from . import paths, instruments, metrics  # noqa: F401  (registers the built-ins)
from .optimisation import (SEARCHES, de_search, gp_search, grid_search, limits, pareto_front,
                           pattern_search, random_search)

__all__ = ["Budget", "Result", "Sample", "Trajectory", "describe", "evaluate",
           "image_as_acquired", "make_image", "make_path", "parameters", "reconstruct",
           "register_image_model", "register_instrument", "register_metric",
           "register_path", "run", "simulate", "grid_search", "random_search",
           "pattern_search", "de_search", "gp_search", "limits", "pareto_front", "SEARCHES",
           "IMAGE_MODELS", "INSTRUMENTS", "METRICS", "PATHS"]
