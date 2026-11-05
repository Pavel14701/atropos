"""Atropos — a hypothesis-killing library for quantitative research.

Atropos is the Fate who cuts the thread of life: every candidate
hypothesis enters the pipeline alive and leaves it with exactly one verdict —
SURVIVED, KILLED, or INCONCLUSIVE.  The package bundles the tools that do the
cutting: preregistration, null controls, incremental regression, robustness
batteries, walk-forward evaluation, Monte Carlo risk envelopes, and ensemble
pipelines that chain every stage in adversarial order.

The two front-door functions are re-exported at the package level::

    from atropos import generate_prereg, run_full_pipeline

The ML kill chain and the unified wrapper are module-level entry
points (no package re-exports):: from atropos.ml_training import
run_ml_kill_protocol; from atropos.kill_chain import
run_full_kill_chain.  Importing ``atropos.ml_ensemble`` has a
documented side effect: it registers the composite ``"ensemble"``
model in ``MODEL_REGISTRY`` and seeds its
``DEFAULT_TRAINING_CONFIG["model_params"]`` entry.

Every module is also importable directly, e.g.::

Run ``python -m atropos.<module>`` for the synthetic demo bundled with each
module (for example ``python -m atropos.ensemble_pipeline``).
"""

__version__ = "1.0.0"

from atropos.ensemble_pipeline import run_full_pipeline
from atropos.prereg_generator import generate_prereg

__all__ = ["__version__", "generate_prereg", "run_full_pipeline"]
