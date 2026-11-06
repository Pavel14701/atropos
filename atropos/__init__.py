"""Atropos — a hypothesis-killing framework for quantitative research.

Atropos is the Fate who cuts the thread of life: every candidate
hypothesis enters the pipeline alive and leaves it with exactly one verdict —
SURVIVED, KILLED, or INCONCLUSIVE.  The package bundles the tools that do the
cutting: preregistration, null controls, incremental regression, robustness
batteries, walk-forward evaluation, Monte Carlo risk envelopes, and ensemble
pipelines that chain every stage in adversarial order.

The two front-door functions are re-exported at the package level::

    from atropos import generate_prereg, run_full_pipeline

The unified kill chain (``run_full_kill_chain``) joins them at the
package level, together with the curated backtest golden path:
simulate → rank → normalize → admission → metrics.  The lambdarank
oracle lives in ``atropos.ranking`` and needs the optional
``[ranking]`` extra.  The unified ML protocol stays module-level
(``from atropos.ml_training import run_ml_kill_protocol``).
Importing ``atropos.ml_ensemble`` has a documented side effect: it
registers the composite ``"ensemble"`` model in ``MODEL_REGISTRY``
and seeds its ``DEFAULT_TRAINING_CONFIG["model_params"]`` entry.

Every module is also importable directly, e.g.::

Run ``python -m atropos.<module>`` for the synthetic demo bundled with each
module (for example ``python -m atropos.ensemble_pipeline``).
"""

__version__ = "1.0.0"

from atropos.backtest_metrics import (
    per_trade_sharpe,
    pooled_stats,
    trade_curve_stats,
)
from atropos.backtesting import (
    CostModel,
    run_admission_policy,
    run_position_state_machine,
    simulate_signal,
)
from atropos.ensemble_pipeline import run_full_pipeline
from atropos.kill_chain import run_full_kill_chain
from atropos.prereg_generator import generate_prereg
from atropos.ranking import (
    RANKER_SUITABILITY,
    RANKING_METRICS,
    CandidateGroup,
    rank_normalize_group,
    rank_within_groups,
)

# Curated surface only: front doors + kill chain, then the backtest
# golden path simulate -> rank -> normalize -> admission -> metrics.
# The oracle/comparison tools (rank_with_lambdarank, ev_threshold,
# paired_bootstrap_diff, the maker/market sims) stay in their
# submodules and are deliberately NOT re-exported here.
__all__ = [
    "RANKER_SUITABILITY",
    "RANKING_METRICS",
    "CandidateGroup",
    "CostModel",
    "__version__",
    "generate_prereg",
    "per_trade_sharpe",
    "pooled_stats",
    "rank_normalize_group",
    "rank_within_groups",
    "run_admission_policy",
    "run_full_kill_chain",
    "run_full_pipeline",
    "run_position_state_machine",
    "simulate_signal",
    "trade_curve_stats",
]
