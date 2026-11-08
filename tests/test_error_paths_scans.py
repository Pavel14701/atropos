"""Error-path ladder pins: feature_scan + strategy + book (W15-2).

Coverage-driven: every case targets an UNCOVERED raise line from the
V-0 coverage report.  Mutation fact: deleting the pinned raise fails
the case.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from atropos.book import Book, _demo_md
from atropos.feature_scan import (
    auto_discovery,
    bucket_scan,
    correlation_cluster,
    feature_analysis,
    feature_importance,
    missingness_scan,
    rule_fit,
)
from atropos.legs import (
    Compare,
    EntrySpec,
    ExitSpec,
    InstrumentRef,
    LegSpec,
    StopSpec,
)
from atropos.strategy import (
    PortfolioPolicy,
    StrategySpec,
    _recount_and_verify,
    run_strategy,
)


def test_missingness_scan_guards() -> None:
    """Threshold gate, name collisions, multi-column rejection."""
    rng = pd.Series([1.0, 2.0], index=pd.RangeIndex(2))
    with pytest.raises(ValueError, match=r"must be a float in \(0, 1\)"):
        missingness_scan(rng, {"f": rng}, {}, nan_threshold=1.5)
    with pytest.raises(ValueError, match="collide"):
        missingness_scan(rng, {"f": rng}, {"f": rng})
    with pytest.raises(ValueError, match="single column"):
        missingness_scan(
            rng, {"f": pd.DataFrame({"a": [1.0], "b": [2.0]})}, {}
        )


def test_correlation_cluster_guards() -> None:
    """Empty features and no-aligned-rows refuse loudly."""
    misaligned = pd.Series([1.0], index=pd.RangeIndex(100, 101))
    with pytest.raises(ValueError, match="must not be empty"):
        correlation_cluster({}, misaligned)
    # disjoint indexes -> inner join is empty -> loud refusal
    off_grid = pd.Series(
        [1.0], index=pd.date_range("2030-01-01", periods=1, tz="UTC")
    )
    with pytest.raises(ValueError, match="no rows survive alignment"):
        correlation_cluster({"f": off_grid}, misaligned)


def test_rule_fit_guards() -> None:
    """Empty rules, empty/overlapping/out-of-range splits, NaN outcome."""
    out = pd.Series([1.0] * 4)
    mask = [True, False, True, False]
    with pytest.raises(ValueError, match="must not be empty"):
        rule_fit(out, {}, train_idx=[0, 1], test_idx=[2, 3])
    nan_out = out.copy()
    nan_out.iloc[1] = np.nan
    with pytest.raises(ValueError, match="outcome contains NaN"):
        rule_fit(nan_out, {"r": mask}, train_idx=[0, 1], test_idx=[2, 3])
    with pytest.raises(ValueError, match="non-empty"):
        rule_fit(out, {"r": mask}, train_idx=[], test_idx=[2, 3])
    with pytest.raises(ValueError, match="overlap"):
        rule_fit(out, {"r": mask}, train_idx=[0, 1, 2], test_idx=[1, 2, 3])
    with pytest.raises(ValueError, match="out of range"):
        rule_fit(out, {"r": mask}, train_idx=[0, 1, 999], test_idx=[2, 3])


def test_bucket_scan_guards() -> None:
    """n_buckets / n_perm / seed rungs."""
    out = pd.Series([1.0] * 8)
    feat = pd.Series([1.0] * 8)
    with pytest.raises(ValueError, match="n_buckets"):
        bucket_scan(out, {"f": feat}, n_buckets=1)
    with pytest.raises(ValueError, match="n_perm"):
        bucket_scan(out, {"f": feat}, n_perm=0)
    with pytest.raises(ValueError, match="seed"):
        bucket_scan(out, {"f": feat}, seed=-1)


def test_feature_analysis_guards() -> None:
    """feature_analysis: single-column rejection and split consistency."""
    out = pd.Series([1.0] * 8)
    frame_feat = pd.DataFrame({"a": [1.0] * 8, "b": [2.0] * 8})
    with pytest.raises(ValueError, match="single column"):
        feature_analysis(out, {"f": frame_feat}, {})
    with pytest.raises(ValueError, match="BOTH train_idx and test_idx"):
        feature_analysis(out, {"f": out}, {}, candidate_rules={"r": [True]})
    with pytest.raises(ValueError, match="positive finite float"):
        feature_analysis(out, {"f": out}, {}, orthogonal_threshold=-1.0)


def test_auto_discovery_guards() -> None:
    """auto_discovery: empty pool, cap, spearman floor, collisions."""
    out = pd.Series([1.0] * 8)
    cand = pd.Series([1.0] * 8)
    with pytest.raises(ValueError, match="must not be empty"):
        auto_discovery(out, {}, {})
    big_pool = {f"c{ix}": cand for ix in range(25)}
    with pytest.raises(ValueError, match="max_features"):
        auto_discovery(out, big_pool, {}, max_features=20)
    with pytest.raises(ValueError, match="min_abs_spearman"):
        auto_discovery(out, {"c": cand}, {}, min_abs_spearman=float("nan"))
    with pytest.raises(ValueError, match="collide"):
        auto_discovery(out, {"c": cand}, {"c": cand})


def test_feature_importance_guards() -> None:
    """feature_importance: bad model, empty features, too-few rows."""
    out = pd.Series([1.0] * 8)
    feat = pd.Series([1.0] * 8)
    with pytest.raises(ValueError, match="model must be"):
        feature_importance(out, {"f": feat}, model="shap")
    with pytest.raises(ValueError, match="must not be empty"):
        feature_importance(out, {})
    with pytest.raises(ValueError, match="too few rows"):
        feature_importance(
            pd.Series([1.0, 2.0]),
            {"a": pd.Series([1.0, 2.0]), "b": pd.Series([2.0, 1.0])},
            model="linear",
            n_permutations=5,
        )


def test_book_guards() -> None:
    """Book rungs: no manifest + no override refuses to guess."""
    no_override = Book(borrow_bps_annual=500.0, funding_bps_annual=0.0)
    with pytest.raises(ValueError, match="refuses to guess"):
        no_override._resolve_periods(_demo_md())


def test_strategy_guards() -> None:
    """StrategySpec rungs: empty name, empty legs, non-LegSpec entry."""
    leg = LegSpec(
        name="l",
        direction="long",
        entry=EntrySpec(Compare("mom", ">", 0.0)),
        exit=ExitSpec(stop=StopSpec("atr", 2.0)),
        instrument=InstrumentRef("field", {"field": "close"}),
        features_used=("mom", "atr", "close"),
        rationale="x",
    )
    with pytest.raises(ValueError, match="name"):
        StrategySpec(
            name="",
            legs=(leg,),
            portfolio=PortfolioPolicy(
                borrow_bps_annual=500.0, funding_bps_annual=100.0
            ),
        )
    with pytest.raises(ValueError, match=r"at least one|legs"):
        StrategySpec(
            name="x",
            legs=(),
            portfolio=PortfolioPolicy(
                borrow_bps_annual=500.0, funding_bps_annual=100.0
            ),
        )
    with pytest.raises(ValueError, match="must be LegSpec"):
        StrategySpec(
            name="x",
            legs=(object(),),  # type: ignore[arg-type]
            portfolio=PortfolioPolicy(
                borrow_bps_annual=500.0, funding_bps_annual=100.0
            ),
        )
    with pytest.raises(ValueError, match=r"duplicate|unique"):
        StrategySpec(
            name="x",
            legs=(leg, leg),
            portfolio=PortfolioPolicy(
                borrow_bps_annual=500.0, funding_bps_annual=100.0
            ),
        )
    result_md = _demo_md()
    strategy = StrategySpec(
        name="attr",
        legs=(leg,),
        portfolio=PortfolioPolicy(
            borrow_bps_annual=500.0,
            funding_bps_annual=100.0,
            periods_override=252.0,
        ),
    )
    outcome = run_strategy(strategy, result_md)
    # corrupt attribution: rename a record's leg after the fact
    outcome["records"]["l"][0]["leg_name"] = "ghost"
    with pytest.raises(ValueError, match="attribution"):
        _recount_and_verify(strategy, outcome["records"], outcome["per_leg"])
