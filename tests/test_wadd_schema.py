"""Tests for the declared particle-filter record schema (wadd_schema)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wadd_schema import PAIR_KEYS, RUN_KEYS, TUBE_KEYS, validate_record


def _record(tags=("1->2",), K=2, D=6, P=3, JJ=4, II=4, complete=False):
    out = {"days": np.array([1.0, 2.0]), "populations": np.array([f"f{i}" for i in range(K)]),
           "lam": 10.0, "lam_I": 0.5, "lam_pi": 1.0, "eps": 0.0,
           "support": "positive_orthant", "gamma": 0.5, "budget": II, "seed": 0}
    for tag in tags:
        out[f"eps_{tag}"] = 0.2
        out[f"tables_q_{tag}"] = np.zeros((3, K, K))
        out[f"table_log_mass_{tag}"] = np.zeros(D)
        for k in ("tables", "ancestor_overlap", "table_identity_mean", "table_identity_var"):
            out[f"{k}_{tag}"] = np.zeros((D, K, K))
        out[f"ancestor_pr_{tag}"] = np.zeros((D, K))
        out[f"dom_dest_{tag}"] = np.zeros((D, K))
        for k in ("growth_q", "growth_rel_q"):
            out[f"{k}_{tag}"] = np.zeros((3, II))
        for k in ("growth_fate_q", "growth_fate_rel_q"):
            out[f"{k}_{tag}"] = np.zeros((3, K))
        out[f"mass_ratio_q_{tag}"] = np.zeros(3)
        out[f"ensemble_{tag}"] = np.zeros((P, JJ))
        out[f"ensemble_mass_{tag}"] = np.zeros(P)
        out[f"parents_{tag}"] = np.zeros(P)
        out[f"pool_{tag}"] = np.zeros((D, JJ))
        out[f"pool_mass_{tag}"] = np.zeros(D)
        out[f"pool_parents_{tag}"] = np.zeros(D)
        out[f"nu0_fates_{tag}"] = np.zeros(K)
        out[f"diag_{tag}"] = "[]"
        out[f"W_fates_{tag.split('->')[1]}"] = np.zeros((JJ, K))
    if complete:
        for key in TUBE_KEYS:
            out[key] = np.zeros(len(tags) + 1)
    return out


def test_conforming_record_passes():
    validate_record(_record(), ["1->2"])
    validate_record(_record(complete=True), ["1->2"], complete=True)


def test_empty_prefix_checks_run_keys_only():
    validate_record(_record(tags=()), [])
    with pytest.raises(ValueError, match="lam_pi"):
        out = _record(tags=())
        del out["lam_pi"]
        validate_record(out, [])


@pytest.mark.parametrize("key", ["tables_1->2", "dom_dest_1->2", "ancestor_overlap_1->2",
                                 "table_identity_var_1->2", "growth_rel_q_1->2", "W_fates_2"])
def test_missing_consumer_key_is_named(key):
    out = _record()
    del out[key]
    with pytest.raises(ValueError, match=key.replace("(", "").split("_1")[0]):
        validate_record(out, ["1->2"])


def test_draw_misalignment_is_caught():
    out = _record()
    out["dom_dest_1->2"] = np.zeros((5, 2))                 # D = 6 elsewhere
    with pytest.raises(ValueError, match="leading dim"):
        validate_record(out, ["1->2"])


def test_wrong_fate_shape_is_caught():
    out = _record()
    out["tables_1->2"] = np.zeros((6, 3, 3))                # K = 2 per populations
    with pytest.raises(ValueError, match="K, K"):
        validate_record(out, ["1->2"])


def test_incomplete_tube_is_caught():
    out = _record(complete=True)
    del out["tube_eta_prop"]
    with pytest.raises(ValueError, match="tube_eta_prop"):
        validate_record(out, ["1->2"], complete=True)


def test_declared_keys_cover_helper():
    """The fake record builder and the schema must not drift apart."""
    out = _record(complete=True)
    for key in RUN_KEYS:
        assert key in out
    for tpl in PAIR_KEYS:
        assert tpl.format(tag="1->2") in out
