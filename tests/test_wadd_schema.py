"""Tests for the declared particle-filter record schema (wadd_schema)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from wadd_schema import (D2_KEYS, PAIR_KEYS, RUN_KEYS, TUBE_KEYS, validate_d2_record,
                         validate_record)


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


def _d2_record(K=2, D=6):
    return {"days": np.array([8.0, 9.0]),
            "populations": np.array([f"f{i}" for i in range(K)]),
            "lam": 10.0, "lam_I": 0.5, "lam_pi": 1.0, "eps": 0.0,
            "support": "positive_orthant", "gamma": 0.5, "budget": 4, "seed": 0,
            "d2_days": np.array([8.0, 8.5, 9.0]), "d2_alpha": 0.5, "d2_eps": 0.2,
            "d2_comp": np.full((D, K), 1.0 / K), "d2_total_mass": np.ones(D),
            "d2_real": np.full(K, 1.0 / K), "d2_real_n": 4,
            "d2_comp_t1": np.full(K, 1.0 / K), "d2_comp_t3": np.full(K, 1.0 / K),
            "d2_diag": "[]"}


def test_d2_conforming_record_passes():
    validate_d2_record(_d2_record())


def test_d2_missing_key_is_named():
    out = _d2_record()
    del out["d2_real_n"]
    with pytest.raises(ValueError, match="d2_real_n"):
        validate_d2_record(out)


def test_d2_rowsum_violation_is_caught():
    out = _d2_record()
    out["d2_comp"] = np.full((6, 2), 0.7)
    with pytest.raises(ValueError, match="summing to 1"):
        validate_d2_record(out)


def test_d2_alpha_inconsistency_is_caught():
    out = _d2_record()
    out["d2_alpha"] = 0.25
    with pytest.raises(ValueError, match="d2_alpha"):
        validate_d2_record(out)


def test_d2_pair_triplet_mismatch_is_caught():
    out = _d2_record()
    out["days"] = np.array([8.0, 9.5])
    with pytest.raises(ValueError, match="d2_days"):
        validate_d2_record(out)


def test_d2_draw_misalignment_is_caught():
    out = _d2_record()
    out["d2_total_mass"] = np.ones(5)                       # D = 6 elsewhere
    with pytest.raises(ValueError, match="leading dim"):
        validate_d2_record(out)


def test_d2_declared_keys_cover_helper():
    """The D2 fake record builder and the schema must not drift apart."""
    out = _d2_record()
    for key in list(RUN_KEYS) + list(D2_KEYS):
        assert key in out
