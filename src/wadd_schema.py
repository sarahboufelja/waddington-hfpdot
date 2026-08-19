"""Declared schemas of the pipeline records -- the single source of truth.

Every key a producer writes is declared here with its shape contract and its consumers.
The process rule this module encodes: a new consumer need is a schema change FIRST, then
a producer change -- never an ad-hoc key bolted onto a loop. ``validate_record`` gates
every ``np.savez`` in the particle filter and ``validate_d2_record`` every D2 coverage
record, so a record that would silently starve a consumer fails at write time, not at
read time.

Shape symbols: P particles, D draws kept per pair, K fates, II/JJ source/target support
sizes, Q quantile levels (2.5/50/97.5%), M identity resamples, n_day staged cells per
day, N real cells behind an observed composition.
"""
from __future__ import annotations

import numpy as np

#: run-level keys, written once per record
RUN_KEYS = {
    "days":        "window day grid",
    "populations": "(K,) fate names -- K is defined by this",
    "lam":         "marginal penalty lambda_1 = lambda_2",
    "lam_I":       "marginal penalty increment lambda_I_1 = lambda_I_2",
    "lam_pi":      "ideal-design term weight; 1 = HFPD-OT Def 2, else DIAGNOSTIC record",
    "eps":         "entropic eps; <= 0 means per-pair C-range/33 (see eps_{tag})",
    "support":     "simplex | positive_orthant",
    "gamma":       "centered-ridge scale (certified 0.5)",
    "budget":      "cells per day",
    "seed":        "staging + sampler seed",
}

#: per-pair keys; ``{tag}`` is "a->b" with %g-formatted days
PAIR_KEYS = {
    "eps_{tag}":                 "scalar; realised per-pair epsilon",
    "tables_q_{tag}":            "(Q, K, K); band quantiles [consumers: fle_plan_bands, reports]",
    "table_log_mass_{tag}":      "(D,); per-draw log mass, aligns every per-draw key",
    "tables_{tag}":              "(D, K, K); RAW fate tables [W-series joint draw-wise eval, "
                                 "Dobrushin-with-bands via genealogy, E14 plan term]",
    "ancestor_pr_{tag}":         "(D, K); effective ancestor count per target fate [W3]",
    "dom_dest_{tag}":            "(D, K); mass-weighted dominant-destination histogram [W1]",
    "ancestor_overlap_{tag}":    "(D, K, K); Bhattacharyya overlap of ancestor profiles [W2c]",
    "table_identity_mean_{tag}": "(D, K, K); E over M identity resamples of the table [E14]",
    "table_identity_var_{tag}":  "(D, K, K); Var over M identity resamples [E14 identity term]",
    "growth_q_{tag}":            "(Q, II); per-cell growth factor quantiles [diagnostic]",
    "growth_fate_q_{tag}":       "(Q, K); per-fate growth quantiles [diagnostic]",
    "growth_rel_q_{tag}":        "(Q, II); relative growth [REPORTABLE, W9]",
    "growth_fate_rel_q_{tag}":   "(Q, K); relative per-fate growth [REPORTABLE, W9]",
    "mass_ratio_q_{tag}":        "(Q,); per-step total-mass ratio [chart-artifact diagnostic]",
    "ensemble_{tag}":            "(P, JJ); trimmed particle marginals (masses) [resume]",
    "ensemble_mass_{tag}":       "(P,); particle log masses [resume]",
    "parents_{tag}":             "(P,); parent indices, -1 for roots [resume, genealogy]",
    "pool_{tag}":                "(D, JJ); pre-trim future marginals [videos, L2 overlay]",
    "pool_mass_{tag}":           "(D,); pool log masses",
    "pool_parents_{tag}":        "(D,); pool parent indices [genealogy composition]",
    "nu0_fates_{tag}":           "(K,); uniform-marginal fate projection at the target day",
    "diag_{tag}":                "json str; per-particle sampler gates (R-hat, ESS, eBFMI, acc)",
}

#: per-day keys; ``{day}`` is the %g-formatted target day of a completed pair
DAY_KEYS = {
    "W_fates_{day}": "(n_day, K); mean-z responsibilities q(c|z) [tables, projections]",
}

#: written once, at window completion only
TUBE_KEYS = {
    "tube_day":       "(n_days,)",
    "tube_eta_prop":  "(n_days,); propagated KL-ball radius, fate simplex",
    "tube_tv_diam":   "(n_days,); pairwise-TV diameter, fate simplex",
    "tube_eta_cells": "(n_days,); KL-ball radius, cell resolution",
    "tube_tv_cells":  "(n_days,); pairwise-TV diameter, cell resolution",
}

#: D2 held-out coverage record, one per triplet (t1, t2, t3); the runner samples the
#: DIRECT (t1, t3) pair (never composed -- no CK gate) and stores per-draw mid-point
#: compositions; multinomial noise is applied by the reader, never stored
D2_KEYS = {
    "d2_days":       "(3,); the triplet (t1, t2, t3) in real time stamps",
    "d2_alpha":      "scalar; interpolation weight (t2 - t1) / (t3 - t1)",
    "d2_eps":        "scalar; realised per-pair epsilon (C-range / 33 when eps <= 0)",
    "d2_comp":       "(D, K); per-draw mid-point compositions p-hat, rows sum to 1 "
                     "[coverage reader: predictive via Multinomial(N, p-hat)]",
    "d2_total_mass": "(D,); raw plan mass per draw before normalisation [scale diagnostic]",
    "d2_real":       "(K,); observed soft composition m-bar at t2 [coverage reader]",
    "d2_real_n":     "scalar int; real cells behind m-bar -- the multinomial N",
    "d2_comp_t1":    "(K,); observed composition at t1 [persistence baseline]",
    "d2_comp_t3":    "(K,); observed composition at t3 [linear-mixture baseline]",
    "d2_diag":       "json str; sampler gates (R-hat, ESS, eBFMI, acc) for the (t1, t3) pair",
}

#: per-draw families whose leading dimension must agree within a pair
_PER_DRAW = ("table_log_mass_{tag}", "tables_{tag}", "ancestor_pr_{tag}", "dom_dest_{tag}",
             "ancestor_overlap_{tag}", "table_identity_mean_{tag}", "table_identity_var_{tag}",
             "pool_{tag}", "pool_mass_{tag}", "pool_parents_{tag}")
#: (K, K)-shaped per-draw keys
_KK = ("tables_{tag}", "ancestor_overlap_{tag}", "table_identity_mean_{tag}",
       "table_identity_var_{tag}")


def validate_record(out: dict, tags, complete: bool = False) -> None:
    """Raise ValueError naming every schema violation; silent on a conforming record.

    ``tags`` are the COMPLETED pair tags (always a prefix of the window); ``complete``
    additionally requires the tube arrays. Values may be numpy arrays or plain scalars
    (as passed to ``np.savez``).
    """
    problems = []
    for key in RUN_KEYS:
        if key not in out:
            problems.append(f"missing run key '{key}'")
    K = len(np.asarray(out["populations"])) if "populations" in out else None

    for tag in tags:
        keys = {t.format(tag=tag): t for t in PAIR_KEYS}
        missing = [k for k in keys if k not in out]
        problems += [f"missing '{k}' (pair {tag})" for k in missing]
        day = tag.split("->")[1]
        if f"W_fates_{day}" not in out:
            problems.append(f"missing 'W_fates_{day}' (pair {tag})")
        if missing or K is None:
            continue
        D = len(np.asarray(out[f"table_log_mass_{tag}"]))
        for tpl in _PER_DRAW:
            k = tpl.format(tag=tag)
            if len(np.asarray(out[k])) != D:
                problems.append(f"'{k}' leading dim {len(np.asarray(out[k]))} != draws {D}")
        for tpl in _KK:
            k = tpl.format(tag=tag)
            if np.asarray(out[k]).shape[1:] != (K, K):
                problems.append(f"'{k}' trailing shape {np.asarray(out[k]).shape[1:]} != (K, K)")
        for k in (f"ancestor_pr_{tag}", f"dom_dest_{tag}"):
            if np.asarray(out[k]).shape[1:] != (K,):
                problems.append(f"'{k}' trailing shape != (K,)")
        for k in (f"tables_q_{tag}", f"growth_fate_q_{tag}", f"growth_fate_rel_q_{tag}"):
            if len(np.asarray(out[k])) != 3:
                problems.append(f"'{k}' must carry 3 quantile levels")
        P = len(np.asarray(out[f"ensemble_mass_{tag}"]))
        if not (len(np.asarray(out[f"ensemble_{tag}"])) == P
                == len(np.asarray(out[f"parents_{tag}"]))):
            problems.append(f"ensemble keys for {tag} disagree on particle count")

    if complete:
        lens = set()
        for key in TUBE_KEYS:
            if key not in out:
                problems.append(f"missing tube key '{key}' on a complete record")
            else:
                lens.add(len(np.asarray(out[key])))
        if len(lens) > 1:
            problems.append(f"tube arrays disagree on length: {sorted(lens)}")

    if problems:
        raise ValueError("record schema violations:\n  " + "\n  ".join(problems))


def validate_d2_record(out: dict) -> None:
    """Raise ValueError naming every D2 schema violation; silent on a conforming record.

    Missing keys are reported first (and alone); shape and consistency checks run once
    the record is key-complete. ``days`` must equal the sampled (t1, t3) pair of
    ``d2_days`` and ``d2_alpha`` must be the mid-point weight implied by the stamps.
    """
    problems = [f"missing key '{k}'" for k in list(RUN_KEYS) + list(D2_KEYS) if k not in out]
    if problems:
        raise ValueError("D2 record schema violations:\n  " + "\n  ".join(problems))

    K = len(np.asarray(out["populations"]))
    days3 = np.asarray(out["d2_days"], dtype=float)
    if days3.shape != (3,) or not (days3[0] < days3[1] < days3[2]):
        problems.append(f"'d2_days' must be 3 increasing stamps, got {days3}")
    else:
        if not np.allclose(np.asarray(out["days"], dtype=float), days3[[0, 2]]):
            problems.append(f"'days' {np.asarray(out['days'])} != sampled pair (t1, t3) "
                            "of 'd2_days'")
        alpha = float(out["d2_alpha"])
        if not np.isclose(alpha, (days3[1] - days3[0]) / (days3[2] - days3[0])):
            problems.append(f"'d2_alpha' {alpha:g} inconsistent with 'd2_days'")
    comp = np.asarray(out["d2_comp"], dtype=float)
    if comp.ndim != 2 or comp.shape[1] != K:
        problems.append(f"'d2_comp' shape {comp.shape} != (D, K)")
    else:
        if len(np.asarray(out["d2_total_mass"])) != len(comp):
            problems.append("'d2_total_mass' leading dim != draws of 'd2_comp'")
        if comp.min() < -1e-9 or not np.allclose(comp.sum(axis=1), 1.0, atol=1e-6):
            problems.append("'d2_comp' rows must be compositions summing to 1")
    for k in ("d2_real", "d2_comp_t1", "d2_comp_t3"):
        v = np.asarray(out[k], dtype=float)
        if v.shape != (K,):
            problems.append(f"'{k}' shape {v.shape} != (K,)")
        elif v.min() < -1e-9 or not np.isclose(v.sum(), 1.0, atol=1e-6):
            problems.append(f"'{k}' must be a composition summing to 1")
    if int(out["d2_real_n"]) <= 0:
        problems.append("'d2_real_n' must be a positive cell count")

    if problems:
        raise ValueError("D2 record schema violations:\n  " + "\n  ".join(problems))
