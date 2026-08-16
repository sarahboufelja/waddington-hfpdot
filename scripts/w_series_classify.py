"""W-series robustness classification under the certified posterior -- pre-registered.

Pure reader: no coupling is computed or recomputed anywhere. Inputs are the four
growth-aware particle-filter window records (adopted by config match + schema, exactly as
the orchestrator does), the deterministic lineage baselines at the locked epsilon grid,
and the CK-ladder diagnostic at the composed span. Outputs one classification per locked
claim row with its posterior probability P (at the 0.90 threshold and the 0.95
sensitivity column), 95% credible effect sizes, and every gate/check readout, to
w_series/<stamp>/classification.json plus a printed table.

Locked semantics (paper claims ledger, rows LOCKED 2026-08-16; changing any LOCK value
is a protocol violation and must be reported as such):
- ROBUST: the certainty-equivalent conclusion holds JOINTLY in >= 90% of posterior
  draws; effect size = 95% CrI. Checked in order: NON-IDENTIFIED (composed claims only)
  -> REGULARISATION-DEPENDENT (epsilon flip on the deterministic baselines) ->
  RESOLUTION-LIMITED -> ROBUST.
- Cross-day functionals use INDEPENDENT-DRAW CHAINS: one stored draw table per step,
  sampled proportionally to its posterior mass, composed by matrix product (fate level,
  row-normalised) -- the same operation the C1 analysis applies to the deterministic
  tables. Only W2a composes; the W2b/W2c day posteriors are single-pair reads chained
  only through draw selection.
- The NON-IDENTIFIED gate (W2a): the CK-ladder direct-vs-composed gap at the claim span
  must not exceed the 95% credible TV radius of the composed functional.

Run from the repo root (first execution = the pre-registered classification run):
    python scripts/w_series_classify.py [run_dir]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import numpy as np

import run_e2e_hfpdot as E2E
from fle_plan_bands import coarse_of
from gmvae_confusion import latest_run
from wadd_artifacts import artifact_dir
from wadd_propagation import weighted_quantiles

#: LOCKED 2026-08-16 -- pre-registration ledger, paper claims doc. Do not edit.
LOCK = dict(date="2026-08-16", s_w1=0.80, tau=0.05, b_w2c=0.50, r_w3=0.10,
            robust_p=0.90, sensitivity_p=0.95, eps_grid=(0.1, 0.2),
            w1_span=(0.0, 4.0), w2a_span=(8.0, 12.0), n_chains=400, chain_seed=20260816)

QS = [0.025, 0.5, 0.975]


def _w(log_mass):
    m = np.exp(log_mass - np.max(log_mass))
    return m / m.sum()


def _tags_between(days, lo, hi):
    return [(a, b, f"{a:g}->{b:g}") for a, b in zip(days, days[1:]) if lo <= a and b <= hi]


def _marginal(T):
    """Arrival composition at the target day per draw: column sums, normalised."""
    col = T.sum(axis=1)
    return col / np.maximum(col.sum(axis=1, keepdims=True), 1e-300)


def _classify(P, reg_flipped, gate_failed=False):
    if gate_failed:
        return "NON-IDENTIFIED"
    if reg_flipped:
        return "REGULARISATION-DEPENDENT"
    if P >= LOCK["robust_p"]:
        return "ROBUST"
    return "RESOLUTION-LIMITED"


class Store:
    """Adopted inputs, loaded once."""

    def __init__(self, run_dir, budget):
        self.run_dir = Path(run_dir)
        self.recs, self.days = {}, {}
        for days in E2E.WINDOWS:
            cfg = dict(E2E.PF_DEFAULTS, days=list(days), budget=budget)
            child = E2E._adopt(self.run_dir, "particle_filter", cfg, E2E._pf_complete(days))
            if child is None:
                raise SystemExit(f"no complete window record for D{days[0]:g}-D{days[-1]:g}; "
                                 "run the orchestrator first")
            name = f"D{days[0]:g}-D{days[-1]:g}"
            self.recs[name] = np.load(E2E._record_of(child), allow_pickle=True)
            self.days[name] = list(days)
        self.pops = [str(p) for p in self.recs["D0-D6"]["populations"]]
        self.coarse = [coarse_of(p) for p in self.pops]
        self.lineage = {}
        for eps in LOCK["eps_grid"]:
            child = E2E._adopt(self.run_dir, "lineage", dict(budget=budget, reg=eps),
                               E2E._lineage_complete)
            self.lineage[eps] = (np.load(child / "transition_tables.npz", allow_pickle=True)
                                 if child else None)

    def idx(self, name):
        return self.pops.index(name)

    def group_idx(self, coarse_name):
        return [i for i, c in enumerate(self.coarse) if c == coarse_name]

    def ck_gate(self, lo, hi, eps=0.1):
        """CK-ladder direct-vs-composed gap at the span, if the diagnostic exists."""
        for npz in sorted((self.run_dir / "ck_ladder").glob("*/ck_ladder.npz"), reverse=True):
            z = np.load(npz)
            if (float(z["day_from"]) == lo and float(z["day_to"]) == hi
                    and abs(float(z["epsilon"]) - eps) < 1e-9):
                return float(np.asarray(z["vs_direct"])[-1]), str(npz)
        return None, (f".venv/bin/python scripts/diag_ck_ladder.py --from-day {lo:g} "
                      f"--to-day {hi:g} --eps {eps} --budget 500")


def w1(store):
    """Initial-phase bimodality: vote-histogram top-2 = {Stromal, MET}, joint share >= s."""
    rec, days = store.recs["D0-D6"], store.days["D0-D6"]
    iS, iM = store.idx("Stromal"), store.idx("MET")
    rows = []
    for a, b, tag in _tags_between(days, *LOCK["w1_span"]):
        dd = np.asarray(rec[f"dom_dest_{tag}"])
        lm = np.asarray(rec[f"table_log_mass_{tag}"])
        top2 = np.argsort(dd, axis=1)[:, -2:]
        share = dd[:, iS] + dd[:, iM]
        ok = np.array([set(t) == {iS, iM} for t in top2]) & (share >= LOCK["s_w1"])
        P = float(np.sum(_w(lm) * ok))
        land = _marginal(np.asarray(rec[f"tables_{tag}"]))
        land_share = land[:, iS] + land[:, iM]
        reg = _reg_w1(store, a, b)
        rows.append(dict(pair=tag, P=P, P_sens=P >= LOCK["sensitivity_p"],
                         cls=_classify(P, reg_flipped=reg["flipped"]),
                         vote_share_cri=weighted_quantiles(share[:, None], lm, QS)[:, 0].tolist(),
                         landing_share_cri=weighted_quantiles(land_share[:, None], lm, QS)[:, 0].tolist(),
                         reg_check=reg))
    return rows


def _reg_w1(store, a, b):
    out = dict(note="landing-share resolution (deterministic baseline stores fate tables)")
    verdicts = {}
    for eps, z in store.lineage.items():
        if z is None:
            return dict(flipped=False, note="lineage baseline missing", verdicts={})
        pairs = np.asarray(z["pairs"])
        hit = np.where((pairs[:, 0] == a) & (pairs[:, 1] == b))[0]
        if not len(hit):
            return dict(flipped=False, note=f"pair D{a:g}->D{b:g} not in lineage grid",
                        verdicts={})
        T = np.asarray(z["matrices"][hit[0]])
        col = T.sum(axis=0) / max(T.sum(), 1e-300)
        iS, iM = store.idx("Stromal"), store.idx("MET")
        top2 = set(np.argsort(col)[-2:])
        verdicts[eps] = bool(top2 == {iS, iM} and col[iS] + col[iM] >= LOCK["s_w1"])
    out.update(flipped=len(set(verdicts.values())) > 1, verdicts={str(k): v for k, v in verdicts.items()})
    return out


def _chain_shares(store, rec, steps, rng, project=None):
    """Independent-draw chains over ``steps``: per chain, the composed MET row (fate
    shares) after the last step. ``steps`` = [(a, b, tag), ...] in order."""
    Rs, ws = [], []
    for _, _, tag in steps:
        T = np.asarray(rec[f"tables_{tag}"], dtype=float)
        rows = T.sum(axis=2, keepdims=True)
        Rs.append(np.divide(T, rows, out=np.zeros_like(T), where=rows > 0))
        ws.append(_w(np.asarray(rec[f"table_log_mass_{tag}"])))
    iMET = store.idx("MET")
    K = len(store.pops)
    out = np.empty((LOCK["n_chains"], K))
    for n in range(LOCK["n_chains"]):
        v = np.zeros(K)
        v[iMET] = 1.0
        for R, w in zip(Rs, ws):
            v = v @ R[rng.choice(len(w), p=w)]
            v = v / max(v.sum(), 1e-300)
        out[n] = v
    return out


def w2a(store):
    """Post-dox diversity: composed MET row D8->D12 contains all four coarse groups."""
    rec, days = store.recs["D6-D12"], store.days["D6-D12"]
    steps = _tags_between(days, *LOCK["w2a_span"])
    rng = np.random.default_rng(LOCK["chain_seed"])
    V = _chain_shares(store, rec, steps, rng)
    groups = {g: store.group_idx(g) for g in ("IPS", "Trophoblast", "Neural", "Epithelial")}
    shares = {g: V[:, ix].sum(axis=1) for g, ix in groups.items()}
    ok = np.all([s >= LOCK["tau"] for s in shares.values()], axis=0)
    P = float(np.mean(ok))
    gap, gate_src = store.ck_gate(*LOCK["w2a_span"])
    vbar = V.mean(axis=0)
    radius = float(np.quantile(0.5 * np.abs(V - vbar).sum(axis=1), 0.95))
    gate_failed = None if gap is None else gap > radius
    reg = _reg_w2a(store, steps, groups)
    cls = ("GATE-PENDING" if gap is None
           else _classify(P, reg_flipped=reg["flipped"], gate_failed=gate_failed))
    return dict(claim="W2a", cls=cls, P=P, P_sens=P >= LOCK["sensitivity_p"],
                group_share_cri={g: np.quantile(s, QS).tolist() for g, s in shares.items()},
                gate=dict(direct_vs_composed=gap, credible_tv_radius=radius,
                          source=gate_src, failed=gate_failed),
                reg_check=reg)


def _reg_w2a(store, steps, groups):
    verdicts = {}
    for eps, z in store.lineage.items():
        if z is None:
            return dict(flipped=False, note="lineage baseline missing", verdicts={})
        pairs = np.asarray(z["pairs"])
        v = np.zeros(len(store.pops))
        v[store.idx("MET")] = 1.0
        for a, b, _ in steps:
            hit = np.where((pairs[:, 0] == a) & (pairs[:, 1] == b))[0]
            if not len(hit):
                return dict(flipped=False, note="span not fully in lineage grid", verdicts={})
            T = np.asarray(z["matrices"][hit[0]])
            rows = T.sum(axis=1, keepdims=True)
            v = v @ np.divide(T, rows, out=np.zeros_like(T), where=rows > 0)
            v = v / max(v.sum(), 1e-300)
        verdicts[eps] = bool(all(v[ix].sum() >= LOCK["tau"] for ix in groups.values()))
    return dict(flipped=len(set(verdicts.values())) > 1,
                verdicts={str(k): v for k, v in verdicts.items()})


def _day_posterior(store, windows, lo, hi, per_draw_stat, crossing):
    """Independent chains over day pairs in [lo, hi]: first target day where
    ``crossing(per_draw_stat(draw))`` -- np.inf when never crossed in range."""
    steps = []
    for name in windows:
        rec, days = store.recs[name], store.days[name]
        for a, b, tag in _tags_between(days, lo, hi):
            T = np.asarray(rec[f"tables_{tag}"], dtype=float)
            steps.append((b, per_draw_stat(T), _w(np.asarray(rec[f"table_log_mass_{tag}"]))))
    rng = np.random.default_rng(LOCK["chain_seed"] + 1)
    out = np.full(LOCK["n_chains"], np.inf)
    for n in range(LOCK["n_chains"]):
        for b, stat, w in steps:
            if crossing(stat[rng.choice(len(w), p=w)]):
                out[n] = b
                break
    return out


def w2b(store):
    """Trophoblast route: epithelial waypoint by D9, trophoblast by D10.5, D12.5 CrI."""
    epi = store.group_idx("Epithelial")
    tro = store.group_idx("Trophoblast")
    day_epi = _day_posterior(store, ["D6-D12"], 6.0, 12.0,
                             lambda T: _marginal(T)[:, epi].sum(axis=1),
                             lambda s: s >= LOCK["tau"])
    day_tro = _day_posterior(store, ["D6-D12"], 6.0, 12.0,
                             lambda T: _marginal(T)[:, tro].sum(axis=1),
                             lambda s: s >= LOCK["tau"])
    P_epi, P_tro = float(np.mean(day_epi <= 9.0)), float(np.mean(day_tro <= 10.5))
    P = min(P_epi, P_tro)
    rec = store.recs["D12-D14"]
    T = np.asarray(rec["tables_12->12.5"], dtype=float)
    lm = np.asarray(rec["table_log_mass_12->12.5"])
    frac = _marginal(T)[:, tro].sum(axis=1)
    cri = weighted_quantiles(frac[:, None], lm, QS)[:, 0]
    return dict(claim="W2b", cls=_classify(P, reg_flipped=_reg_w2b(store, epi, tro)["flipped"]),
                P=P, P_sens=P >= LOCK["sensitivity_p"],
                P_epithelial_by_D9=P_epi, P_trophoblast_by_D10_5=P_tro,
                trophoblast_frac_D12_5_cri=cri.tolist(),
                vs_published_0_20=bool(cri[0] <= 0.20 <= cri[2]),
                scope="serum arm only; published claim covers both conditions",
                reg_check=_reg_w2b(store, epi, tro))


def _reg_w2b(store, epi, tro):
    verdicts = {}
    for eps, z in store.lineage.items():
        if z is None:
            return dict(flipped=False, note="lineage baseline missing", verdicts={})
        pairs, M = np.asarray(z["pairs"]), np.asarray(z["matrices"])
        d_epi = d_tro = np.inf
        for (a, b), T in zip(pairs, M):
            if not (6.0 <= a and b <= 12.0):
                continue
            col = T.sum(axis=0) / max(T.sum(), 1e-300)
            if d_epi is np.inf and col[epi].sum() >= LOCK["tau"]:
                d_epi = b
            if d_tro is np.inf and col[tro].sum() >= LOCK["tau"]:
                d_tro = b
        verdicts[eps] = bool(d_epi <= 9.0 and d_tro <= 10.5)
    return dict(flipped=len(set(verdicts.values())) > 1,
                verdicts={str(k): v for k, v in verdicts.items()})


def w2c(store):
    """Neural route: ancestor divergence by D9; epithelial->neural transition vs D12.5."""
    iN, iT, iI = store.idx("Neural"), store.idx("Trophoblast"), store.idx("IPS")
    b_lock = LOCK["b_w2c"]
    steps = []
    for name in ("D6-D12", "D12-D14"):
        rec, days = store.recs[name], store.days[name]
        for a, b, tag in _tags_between(days, 6.0, 14.0):
            O = np.asarray(rec[f"ancestor_overlap_{tag}"])
            steps.append((b, O, _w(np.asarray(rec[f"table_log_mass_{tag}"]))))
    rng = np.random.default_rng(LOCK["chain_seed"] + 2)
    div = np.full(LOCK["n_chains"], np.inf)
    for n in range(LOCK["n_chains"]):
        for b, O, w in steps:
            o = O[rng.choice(len(w), p=w)]
            if o[iN, iT] < b_lock and o[iN, iI] < b_lock:
                div[n] = b
                break
    P_div = float(np.mean(div <= 9.0))
    neu, epi = store.group_idx("Neural"), store.group_idx("Epithelial")
    trans = _day_posterior(store, ["D6-D12", "D12-D14"], 9.0, 14.0,
                           lambda T: (_marginal(T)[:, neu].sum(axis=1)
                                      - _marginal(T)[:, epi].sum(axis=1)),
                           lambda s: s > 0)
    finite = trans[np.isfinite(trans)]
    trans_cri = (np.quantile(finite, QS).tolist() if len(finite) >= 0.5 * len(trans)
                 else None)
    return dict(claim="W2c", cls=_classify(P_div, reg_flipped=False), P=P_div,
                P_sens=P_div >= LOCK["sensitivity_p"],
                divergence_day_cri=np.quantile(div[np.isfinite(div)], QS).tolist()
                if np.isfinite(div).any() else None,
                never_diverged_frac=float(np.mean(~np.isfinite(div))),
                transition_day_cri=trans_cri, vs_published_D12_5=12.5,
                reg_check=dict(flipped=False,
                               note="cell-level ancestor overlap has no deterministic "
                                    "fate-table analogue; REG check not applicable at "
                                    "the locked resolution"))


def w3(store):
    """iPSC ancestor bottleneck: PR/support <= r, with mass-share companion."""
    iI = store.idx("IPS")
    rows = []
    for name in ("D12-D14", "D14-D18"):
        rec, days = store.recs[name], store.days[name]
        for a, b, tag in _tags_between(days, days[0], days[-1]):
            pr = np.asarray(rec[f"ancestor_pr_{tag}"])[:, iI]
            lm = np.asarray(rec[f"table_log_mass_{tag}"])
            T = np.asarray(rec[f"tables_{tag}"], dtype=float)
            share = _marginal(T)[:, iI]
            med_share = float(weighted_quantiles(share[:, None], lm, [0.5])[0, 0])
            if med_share < 0.01:
                continue                       # numerically empty fate: PR undefined
            n_src = np.asarray(rec[f"growth_q_{tag}"]).shape[1]
            frac = pr / n_src
            P = float(np.sum(_w(lm) * (frac <= LOCK["r_w3"])))
            rows.append(dict(pair=tag, P=P, P_sens=P >= LOCK["sensitivity_p"],
                             cls=_classify(P, reg_flipped=False),
                             pr_over_support_cri=weighted_quantiles(frac[:, None], lm, QS)[:, 0].tolist(),
                             ips_mass_share_median=med_share))
    return rows


def main(run_dir, budget):
    store = Store(run_dir, budget)
    print(f"pops: {store.pops}")
    print(f"coarse mapping: { {p: c for p, c in zip(store.pops, store.coarse)} }")
    result = dict(lock=dict(LOCK, eps_grid=list(LOCK["eps_grid"]),
                            w1_span=list(LOCK["w1_span"]), w2a_span=list(LOCK["w2a_span"])),
                  W1=w1(store), W2a=w2a(store), W2b=w2b(store), W2c=w2c(store),
                  W3=w3(store),
                  W6=dict(cls="ROBUST", note="control: wet-lab validated, coupling-independent"),
                  W10=dict(cls="ROBUST", note="control: expression-only, coupling-independent"))
    out_dir = artifact_dir(Path(run_dir), "w_series",
                           config=dict(budget=budget, lock_date=LOCK["date"]))
    out = out_dir / "classification.json"
    out.write_text(json.dumps(result, indent=2, default=str))

    print(f"\n{'claim':>8} {'class':>26} {'P':>6}  detail")
    for r in result["W1"]:
        print(f"{'W1 ' + r['pair']:>8} {r['cls']:>26} {r['P']:>6.3f}  "
              f"vote share CrI {['%.2f' % v for v in r['vote_share_cri']]}")
    a = result["W2a"]
    print(f"{'W2a':>8} {a['cls']:>26} {a['P']:>6.3f}  gate {a['gate']}")
    b = result["W2b"]
    print(f"{'W2b':>8} {b['cls']:>26} {b['P']:>6.3f}  epi<=D9 {b['P_epithelial_by_D9']:.3f} "
          f"tro<=D10.5 {b['P_trophoblast_by_D10_5']:.3f} "
          f"D12.5 frac CrI {['%.3f' % v for v in b['trophoblast_frac_D12_5_cri']]}")
    c = result["W2c"]
    print(f"{'W2c':>8} {c['cls']:>26} {c['P']:>6.3f}  div-day CrI {c['divergence_day_cri']} "
          f"never {c['never_diverged_frac']:.2f} trans CrI {c['transition_day_cri']}")
    for r in result["W3"]:
        print(f"{'W3 ' + r['pair']:>8} {r['cls']:>26} {r['P']:>6.3f}  "
              f"PR/support CrI {['%.3f' % v for v in r['pr_over_support_cri']]} "
              f"(iPSC share {r['ips_mass_share_median']:.3f})")
    print(f"\nclassification -> {out}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("run_dir", nargs="?", default=None)
    p.add_argument("--budget", type=int, default=500)
    a = p.parse_args()
    main(a.run_dir or latest_run(), a.budget)
