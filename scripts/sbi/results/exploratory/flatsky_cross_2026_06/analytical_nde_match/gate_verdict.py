#!/usr/bin/env python3
"""Parameterized GATE-C verdict reader (same logic as run_laneB_gate_c.py, but path-agnostic).

Reads TARP curves + posterior dumps under a given gate dir and prints, per arm:
  worst-tercile signed dev (dim3), net signed bias (mean±std), SBC rank std, verdict.
Thresholds identical to run_laneB_gate_c.py: DEV_PASS 0.05, DEV_CAVEAT 0.10, SBC std band
[0.275,0.305]. Verdict: PASS / PASS-with-caveat / FAIL.

Usage:
  python gate_verdict.py --gate-dir <DIR> --arms pair2d_rnvp l1product_rnvp ...
where <DIR> contains tarp_drp/curves/ and tarp_drp/dumps/ (as produced by
tarp_stratified_val_nde.py + run_tarp_coverage.py).
"""
import argparse, glob, json
import numpy as np

DEV_PASS, DEV_CAVEAT = 0.05, 0.10
STD_LO, STD_HI, STD_FAIL = 0.275, 0.305, 0.02
TERCILES = ("LOW", "MID", "HIGH")


def sbc_from_dumps(gc, arm):
    rs = []
    for f in sorted(glob.glob(f"{gc}/tarp_drp/dumps/{arm}_*/seed_*/n*_m*/posterior_samples.npz")):
        z = np.load(f); rs.append((z["samples"] < z["theta"][:, None, :]).mean(1))
    if not rs:
        return None
    from scipy import stats as st
    r = np.concatenate(rs, 0)
    return {"n": int(r.shape[0]), "mean": [float(m) for m in r.mean(0)[:3]],
            "std": [float(s) for s in r.std(0)[:3]],
            "min_ks_p": float(min(st.kstest(r[:, i], "uniform").pvalue for i in range(3)))}


def tarp_devs(gc, arm, dim=3):
    out = {}
    for terc in TERCILES:
        worst = None
        for f in sorted(glob.glob(f"{gc}/tarp_drp/curves/tarp_curve_{arm}_{terc}_seed*_dim{dim}.npz")):
            z = np.load(f); a = np.asarray(z["alpha"]); e = np.asarray(z["ecp_bootstrap"]).mean(0)
            i = int(np.argmax(np.abs(e - a))); d = float(e[i] - a[i])
            if worst is None or abs(d) > abs(worst):
                worst = d
        if worst is not None:
            out[terc] = worst
    return out


def net_bias(gc, arm, dim=3):
    devs = []
    for f in glob.glob(f"{gc}/tarp_drp/curves/tarp_curve_{arm}_*_seed*_dim{dim}.npz"):
        z = np.load(f); a = z["alpha"]; e = z["ecp_bootstrap"].mean(0)
        devs.append(float(np.trapz(e - a, a) * 2))
    return (float(np.mean(devs)), float(np.std(devs))) if devs else (None, None)


def verdict(devs, sbc):
    if not devs or not sbc:
        return "INCOMPLETE"
    worst = max(abs(d) for d in devs.values())
    stds = sbc["std"]
    std_off = max((max(0.0, STD_LO - s, s - STD_HI) for s in stds), default=9)
    if worst > DEV_CAVEAT or std_off >= STD_FAIL:
        return "FAIL"
    if worst <= DEV_PASS and std_off == 0.0:
        return "PASS"
    return "PASS-with-caveat"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gate-dir", required=True)
    p.add_argument("--arms", nargs="+", required=True)
    p.add_argument("--json-out", default=None)
    a = p.parse_args()
    rows = []
    for arm in a.arms:
        devs = tarp_devs(a.gate_dir, arm)
        sbc = sbc_from_dumps(a.gate_dir, arm)
        nb = net_bias(a.gate_dir, arm)
        v = verdict(devs, sbc)
        rows.append(dict(arm=arm, devs=devs, net_bias=nb,
                         sbc_std=(sbc["std"] if sbc else None),
                         sbc_mean=(sbc["mean"] if sbc else None), verdict=v))
        dstr = " ".join(f"{k}{devs[k]:+.3f}" for k in TERCILES if k in devs) if devs else "—"
        nbstr = f"{nb[0]:+.3f}±{nb[1]:.3f}" if nb[0] is not None else "—"
        sstr = "/".join(f"{s:.3f}" for s in sbc["std"]) if sbc else "—"
        print(f"{arm:28s} | TARP {dstr} | net {nbstr} | SBCstd {sstr} | {v}")
    if a.json_out:
        json.dump(rows, open(a.json_out, "w"), indent=2)
        print(f"-> {a.json_out}")


if __name__ == "__main__":
    main()
