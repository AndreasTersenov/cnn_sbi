#!/usr/bin/env python3
"""Proper-band calibration gate for the A1 (deeper/wider RealNVP) sweep.

For each config dir L*_H*/ : pool the gate stratified dumps (LOW+MID+HIGH) over the 3 NDE seeds -> the
reported un-stratified posterior -> sightline-bootstrap TARP net + 1-sigma, and SBC rank-std; read the
screen FoM3. Print a decision table vs the pre-registered rule:
  calibrated = TARP net in [-0.02,+0.02] AND SBC std in [0.282,0.296] all params (guard: net<-0.02 or
  SBC>0.305 = reject/over-confident). Targets: FoM3>=3371 clean win; [3304,3371) tie. CPU only.
"""
from pathlib import Path
import glob, json, re
import numpy as np
import tarp

OUT = Path("/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/cnn_phase/nde_expressivity_2026_06")
ALPHA = np.linspace(0.0, 1.0, 61)
NBOOT = 200
RNG = np.random.default_rng(0)
SEEDS = (41, 42, 43)
BASE_FOM3, JOINTL1 = 3304.0, 3371.0


def ecp_once(samp, theta):
    s = np.transpose(samp, (1, 0, 2))
    ecp, al = tarp.get_tarp_coverage(s, theta, references="random", norm=True)
    ecp = np.asarray(ecp); ecp = ecp.mean(0) if ecp.ndim == 2 else ecp
    return np.interp(ALPHA, np.asarray(al), ecp)


def pooled_unstrat(gate_dumps, arm):
    """LOW+MID+HIGH per seed -> 600 obs (same order across seeds); pool seeds along sample axis."""
    per_seed = []
    theta = None
    for s in SEEDS:
        S, T = [], []
        for terc in ("LOW", "MID", "HIGH"):
            g = glob.glob(f"{gate_dumps}/{arm}_{terc}/seed_{s}/n*_m*/posterior_samples.npz")
            if not g:
                return None, None
            z = np.load(g[0]); S.append(z["samples"][:, :, :3]); T.append(z["theta"][:, :3])
        per_seed.append(np.concatenate(S, 0).astype(np.float32))
        if theta is None:
            theta = np.concatenate(T, 0).astype(np.float32)
    return np.concatenate(per_seed, axis=1), theta            # (600, 3*M, 3), (600,3)


def calib(samp, theta):
    N = theta.shape[0]
    boot = np.array([ecp_once(samp[i := RNG.integers(0, N, N)], theta[i]) for _ in range(NBOOT)])
    mean = boot.mean(0)
    net = float(np.trapz(mean - ALPHA, ALPHA) * 2)
    se = float(boot[:, 30].std())
    sbc = [float((samp[:, :, p] < theta[:, p, None]).mean(1).std()) for p in range(3)]
    return net, se, sbc


def verdict(net, sbc, fom3):
    if net < -0.02 or max(sbc) > 0.305:
        return "REJECT (over-confident)"
    cal = (-0.02 <= net <= 0.02) and all(0.282 <= s <= 0.296 for s in sbc)
    if not cal:
        return "not-calibrated (net or SBC out)"
    if fom3 and fom3 >= JOINTL1:
        return "CALIBRATED + FoM3>=jointL1  *** clean win ***"
    return "CALIBRATED (tie band)"


def main():
    rows = []
    for d in sorted(OUT.glob("L*_H*")):
        m = re.match(r"L(\d+)_H(\d+)", d.name)
        arm = f"realnvp_L{m.group(1)}_H{m.group(2)}"
        samp, theta = pooled_unstrat(d / "gate" / "dumps", arm)
        if samp is None:
            print(f"{d.name}: dumps missing (still running?)"); continue
        net, se, sbc = calib(samp, theta)
        fj = d / "screen" / "median_summary.json"
        fom3 = float(json.load(open(fj))["fom3"]) if fj.exists() else None
        rows.append((d.name, fom3, net, se, sbc, verdict(net, sbc, fom3)))
    print(f"\n{'config':10s} {'FoM3':>6s} {'TARPnet':>9s}  {'SBC(Om/s8/w0)':>20s}  verdict")
    print(f"{'4x128(base)':10s} {BASE_FOM3:6.0f} {'+0.033':>9s}  {'0.290/0.289/0.282':>20s}  (current, conservative)")
    for name, fom3, net, se, sbc, v in rows:
        fs = f"{fom3:6.0f}" if fom3 else "   NA "
        print(f"{name:10s} {fs} {net:+8.4f}±{se:.3f}  {sbc[0]:.3f}/{sbc[1]:.3f}/{sbc[2]:.3f}  {v}")


if __name__ == "__main__":
    main()
