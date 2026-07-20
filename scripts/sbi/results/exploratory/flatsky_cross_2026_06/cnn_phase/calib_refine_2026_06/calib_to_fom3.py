#!/usr/bin/env python3
"""How much FoM3 does the CNN's mild over-coverage cost? Sharpen the posterior isotropically about its
mean until the un-stratified TARP net hits 0, and read off the implied FoM3 (FoM3 ∝ width^-3).

For a scale s (<1 = sharpen): samp_s = mu + s*(samp - mu). We sweep s, recompute un-stratified TARP
net (averaged over reference draws) and SBC rank-std, find s* where net->0, and report the implied
corrected FoM3 = current * s*^-3. CAVEAT printed: isotropic sharpening also narrows the (already
~calibrated) marginals, so SBC goes over-confident at s* -> this is an UPPER estimate of the honest,
covariance-aware correction. CPU only.
"""
import glob
import numpy as np
import tarp

G = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi/results/exploratory/flatsky_cross_2026_06/cnn_phase/arch_sweep_2026_06_13/gate_c_resnet18/dumps_all/resnet18_all"
ALPHA = np.linspace(0.0, 1.0, 61)
NREF = 10
FOM3_S41, FOM3_MEAN, JOINTL1 = 3326.0, 3304.0, 3371.0


def net_and_sbc(samp, theta):
    s = np.transpose(samp, (1, 0, 2))
    nets = []
    for _ in range(NREF):
        ecp, al = tarp.get_tarp_coverage(s, theta, references="random", norm=True)
        ecp = np.asarray(ecp); ecp = ecp.mean(0) if ecp.ndim == 2 else ecp
        nets.append(np.trapz(np.interp(ALPHA, np.asarray(al), ecp) - ALPHA, ALPHA) * 2)
    ranks = np.stack([(samp[:, :, p] < theta[:, p, None]).mean(1) for p in range(3)], 1)
    return float(np.mean(nets)), [float(ranks[:, p].std()) for p in range(3)]


def main():
    dd = sorted(glob.glob(f"{G}/seed_*/n*_m*/posterior_samples.npz"))
    arrs = [np.load(d) for d in dd]
    theta = arrs[0]["theta"][:, :3].astype(np.float32)
    samp = np.concatenate([a["samples"][:, :, :3] for a in arrs], axis=1).astype(np.float32)
    mu = samp.mean(axis=1, keepdims=True)
    grid = [0.94, 0.95, 0.96, 0.97, 0.98, 0.99, 1.00]
    rows = []
    for s in grid:
        net, sbc = net_and_sbc(mu + s * (samp - mu), theta)
        rows.append((s, net, sbc))
        print(f"s={s:.2f} | TARP net {net:+.4f} | SBC {sbc[0]:.3f}/{sbc[1]:.3f}/{sbc[2]:.3f} | "
              f"FoM3(mean) {FOM3_MEAN * s**-3:.0f}", flush=True)
    ss = np.array([r[0] for r in rows]); nn = np.array([r[1] for r in rows])
    sstar = float(np.interp(0.0, nn[::-1], ss[::-1]))            # s where net crosses 0 (net decreasing in s)
    gain = sstar ** -3
    print(f"\ns* (net->0) = {sstar:.4f}  => FoM3 gain s*^-3 = {gain:.4f}")
    print(f"corrected CNN FoM3:  s41 {FOM3_S41 * gain:.0f}  | 3-seed mean {FOM3_MEAN * gain:.0f}   "
          f"(joint L1 = {JOINTL1:.0f})")
    print("CAVEAT: isotropic sharpen also narrows the already-calibrated marginals (see SBC rising "
          ">0.289 near s*), so this is an UPPER estimate of the honest covariance-aware correction.")


if __name__ == "__main__":
    main()
