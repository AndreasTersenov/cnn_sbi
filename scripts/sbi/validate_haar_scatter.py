#!/usr/bin/env python
"""Validate the 2D-1D Haar SCATTERING transform BEFORE any expensive run.

Checks (cheap, ~few hundred examples):
 A. Mechanical correctness of modulus_haar_fields:
    A1 einsum == explicit Σ_b H[m,b]|W_b| (exact);
    A2 deep mode J_deep = ½Σ|W_b| is >= 0 EVERYWHERE (all-positive Haar row × non-neg moduli);
    A3 difference modes are ~zero-mean (mixed-sign rows);
    A4 shape / NaN / finite of the full scatter ℓ1 datavector.
 B. Scientific sense — does the statistic carry cosmological information?
    For ~N train examples (varied cosmology), compute per-(mode,scale) summaries of J and
    correlate with σ8 and Ω_m. Report no-BNT AND BNT-space. If the BNT-space modes show
    σ8/Ωm response comparable to no-BNT, goal 2 is promising; if flat, goal 2 will fail and
    we should NOT run the full sweep. Compare to the LINEAR Haar's per-channel response.
Run on a small sample; prints a verdict + writes a sensitivity figure.
"""
import os, sys
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import flatsky_cross as fx
import flatsky_haar_scatter as hs

TFDS = "nbody_cosmogrid_dataset_tomo_cross/grid_10deg_80px_nonoverlap180"
DDIR = "/home/tersenov/tensorflow_datasets"
OUT = HERE + "/results/exploratory/flatsky_cross_2026_06/plots_2d1d"
NS = 5
MODES = ["deep", "coarse", "d12", "d34"]


def load_examples(n_target=600, perm_lo=5, perm_hi=6):
    from tfds_cross_tfdata_loader import iter_cross_tfds_batches
    A, T = [], []
    n = 0
    for autos, theta in iter_cross_tfds_batches(TFDS, DDIR, "train", 256, flip=False,
                                                channel_scale=None, channel_slice=slice(0, 4),
                                                perm_lo=perm_lo, perm_hi=perm_hi, seed=0):
        if np.isnan(autos).any():
            continue
        A.append(autos); T.append(theta); n += autos.shape[0]
        if n >= n_target:
            break
    return np.concatenate(A)[:n_target], np.concatenate(T)[:n_target]


def main():
    import torch
    from wl_stats_torch import WLStatistics
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    stats = WLStatistics(n_scales=NS, device=dev, pixel_arcmin=7.5, dtype=torch.float64)
    H = torch.from_numpy(hs.haar4()).to(dev, torch.float64)
    print(f"device={dev}\n===== A. MECHANICAL =====")

    autos, theta = load_examples(600)
    print(f"loaded {autos.shape[0]} train examples (varied cosmology)")
    a32 = autos[:64]
    at = torch.from_numpy(np.ascontiguousarray(a32, np.float64)).to(dev)

    # A1: einsum vs explicit, on the modulus fields
    ch = fx.build_channels_torch(at, "none", bnt=False)            # (B,H,W,4) = autos
    aWc = []
    for b in range(4):
        stats.compute_wavelet_transform(ch[..., b].contiguous(), 1.0, subtract_coarse_mean=True)
        aWc.append(stats.wavelet_coeffs.abs())
    aWc = torch.stack(aWc, 0)                                       # (4,B,ns,H,W)
    J = hs.modulus_haar_fields(ch, H, stats)                       # (4,B,ns,H,W)
    Hnp = hs.haar4()
    explicit = sum(Hnp[0, b] * aWc[b] for b in range(4))          # deep, explicit
    err = (J[0] - explicit).abs().max().item()
    print(f"A1 einsum vs explicit (deep): max|Δ| = {err:.2e}  {'OK' if err < 1e-9 else 'FAIL'}")

    # A2: deep mode >= 0 everywhere
    jmin = J[0].min().item()
    print(f"A2 deep mode min = {jmin:.3e}  {'OK (>=0)' if jmin >= -1e-12 else 'FAIL (<0!)'}")

    # A3: difference modes ~zero-mean (per scale)
    for m in (2, 3):
        means = [J[m, :, s].mean().item() for s in range(NS)]
        std0 = J[m, :, 0].std().item()
        ok = abs(means[0]) < 0.1 * std0
        print(f"A3 mode {MODES[m]} per-scale means {[f'{x:.2e}' for x in means]} "
              f"(scale0 std {std0:.2e}) {'OK ~0' if ok else 'CHECK'}")

    # A4: full datavector shape/NaN with a placeholder sigma + percentile ranges
    sig_pl = torch.tensor([[float(aWc[b].std()) for _ in range(NS)] for b in range(4)],
                          device=dev, dtype=torch.float64)         # crude per-mode placeholder
    # placeholder per-mode sigma = std of J_m
    sig_modes = torch.stack([J[m].std() * torch.ones(NS, device=dev, dtype=torch.float64)
                             for m in range(4)])
    rng = np.array([[float((J[m] / sig_modes[m].view(1, NS, 1, 1)).quantile(0.005)),
                     float((J[m] / sig_modes[m].view(1, NS, 1, 1)).quantile(0.995))] for m in range(4)])
    x = hs.scatter_l1(ch, H, sig_modes, stats, 40, rng, clamp_overflow=True)
    print(f"A4 datavector {x.shape} (expect (64, 800)) NaN={np.isnan(x).any()} "
          f"finite={np.isfinite(x).mean():.4f}  {'OK' if x.shape==(64,800) and not np.isnan(x).any() else 'FAIL'}")

    # ===== B. SCIENTIFIC SENSITIVITY =====
    print("\n===== B. SENSITIVITY TO COSMOLOGY (mean J per mode/scale vs σ8, Ωm) =====")
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    s8 = theta[:, 1]; om = theta[:, 0]

    def feats(pre):
        """mean |J_m| per (mode,scale) for each example -> (N, 4, NS)."""
        out = []
        for i0 in range(0, autos.shape[0], 128):
            ab = torch.from_numpy(np.ascontiguousarray(autos[i0:i0+128], np.float64)).to(dev)
            chb = fx.build_channels_torch(ab, "none", bnt=pre)
            Jb = hs.modulus_haar_fields(chb, H, stats)               # (4,B,ns,H,W)
            out.append(Jb.abs().mean(dim=(3, 4)).permute(1, 0, 2).cpu().numpy())  # (B,4,ns)
        return np.concatenate(out, 0)

    fig, axes = plt.subplots(2, 4, figsize=(16, 7), squeeze=False)
    verdict = {}
    for col, (pre, lab) in enumerate([(False, "no-BNT"), ("bnt", "BNT space")]):
        F = feats(pre)                                              # (N,4,NS)
        for m in range(4):
            ax = axes[col, m]
            cs8, com = [], []
            for s in range(NS):
                v = F[:, m, s]
                c8 = np.corrcoef(v, s8)[0, 1]; cm = np.corrcoef(v, om)[0, 1]
                cs8.append(c8); com.append(cm)
            ax.plot(range(NS), cs8, "o-", label=r"corr $\sigma_8$")
            ax.plot(range(NS), com, "s--", label=r"corr $\Omega_m$")
            ax.axhline(0, color="k", lw=0.5); ax.set_ylim(-1, 1)
            ax.set_title(f"{lab}: {MODES[m]}", fontsize=10); ax.set_xlabel("scale")
            if m == 0:
                ax.set_ylabel("corr(mean J, param)"); ax.legend(fontsize=8)
            verdict[(lab, MODES[m])] = (max(abs(np.array(cs8))), max(abs(np.array(com))))
    fig.suptitle("Phase-2 sensitivity check: does mean modulus-Haar field respond to cosmology? "
                 "(|corr| near 0 at all scales = no information)", fontsize=11)
    os.makedirs(OUT, exist_ok=True)
    p = f"{OUT}/validate_scatter_sensitivity.png"; fig.savefig(p, dpi=120, bbox_inches="tight")
    print(f"wrote {p}")
    print("\n  max |corr| per mode (σ8 / Ωm):")
    for (lab, mode), (c8, cm) in verdict.items():
        print(f"    {lab:10s} {mode:6s}: σ8 {c8:.3f}  Ωm {cm:.3f}")
    nb = max(c8 for (lab, _), (c8, _) in verdict.items() if lab == "no-BNT")
    bn = max(c8 for (lab, _), (c8, _) in verdict.items() if lab == "BNT space")
    print(f"\n  VERDICT: max σ8-corr  no-BNT={nb:.3f}  BNT={bn:.3f}")
    print("  (B carries info if these are clearly > the ~0.1-0.2 you'd get from 600-sample noise;"
          " BNT comparable to no-BNT => goal-2 promising; BNT≈0 => goal-2 will fail.)")


if __name__ == "__main__":
    main()
