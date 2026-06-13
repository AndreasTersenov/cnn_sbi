#!/usr/bin/env python
"""Apples-to-apples corner: A1 (VMIM joint PDF) at the SAME fiducial obs as the canonical
representative_corner_typical figure (perm16/patch23 'typical' + perm0/patch90 'favorable'),
overlaid with the trusted saved l1+product and l1+both fiducial samples.

This replaces the misleading single-val-point corner: the val point (truth sigma8=0.79) was
an off-center realization where BOTH arms are ~1.7x wider than their medians. The fiducial
obs is the canonical comparison point.

Serializes after run_followups (no GPU race), then waits for a free GPU, samples A1 (3 seeds,
compressed cache, preproc none), and overlays. CPU-only after sampling.
Detached:  (cd scripts/sbi && setsid nohup <py> make_a1_fiducial_corner.py > .../a1_fid_corner.out 2>&1 &)
"""
import subprocess
import time
from pathlib import Path
import numpy as np

SBI = "/mnt/home/tersenov/software/cnn_sbi/scripts/sbi"
PY = "/home/tersenov/anaconda3/envs/jaxili/bin/python"
FC = f"{SBI}/results/exploratory/flatsky_cross_2026_06"
OUT = f"{FC}/overnight_menu_2/lane_a_plots"
A1 = f"{FC}/overnight_menu_2/A1_pair2d_vmim"
REP = f"{FC}/representative_corner"
PN = ["Om", "s8", "w0"]
TRUTH = {"typical": None, "favorable": None}  # fiducial cosmology marked from fz truth
FREE_MEM_MB, FREE_UTIL = 2000, 15


def wait_followups_done():
    while subprocess.run(["pgrep", "-f", "[r]un_followups"],
                         capture_output=True).returncode == 0:
        print("waiting for run_followups to finish...", flush=True)
        time.sleep(180)


def wait_free_gpu():
    while True:
        for g in (1, 0, 2):
            try:
                o = subprocess.run(["nvidia-smi", "--query-gpu=memory.used,utilization.gpu",
                                    "--format=csv,noheader,nounits", "-i", str(g)],
                                   capture_output=True, text=True, timeout=20
                                   ).stdout.strip().split(",")
                m, u = int(o[0]), int(o[1])
            except Exception:
                m, u = 1 << 30, 100
            if m < FREE_MEM_MB and u < FREE_UTIL:
                time.sleep(20)
                return g
        time.sleep(180)


def main():
    Path(OUT).mkdir(parents=True, exist_ok=True)
    wait_followups_done()
    gpu = wait_free_gpu()
    print(f"sampling A1 at fiducial obs on GPU{gpu}", flush=True)
    # sample A1 at the canonical fiducial obs (preproc none — compressed features)
    rc = subprocess.run(
        [PY, "representative_corner_flatsky.py",
         "--train-cache-dir", f"{A1}/cache", "--arm-label", "A1_vmim",
         "--fiducial-summaries-npz", f"{A1}/fiducial_summaries.npz",
         "--output-dir", f"{REP}/A1_vmim",
         "--preproc-transform", "none", "--clip-value", "0",
         "--min-feature-variance", "1e-12",
         "--seeds", "41,42,43", "--cuda-visible-devices", str(gpu)],
        cwd=SBI, env=dict(__import__("os").environ, CUDA_VISIBLE_DEVICES=str(gpu),
                          XLA_PYTHON_CLIENT_PREALLOCATE="false",
                          XLA_PYTHON_CLIENT_MEM_FRACTION="0.85")).returncode
    if rc != 0:
        print("A1 fiducial sampling FAILED", flush=True); return 1

    # overlay: A1 vs the trusted saved l1+product (+ both) at each fiducial obs
    from getdist import MCSamples, plots as gdplt
    a1 = np.load(f"{REP}/A1_vmim/corner_samples.npz")
    prod = np.load(f"{REP}/flat_product/corner_samples.npz")
    both = np.load(f"{REP}/flat_both/corner_samples.npz")
    fz = np.load(f"{A1}/fiducial_summaries.npz")
    truth = fz["truth"][:3] if "truth" in fz.files else np.array([0.26, 0.84, -1.0])
    for tag in ("typical", "favorable"):
        arms = [("A1 VMIM joint PDF", a1[tag], "tab:red"),
                ("l1 + product (prev best)", prod[tag], "tab:purple"),
                ("l1 + both", both[tag], "tab:green")]
        mcs = [MCSamples(samples=s[:, :3], names=PN,
                         labels=[r"\Omega_m", r"\sigma_8", "w_0"], label=lab)
               for lab, s, _ in arms]
        g = gdplt.get_subplot_plotter(width_inch=7.0)
        g.settings.legend_fontsize = 9
        g.triangle_plot(mcs, PN, filled=True, colors=[c for *_, c in arms],
                        legend_labels=[lab for lab, *_ in arms])
        for ai, t in enumerate(truth):
            for ax in g.subplots[:, ai]:
                if ax is not None:
                    ax.axvline(t, color="k", ls=":", lw=0.8)
        for ext in ("png", "pdf"):
            g.export(f"{OUT}/fiducial_corner_a1_vs_product_{tag}.{ext}")
        # widths table
        print(f"[{tag}] std(Om,s8,w0): "
              + " | ".join(f"{lab}={np.round(s[:, :3].std(0), 3)}" for lab, s, _ in arms),
              flush=True)
    print("A1 FIDUCIAL CORNER DONE", flush=True)


if __name__ == "__main__":
    main()
