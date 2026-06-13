#!/usr/bin/env python
"""Aggressive parallel follow-up scheduler (Andreas: 'use as much free GPU as there is, no
need to be polite'). Supersedes the polite run_followups.py / make_a1_fiducial_corner.py.

3 workers bound to GPUs 0,1,2 (NEVER 3); no free-GPU wait, co-resides with foreign tenants.
One job per GPU at a time (predictable memory). Priority queue (FIFO):
  vmim_s42, vmim_s43   -> build+sweep+gate  (the load-bearing 3-seed band; mem 0.5)
  fid_corner           -> A1 (seed41) sampled at the canonical fiducial obs (mem 0.5)
  k15_C2, k15_C3       -> full sweep (+C2 gate); mem 0.75 (the big arms)
Then finalize: VMIM_ROBUSTNESS.md + fiducial A1-vs-product overlay + regen result doc.
Detached: (cd scripts/sbi && setsid nohup <py> run_followups2.py > .../followups2.out 2>&1 &)
"""
import queue
import subprocess
import threading
import time
from pathlib import Path
import numpy as np

import run_followups as rf  # reuse run(), sweep(), gate(), vmim_build(), med(), finalize(), paths

PY, SBI, OM2, GC2, FC = rf.PY, rf.SBI, rf.OM2, rf.GC2, rf.FC
REP = f"{FC}/representative_corner"
A1 = f"{OM2}/A1_pair2d_vmim"
GPUS = [0, 1, 2]


def vmim_pipeline(seed, gpu, t0):
    out = f"{OM2}/A1_vmim_s{seed}"
    if not rf.run(f"vmim_build_s{seed}", rf.vmim_build(seed, gpu), gpu, t0, mem=0.5):
        rf.status(f"- [f2] VMIM s{seed} BUILD FAIL"); return
    rf.run(f"vmim_sweep_s{seed}", rf.sweep(f"{out}/cache", f"{out}/fiducial_summaries.npz",
           f"{out}/population_sweep_full", gpu, t0, preproc=("none", "0", "1e-12")),
           gpu, t0, mem=0.5)
    m = rf.med(f"{out}/population_sweep_full")
    rf.status(f"- [f2] VMIM s{seed} {'FoM3 %.0f' % m['fom3'] if m else 'SWEEP FAIL'}")
    rf.run(f"vmim_gate_s{seed}", rf.gate(f"{out}/cache", f"A1_vmim_s{seed}", gpu,
           preproc=("none", "0", "1e-12")), gpu, t0, mem=0.5)


def k15_pipeline(name, gpu, t0, do_gate):
    ok = rf.run(f"k15_{name}", rf.sweep(f"{OM2}/{name}/cache",
                f"{OM2}/{name}/fiducial_summaries.npz",
                f"{OM2}/{name}/population_sweep_full", gpu, t0), gpu, t0, mem=0.75)
    m = rf.med(f"{OM2}/{name}/population_sweep_full")
    rf.status(f"- [f2] {name} {'FoM3 %.0f' % m['fom3'] if m else 'FAIL'}")
    if ok and do_gate:
        rf.run(f"k15_gate_{name}", rf.gate(f"{OM2}/{name}/cache", name, gpu), gpu, t0, mem=0.75)


def fid_corner_sample(gpu, t0):
    """Sample A1 (seed41 compressor) at the canonical fiducial obs (preproc none)."""
    rf.run("fid_sample_A1", [PY, "representative_corner_flatsky.py",
           "--train-cache-dir", f"{A1}/cache", "--arm-label", "A1_vmim",
           "--fiducial-summaries-npz", f"{A1}/fiducial_summaries.npz",
           "--output-dir", f"{REP}/A1_vmim",
           "--preproc-transform", "none", "--clip-value", "0",
           "--min-feature-variance", "1e-12", "--seeds", "41,42,43",
           "--cuda-visible-devices", str(gpu)], gpu, t0, mem=0.5)


def worker(gpu, q, t0):
    while True:
        try:
            job = q.get_nowait()
        except queue.Empty:
            return
        try:
            job(gpu, t0)
        except Exception as exc:
            rf.status(f"- [f2] job on GPU{gpu} raised: {exc}")


def fiducial_overlay():
    """CPU: overlay A1 fiducial samples with saved l1+product / l1+both at typical+favorable."""
    try:
        from getdist import MCSamples, plots as gdplt
        a1 = np.load(f"{REP}/A1_vmim/corner_samples.npz")
        prod = np.load(f"{REP}/flat_product/corner_samples.npz")
        both = np.load(f"{REP}/flat_both/corner_samples.npz")
        fz = np.load(f"{A1}/fiducial_summaries.npz")
        truth = fz["truth"][:3] if "truth" in fz.files else np.array([0.26, 0.84, -1.0])
        PN = ["Om", "s8", "w0"]
        OUTp = f"{OM2}/lane_a_plots"
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
            for ai, tv in enumerate(truth):
                for ax in g.subplots[:, ai]:
                    if ax is not None:
                        ax.axvline(tv, color="k", ls=":", lw=0.8)
            for ext in ("png", "pdf"):
                g.export(f"{OUTp}/fiducial_corner_a1_vs_product_{tag}.{ext}")
            rf.status(f"- [f2] fiducial corner [{tag}] std: "
                      + " ".join(f"{lab.split()[0]}={np.round(s[:, :3].std(0), 3)}"
                                 for lab, s, _ in arms))
        print("fiducial overlay written", flush=True)
    except Exception as exc:
        rf.status(f"- [f2] fiducial overlay FAILED: {exc}")


def main():
    t0 = time.time()
    rf.status(f"## followups2 (aggressive) started {time.strftime('%F %T')} GPUs {GPUS}")
    q = queue.Queue()
    for seed in (42, 43):
        q.put(lambda g, t, s=seed: vmim_pipeline(s, g, t))
    q.put(fid_corner_sample)
    q.put(lambda g, t: k15_pipeline("C2_pair2d_k15", g, t, do_gate=True))
    q.put(lambda g, t: k15_pipeline("C3_pair2d_k15_bnt_ar", g, t, do_gate=False))
    threads = [threading.Thread(target=worker, args=(g, q, t0), daemon=True) for g in GPUS]
    [t.start() for t in threads]
    [t.join() for t in threads]

    # coverage recompute (gates added A1_vmim_s42/43 + C2 dumps) + finalize
    rf.run("f2_coverage", [PY, "run_tarp_coverage.py", "--dumps-root",
           f"{GC2}/tarp_drp/dumps", "--outdir", f"{GC2}/tarp_drp", "--dims", "3"],
           GPUS[0], t0, mem=0.4)
    rf.finalize()
    fiducial_overlay()
    subprocess.run([PY, "run_overnight_menu_2.py", "--regen-only"], cwd=SBI)
    rf.status(f"## followups2 complete {time.strftime('%F %T')} ({(time.time()-t0)/3600:.1f} h)")
    print("ALL FOLLOWUPS DONE", flush=True)


if __name__ == "__main__":
    main()
