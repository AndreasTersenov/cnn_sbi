# PLAN — Tier-1 GPU-packing benchmarks (scoping; awaiting go/no-go)

**Date:** 2026-06-11. **Status:** DECIDED (Andreas, 2026-06-11): Q-1 = DEFER — run as the
first phase of the next GPU campaign (joint-PDF pillar would be the first consumer), zero GPU
spend today; Q-2 = land the FOOTGUN FIXES now — VERIFIED 2026-06-11: all three are ALREADY fixed (the
06-10 audit-fixes pass): repr-corners defaults `--gpus 1,2` (GPU 3 excluded), and both
multiseed + fidsumm drivers carry SKIP-on-cmd-build-failure. Nothing to land; scheduler/
preamble code waits for the measured table; Q-3 = 3-pack only (drop the 4-pack condition).
Source spec: EFFICIENCY_AUDIT_2026-06-10.md §Measurement plans.
**Objective:** replace the assumed per-phase packing table
`{compressor: 1→2?, fidsumm: 2, sweep: 3→4?, tarp: 2, lc2st: 2}` with MEASURED defaults,
so future campaign drivers (incl. a possible joint-PDF third pillar) pack by evidence.
**Decision metric (per benchmark, from the audit):** sweep — accept largest N with aggregate
throughput ≥ 0.9 × N × solo; compressor 2/GPU — accept if EACH job ≥ 0.85× solo it/s;
cross-class — accept if compressor ≥ 0.9× solo it/s while a sweep co-resides.

## Design (all three benchmarks; `feedback_benchmark_dont_assume` rules)

Common controls: same day, same GPU; ≥3 reps per condition; record `load1`, co-tenant
memory (nvidia-smi probe before/after each rep), and cache state; tenant-checked GPU from
the 0/1/2 pool, abort the rep if a foreign tenant arrives mid-run; thread caps SET
EXPLICITLY in each launched env (`OMP/MKL/OPENBLAS/NUMEXPR_NUM_THREADS=8`,
`TF_NUM_INTRAOP_THREADS=8`, `TF_NUM_INTEROP_THREADS=2` — don't rely on login-shell exports).
Packing mechanics: Tier-0 (duplicated GPU ids + `XLA_PYTHON_CLIENT_MEM_FRACTION` per job) —
no code changes needed to benchmark.

**B1 — sweep packing (3/GPU, then 4/GPU).** Workload = one real
`population_sweep_flatsky.py` arm re-run to a `dryruns/` output (jit path, NDE-train-
dominated, ~30 min/arm measured solo). Conditions: solo ×3 reps; 3-pack (mem frac 0.30)
×3; 4-pack (frac 0.22) ×3 — identical arm, distinct seeds/output dirs to avoid cache
collisions. Measure wall time per job + aggregate. GPU budget: roughly a half-day on one
GPU (solo reps serial) — I will not quote a tighter number I haven't measured.

**B2 — compressor packing (2/GPU).** Workload = short compressor run (flat-local CNN,
~5k steps, `--exit-after-compress`-style truncation; exact flags verified at build).
Metric = steady-state it/s after warmup (read from the step log). Conditions: solo ×3;
2-pack (frac 0.45) ×3.

**B3 — cross-class co-residency.** One compressor (as B2) + one sweep arm co-resident;
measure the compressor's it/s degradation vs its B1-day solo baseline.

Output: `packing_benchmarks/PACKING_RESULT.md` with the derived (not asserted) accept/
reject per condition + the resulting packing table; then the Tier-1 code diffs (multi-slot
scheduler + tenant probe + env preamble + the two footgun fixes, audit items 1–4) land in a
separate commit with the measured table as defaults.

## Open questions for Andreas

- **Q-1 (worth it now?):** sweeps are jit-fast (~30 min/arm) and no new campaign is queued;
  the payoff is conditional on future campaigns (joint-PDF pillar would be the first
  consumer). Run the benchmarks opportunistically post-whiten, or defer until a concrete
  campaign exists?
- **Q-2 (scope):** benchmarks only, or also land the Tier-1 code (scheduler/probe/preamble/
  footguns) immediately after, with the measured defaults?
- **Q-3:** B1's 4-pack condition doubles B1's wall budget for a secondary data point —
  include or drop to 3-pack only?
