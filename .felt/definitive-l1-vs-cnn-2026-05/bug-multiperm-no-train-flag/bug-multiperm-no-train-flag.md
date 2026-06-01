---
name: Multi-perm obs recompress used the wrong flag
status: closed
tags:
    - bug
    - definitive
created-at: 2026-06-01T02:47:18.701652757Z
outcome: 'FIXED (2026-05-31): used --no-train (that is for the FLOW, not the compressor) + --save-dir, so it fell back to default tomo/save_params/...batch150000.pkl -> FileNotFound, and the gate self-skipped multi-perm. Fix: OMIT --train-compressor and pass --compressor-params <best_val.pkl> --compressor-state <...>. Multi-perm then ran clean (18/18 posteriors, 2 arms x 3 seeds x 3 perms). Artifacts: run_multiperm_fixed.sh; HANDOFF section 5. Next: per-perm-average the multi-perm arms in aggregate_all_arms.py.'
---

The gated design meant the wrong flag cost no compute (self-skip). Part of [[definitive-l1-vs-cnn-2026-05]]; see also [[refine-phase-c-perm-matched]].
