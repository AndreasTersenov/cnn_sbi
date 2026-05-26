---
name: Extend harmonic-cache code to support train[:70%] slicing + disjointness audit
tags:
    - code-change
    - canonical-anchors-refresh-2026-05
    - ship-blocker
    - methodology-fix
created-at: 2026-05-24T20:35:52.17081715Z
outcome: 'Required for the canonical-anchors refresh to apply uniform 70/30 split discipline across all 4 arms. Current limitation: scripts/sbi/npe_cnn_nbody_tomo.py _normalize_harmonic_split (line 951) rejects slicing syntax like ''train[:70%]'', and main() explicitly rejects --require-disjoint-train-examples for harmonic-cache route (line 3249). Changes needed: (1) _normalize_harmonic_split parses ''train[:70%]'' / ''train[70%:]'' returning (base_name, slice_low, slice_high); (2) _list_harmonic_cache_files accepts and applies the slice deterministically on the sorted file list; (3) parallel audit_harmonic_split_overlap function checks file-set disjointness; (4) remove the rejection at line 3249. Smoke test: --no-train --no-sample on the cache, verify file counts add up + zero overlap. Est ~1h coding + ~10min smoke.'
---
