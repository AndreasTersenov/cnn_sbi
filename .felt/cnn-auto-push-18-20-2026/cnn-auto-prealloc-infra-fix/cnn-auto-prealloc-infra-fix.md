---
name: PREALLOCATE=false unlocks 5x parallelism on titan A100s
status: closed
tags:
    - cnn-auto-push
created-at: 2026-05-18T19:47:43.572311024Z
closed-at: 2026-05-18T19:47:43.580003271Z
outcome: Inner script doesn't set XLA_PYTHON_CLIENT_PREALLOCATE -> JAX preallocates xla_mem_fraction × total_GPU upfront. Real compressor + flow uses ~3 GB (cbs=128) but was reserving 36 GB. Forcing PREALLOCATE=false in subprocess env + xla=0.3 cap enabled 5 parallel jobs on one A100. Chore commit 73ba6a4.
---
