---
name: 3-seed FoM3 mean is outlier-sensitive
status: closed
tags:
    - cnn-auto-push
created-at: 2026-05-18T19:47:43.426263507Z
closed-at: 2026-05-18T19:47:43.450899644Z
outcome: iter-1 mean=16149 was inflated by seed 41 (18307) vs other two ~15k. Differences of ~1-2k between configs can be one seed swinging. iter-7 (lr=3e-3) had mean tied with iter-5 (lr=1e-3) but tighter std — possibly the same underlying truth. Replication / 5-seed runs needed for fine-grained ranking.
---
