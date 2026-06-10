# CNN best single (MAF) seed — un-pooled — at the typical obs (perm16/patch23)

The CNN has ONE compressor (seed 41); the 3 pooled seeds are MAF/NDE seeds. Pooling haircuts the FoM3, so the best single seed is the CNN at its most favorable. Reloaded the trained MAF checkpoints (10-d reloads bit-exact). L1 pooled shown for reference (L1's 2000-d datavector can't be reloaded per-seed without a retrain).

| arm | seed41 | seed42 | seed43 | **best** | (CNN pooled-median) | (L1 pooled-median) |
|---|---|---|---|---|---|---|
| auto-only | 2620 | 2364 | 2387 | **2620** (s41) | 2325 | 2405 |
| +conv | 2418 | 1968 | 2491 | **2491** (s43) | 2192 | 2499 |
| +product | 2225 | 2331 | 2017 | **2331** (s42) | 2181 | 2875 |
| +both | 2475 | 2436 | 2205 | **2475** (s41) | 2306 | 2910 |

**Best-seed vs-auto ratios (does the no-cross-gain survive un-pooled?):**
auto-only: 1.00× | +conv: 0.95× | +product: 0.89× | +both: 0.94×
