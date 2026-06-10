# Recipe-level check — 160k steps + val-batches 16 vs the 80k baseline

Pooled 3-MAF 9000-obs median FoM3, paired per (arm, compressor seed).

| arm/seed | 80k | 160k | 160k/80k | CNN/L1 product (160k) |
|---|---|---|---|---|
| none_s42 | 2170 | 2334 | 1.08× | — |
| none_s43 | 2480 | 2402 | 0.97× | — |
| product_s42 | 2393 | 2389 | 1.00× | 0.83× |
| product_s43 | 2433 | 2451 | 1.01× | 0.85× |

**Observed:** mean 160k/80k lift — auto 1.02×, product 1.00×; best CNN/L1(product) at 160k = 0.85× (80k range was 0.83–0.85× for these seeds). No product-specific recipe gain (product lift ≈/≤ auto lift). CNN still well below L1 product.


NB the 160k recipe bundles TWO changes vs 80k: 2× steps AND de-noised best_val (val-batches 16 vs 1). If it moves, ablate before attributing.