# Project experiments: very short summary

- We tested SBI contours from `CNN`, `L1 (jaxili, no PCA)`, and `L1+VMIM` across many noBNT/BNT/tomographic setups.
- In the **final noBNT benchmark**, CNN is best overall, and its full-tomography gain is strongly driven by cross-bin correlations (large extra gain from single-bin to tomo4).
- For **L1**, moving to jaxili + no PCA fixed earlier over-broad/unstable behavior; this is the current trusted L1 baseline.
- For **L1+VMIM**, optimization improved results a lot; best calibrated run is near-lossless vs raw L1, while some tighter runs were biased.
- In **BNT tests**, all methods inflate at first; optimized CNN became near-lossless, while L1/L1+VMIM stayed more BNT-sensitive.
- In **baryon-mismatch tests** (baryonified observation, no-bary training), all methods show bias; L1-jaxili had the best truth-proximity on average, CNN had the strongest coherent \(\Omega_m,w_0\) shifts.
- Main practical status: use optimized CNN for strongest constraints; keep L1-jaxili as robust cross-check; treat very tight legacy L1 runs with caution unless calibration is verified.

