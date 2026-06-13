# Packing benchmark (B1 spec, 3-pack only; workload = B3_nobnt_unicut screen)
- solo reps (s): ['1388', '1319', '1262'] (median 1319)
- 3-pack wall (s): 2757 (all rc=0: True)
- aggregate throughput ratio = (3/2757) / (3/1319) = 0.48 (accept >= 0.9 per job-rate... derived as 3-pack aggregate vs 3x solo)
- **3-pack REJECTED** for sweep phases tonight
- B2/B3 (compressor packing / cross-class) DEFERRED: no compressor workload in this campaign.
- load1 at bench: 39.6; foreign MB on GPU1: 6431
