---
name: Switch autoresearch keep-rule from mean-of-seeds to pooled FoM3?
status: open
tags:
    - question
    - cnn-auto-push-18-20-2026
created-at: 2026-05-18T22:34:42.667533495Z
outcome: 'Mode-drift audit (A4) shows CNN MoS gain since iter-0 is +30% (14295->18568) but pooled gain is only +10% (11700->12894). pooled/mos = 0.69 vs L1 ref 0.89. Question: change the autoresearch keep-rule to use pooled FoM3 (primary) with MoS as a secondary report? Risk: pooled is dominated by per-seed centroid drift which may settle as compressor trains longer; switching too early may discard improvements that DO compound in pooled at 240k. Proposed compromise: report BOTH; require pooled to clear half-noise OR keep showing MoS gain >0.5*std AND pooled/mos ratio not degrading by >5%.'
---
