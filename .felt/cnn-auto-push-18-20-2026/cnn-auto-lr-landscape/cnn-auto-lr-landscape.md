---
name: 'Compressor LR: peak at 1e-3, possible ''bump'' at 3e-3 (noise?)'
status: closed
tags:
    - cnn-auto-push
created-at: 2026-05-18T19:47:43.759712068Z
closed-at: 2026-05-18T19:47:43.769525441Z
outcome: 'Sweep on cdim=16: 5e-4=16149, 1e-3=18568, 1.5e-3=17415, 2e-3=17871, 3e-3=18485, 4e-3=16933. Peak at 1e-3; 3e-3 ties on mean but better per_seed_min and tighter std. Could be real bimodality or 3-seed noise. To disentangle: replicate 1e-3 and 3e-3 with 5 seeds each. Currently lr=1e-3 wins by strict-mean rule.'
---
