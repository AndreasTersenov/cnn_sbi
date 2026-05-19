---
name: 'Compressor checkpoint policy: best_val (default) vs last_step'
tags:
    - finding
    - cnn-auto-push-18-20-2026
    - bug-fix
created-at: 2026-05-19T08:55:02.454717682Z
outcome: Fixed [[cnn-auto-compressor-returns-last-step]] (2026-05-19, branch autoresearch/cnn-auto-push-18-20-2026). Added --compressor-checkpoint-policy {best_val,last_step} (default best_val) to npe_cnn_nbody_tomo.py. train_compressor_vmim now tracks (params,state,step,val_loss) at every save_every val eval, persists params_nd_compressor_best_val.pkl + opt_state_resnet_best_val.pkl, returns (params,state,params_path,state_path) per policy. Caller uses the returned canonical path for the cache fingerprint instead of rglob'ing the last per-step pkl. last_step preserved as opt-in for replicating pre-fix campaign numbers. Smoke pilot pending.
---
