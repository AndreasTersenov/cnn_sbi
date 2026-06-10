# Overnight CNN pipeline status
[2026-06-09 22:09:25] PIPELINE START (pid 1518292)
[2026-06-09 22:09:25] waiting for population sweep (run_flatsky_cnn_population_sweep.py) ...
[2026-06-09 22:53:40] population sweep no longer running
[2026-06-09 22:53:40] START headline-consolidate
[2026-06-09 22:53:41] PASS  headline-consolidate
[2026-06-09 22:53:41] headline table -> FLATSKY_CNN_RESULT.md (overlays added after representative corners)
[2026-06-09 22:53:41] START sbc
[2026-06-09 22:53:44] PASS  sbc
[2026-06-09 22:53:44] START lc2st
[2026-06-10 01:01:54] PASS  lc2st
[2026-06-10 01:01:54] START repr-corners
[2026-06-10 02:43:44] PASS  repr-corners
[2026-06-10 02:43:44] START final-consolidate
[2026-06-10 02:43:49] PASS  final-consolidate
[2026-06-10 02:43:49] PIPELINE DONE

Deliverables: FLATSKY_CNN_RESULT.md (root), cnn_phase/figs/ (overlays + bars), 
cnn_phase/gate_c/{sbc,lc2st,tarp_drp}/ (calibration).
