# First prompt for the Fable 5 session (copy the block below)

Suggested Claude Code setup: model `claude-fable-5`, effort `high` (use `xhigh` for the scientific
reasoning / reframing and the bug audit). Working directory: `/mnt/home/tersenov/software/cnn_sbi`.

---

I'm continuing a weak-lensing SBI project (cnn_sbi repo on titan, conda env `jaxili`). We're comparing
two summary statistics — a wavelet-L1 datavector and a CNN-VMIM learned compressor — for inferring
cosmology from tomographic convergence maps built into **patch-local (flat-sky) cross-maps**. The
previous session (Opus 4.8) finished the de-leaked L1-vs-CNN comparison; you're picking it up.

**Start by reading `HANDOFF_FABLE5_2026-06-10.md` in the repo root — it's the entry point and links
everything** (the result, the calibration, the live scientific question, the running job, the file
map, and the guardrails). Read it and the docs it points to (esp. `FLATSKY_CNN_RESULT.md`,
`CROSS_MAP_LEAKAGE_FINDING.md` §6, and the memory index) before doing anything.

Context for why this matters: this is for a paper (and my thesis) comparing whether a learned CNN
compressor or a hand-crafted wavelet statistic better extracts cross-tomographic-bin cosmological
information from data a real survey could actually build. The headline so far is that on the
physically-buildable cross, L1 gains ~+20% while the CNN gains nothing — but I think that's likely
**compressor training inefficiency, not a real method limitation** (the CNN gets the bins as channels,
so it should be able to learn their cross-correlations; and the best seed already beats L1 on
auto-only). §4 of the handoff lays out my reasoning and the caveats precisely — treat it as the
central open question, not settled.

What I'd like you to do, in order:

1. **Check the experiment that's running** (a multi-compressor-seed check, handoff §5 — do NOT
   relaunch it; check `…/cnn_phase/multiseed/driver.out` and `MULTISEED_COMPRESSOR_CHECK.md`). When it
   finishes, interpret it against §4: does a well-trained compressor lift `product` toward/over L1, and
   does the no-cross-gain hold across compressor seeds? Then **reframe `FLATSKY_CNN_RESULT.md` and the
   relevant memory** along the optimization-limited framing **if the numbers support it** (and tell me
   if they don't). Commit + push when done.

2. After that, the broader threads I'd like to pursue (you scope and prioritize, then check with me
   before any long GPU campaign): a **principled best-seed comparison** (select by validation loss, not
   post-hoc FoM3; handoff §6.2); **implementing BNT for this flat-sky cross setup** and producing a
   calibrated BNT L1-vs-CNN result (§6.3 — scope the design first, it's a multi-day campaign); and a
   **bug/inefficiency audit of our analysis** (§6.4 — you're good at this, please use parallel
   subagents to fan it out: example-disjointness, the RMS whitening path, the sweep aggregation, and
   the ~2 h/arm sampling throughput are the high-value targets).

How I work: plan non-trivial work to a markdown file and get my sign-off before building or launching
GPU jobs; honor the guardrails in the handoff (GPU 1 only unless I grant more — and re-check nvidia-smi
for other tenants; stage git by path, never commit the big `.npz/.pkl` artifacts; titan has no
scheduler so run jobs detached). We track campaign state in a **felt fiber** (handoff §0 explains it) —
read its top loop-status stanzas for the live state, and keep it updated (prepend a short stanza when
state changes), committing the fiber `.md` alongside your work. When I'm thinking out loud or asking a
question, give me your assessment and stop — don't change code until I ask. Read everything first, confirm back your
understanding of the state and the open question, then propose how you'd tackle (1). Interview me if
anything is ambiguous.
