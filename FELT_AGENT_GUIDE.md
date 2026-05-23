# felt — project-specific operating guide for agents

You (the agent) are working in a repo that uses [felt](https://github.com/cailmdaley/felt)
as persistent context. The **canonical skill body** at
`~/.claude/plugins/marketplaces/cailmdaley-felt/claude-plugin/skills/felt/SKILL.md`
(+ its `references/*.md`) is the source of truth for the CLI surface, frontmatter
spec, fiber lifecycle, maintenance protocol, and transcript-extraction workflow.
**Load that skill before doing felt work — read this document on top of it, not
instead of it.** This file documents what's special about *this* project: the
operating conventions we enforce, the scope-fit rules we've learned the hard way,
and the anti-patterns we've observed.

The human-readable tutorial for the same material is [FELT_TUTORIAL.md](FELT_TUTORIAL.md).

---

## 1. When you should — and should not — use felt on this repo

felt has a real overhead per fiber. The retrospective on the
`cnn-auto-push-18-20-2026` and `cnn-auto-cross-push-18-20-2026` campaigns
established that we were over-using it. Internalize the split:

**Use felt when:**

- The workstream spans multiple sessions and you'd want a different version of
  the same agent (or a human) to pick it up cleanly without a re-onboarding.
- There are *many parallel arms* — multi-seed sweeps, multi-variant
  comparisons — where the deliverable is breadth + a comparison report, not a
  single sharp insight.
- A Ralph loop or autonomous-loop is the substrate: felt is the shared state
  those loops read from and write to.
- Persistent context, prose narrative, and `[[wikilink]]` cross-references
  carry real load — e.g. "here's how this finding contradicts a prior decision
  we'd already filed".

**Do NOT use felt when:**

- The task is **pure hyperparameter optimization**. Use optuna + a SLURM array
  (or a shell loop with a CSV at the end). felt's prose layer adds no value
  here and the iteration cost is ~10× higher.
- The task is **bug-hunting in code**. A single sharp Claude Code session
  out-performs a Ralph loop by ~order of magnitude on bugs because the bug
  wants *depth per iteration*, not breadth + persistence. The
  `cnn-auto-compressor-last-not-best-ckpt` bug sat unfixed across the entire
  cnn-auto-push campaign because Ralph never asked "wait, what's actually
  being persisted on disk".
- The task **fits in one PR**. If you can do it in a single session, do it
  inline. Don't ceremoniously open a fiber for a one-line change.
- The task is **"is this question well-posed"** — that wants a human in the
  loop, not autonomous iteration.

If you're starting a workstream that looks like the second list, push back on
the user. Suggest the alternative tool. Don't reflexively open a constitution
fiber just because the user mentioned felt.

## 2. Read the constitution first, every session

When you enter a session in a felt-tracked workstream:

1. Run `felt ls -s all -t <campaign-tag>` to inventory current state.
2. Read the campaign's constitution fiber end-to-end. It carries the
   primary metric, budget, plateau-stop, current best, and loop status.
3. Check the "Loop Status (live)" stanza of the constitution. If you find a
   wait-for-Andreas state with no unblockers met, *exit without launching
   work*. The polish-make-work pattern that ate iters 17-20 of cnn-auto-push
   is what this is designed to stop.

`felt show <constitution-id>` renders the fiber. `felt show <id> --field
primary-metric` extracts a single field if the constitution declares one.

## 3. The seven operating conventions (mandatory)

These are the same conventions documented in our project CLAUDE.md under "Felt
/ Ralph operating conventions". They are **load-bearing**, not advisory.
Adopted 2026-05-22 after the cnn-auto-push retrospective.

### 3.1 Declare ONE primary metric in the constitution.

Pick `pooled_fom3` OR `mean_of_seeds_fom3` OR `per_seed_min_fom3` — not
"headline 25k but the keep-rule uses something else". Every iteration's
keep/discard decision uses this metric. STATUS.md headline numbers must match
it. Mixing metrics is the failure mode that landed the auto-vs-cross overlay
on the wrong cross-arm baseline. If a constitution lacks an explicit primary
metric, **fix that before launching new work** — file a fiber for it if
needed.

### 3.2 Declare a budget AND a plateau-stop.

Format in the constitution's "Done condition" stanza:
> *"Auto-close when N consecutive iters land within ±X% of current best on
> the primary metric, OR when iteration count reaches M, whichever is first."*

Default for hyperparameter sweeps: **N=3, X=5%, M=30**. Ralph survey reads
this at the start of each iteration and exits without launching work when the
trigger fires.

### 3.3 `ship-blocker` is a reserved tag.

When a fiber tagged `ship-blocker` is open, you must either ship its fix in
the current iteration or explicitly demote it with rationale before launching
new training. `cnn-auto-compressor-last-not-best-ckpt` sat unfixed for the
whole campaign because it lacked this tag. If you spot a load-bearing bug,
tag it `ship-blocker` immediately.

### 3.4 Constitution must include a "Loop Status (live)" stanza.

Near the top of the constitution body, when in a wait-for-Andreas or
wait-for-compute state, list the 2-3 concrete things that unblock work
(e.g. *"(a) Andreas appends CEILING CONFIRMED; (b) Andreas requests a new
branch; (c) Andreas answers methodology fiber"*). Cold-read Ralph iterations
that find none of the conditions exit with `kill $PPID` and no commits.

### 3.5 Self-review every 5 iterations.

A self-review iteration produces a `<run-dir>/loop_review.md` append:

```
marginal-info-gained: …
current-best-delta:   …
wall-time-used:       …
verdict:              continue | close
```

If the verdict is `close` two reviews in a row, auto-close the campaign.
Goes in `loop_review.md`, **not** STATUS.md. STATUS.md is for substantive
findings.

### 3.6 Compress STATUS.md proactively.

Use `scripts/sbi/results/exploratory/tools/compact_status.py` to collapse
the calibration-ledger and lesson-tracking sections into a digest when
STATUS.md exceeds ~30 KB. Keep the last 10 substantive events + current best
+ open ship-blockers + next 3 planned moves at the top; archive the rest.

```bash
python scripts/sbi/results/exploratory/tools/compact_status.py \
    path/to/STATUS.md --keep-last 10
```

### 3.7 Pin the autoresearch driver's checkpoint policy.

Constitutions must declare which autoresearch driver they use
(`scripts/sbi/autoresearch_cnn-auto-push/run_arm.py` or
`scripts/sbi/autoresearch_cnn-auto-cross-push/run_arm.py`) AND pin its
checkpoint policy:

- New campaigns default to `--compressor-checkpoint-policy best_val`.
- Campaigns that continue a historical baseline pin to
  `--compressor-checkpoint-policy last_step` *and explicitly state that the
  reason is reproducing pre-2026-05-19 results*.

## 4. Anti-patterns we have observed

These are the failure modes the seven conventions are designed to prevent.
Recognize and short-circuit them.

### Polish-make-work

Symptom: Ralph iteration cold-reads the constitution, finds no clear
blocker, runs a minor formatting/STATUS.md/figure-polish pass to justify a
commit, and continues the loop. Iterations 17-20 of cnn-auto-push were almost
entirely this.

Fix: §3.4 (Loop Status stanza with concrete unblockers) + §3.5 (every-5
self-review with `close` verdict).

### Headline drift

Symptom: the constitution's "best so far" cites one number, the keep-rule in
the iteration log uses a different number, and the comparison report uses a
third. When the overlay finally lands, it anchors on the wrong number and
must be redone.

Fix: §3.1 (one primary metric, declared at the top of the constitution).

### Buried ship-blocker

Symptom: a real bug is noticed mid-campaign, filed as a `bug` fiber, but
never tagged `ship-blocker`. The campaign keeps grinding, generating
noise-quality results because the load-bearing bug is masking real signal.

Fix: §3.3 (any bug that invalidates the campaign's primary metric MUST be
tagged `ship-blocker` immediately, *not at the next sweep*).

### Hyperparameter-search-wearing-investigation-costume

Symptom: a campaign is structurally just "vary LR ∈ {1e-3, 3e-3, 5e-3} ×
seeds ∈ {41, 42, 43, 44, 45}" but is run as a felt/Ralph workstream with 30
prose iterations, a STATUS.md, and a constitution.

Fix: don't start the campaign. Use optuna + a SLURM array. Generate the CSV.
Write one analysis notebook at the end.

### Outcome = "done"

Symptom: closed fibers with outcome strings like "completed", "ran
successfully", or "see logs". Future-you (or another agent) cannot act on
these.

Fix: every outcome must contain a verdict, a number, a pointer to artifacts,
and a next move. See [FELT_TUTORIAL.md](FELT_TUTORIAL.md) §5 step 4.

### Cold-read drift

Symptom: each Ralph iteration is essentially independent — the prose lacks
genuine continuity between iters because each agent re-derives the headline
from scratch and doesn't engage with prior synthesis.

This is the structural per-iteration-shallowness of Ralph loops and is
*not fully fixable by procedure*. Mitigate with §3.5 (self-review checkpoints
force genuine synthesis) and choose differently whether to start a felt run
in the first place (§1).

## 5. Quick-reference workflows

Common felt operations as they apply on this project.

### Closing a fiber after an experiment lands

```bash
felt edit <campaign>/<exp-fiber-slug> \
    --status closed \
    --outcome "<verdict>: <metric>=<value>. <one-sentence interpretation>. \
               Artifacts: <path>. Next: <follow-up fiber slug or 'no follow-up'>."
```

Outcome must have all four (verdict + number + pointer + next). Update
[[wikilinks]] in the parent constitution's body if this changes the campaign's
"current best".

### Filing a new question during an iteration

```bash
felt add <campaign>/q-<slug> "Question: <text>" \
    -t deferred-question -t <campaign-tag> \
    -o "Filed during iter <N>. Hypothesis: <text>. Falsifier: <test>. Tier: 1|2|3."
```

Then add a `[[q-<slug>]]` reference in the iteration's STATUS.md row and in
the constitution's open-questions stanza.

### Spotting a ship-blocker

```bash
felt add <campaign>/bug-<slug> "Bug: <text>" \
    -t bug -t ship-blocker -t <campaign-tag> \
    -o "Noticed during iter <N>. Effect on primary metric: <quantified>. \
        Fix path: <plan>. Until fixed, NEW training is paused."
```

Then halt new training launches in the current iteration. Either fix in the
same iteration or explicitly demote-with-rationale.

### Updating a constitution's "current best"

The constitution's primary-metric headline is its load-bearing surface. When
a new best lands:

```bash
felt edit <campaign> \
    --outcome "<campaign verdict, may be evolving>"
```

Then edit the body directly to update the "Current best (live)" stanza:

```
**Current best**: pooled FoM3 = 23,986 (iter-108-Q6ON-60k, 3 seeds)
**Previous best**: pooled FoM3 = 22,400 (iter-95-cdim20-baseline)
**Cumulative budget**: 108/300 iterations
```

`felt edit --outcome` and direct body edits are both fine. The CLI doesn't
touch the body; use the Edit tool for body edits.

### Sweep / maintenance pass

Periodically (every ~10 iters of substantive work):

1. `felt ls -s all -t <campaign-tag>` — inventory.
2. Read every `open`/`active` fiber.
3. Close anything resolved (with proper outcome).
4. Demote stale ship-blockers with explicit rationale.
5. Sweep the body of the constitution: are wikilinks fresh? does the loop
   status stanza still apply?
6. Compact STATUS.md if it's >30 KB (see §3.6).

See the canonical skill's `references/maintenance.md` for the full protocol.

### Exit interview when closing a campaign

When a campaign hits its done condition (budget or plateau), close the
constitution and append an *exit interview* to its body:

```markdown
## Exit interview (YYYY-MM-DD)

**What worked**: <2-3 sentences>
**What was wasted effort**: <2-3 sentences>
**What we'd do differently**: <2-3 sentences>
**Convention updates proposed**: <bullets — feed back into CLAUDE.md §"Felt /
Ralph operating conventions">
```

The "Felt / Ralph operating conventions" section of CLAUDE.md was written from
one of these. This is the loop through which felt-on-this-project improves
over time.

## 6. What lives where (for cold-read agents)

```
.felt/                     — fiber tree, one folder per fiber
.felt/<slug>/<slug>.md     — the fiber (frontmatter + body markdown)
.felt/index.db             — felt's FTS5 search index (don't edit)
CLAUDE.md                  — project rules, including §"Felt / Ralph operating
                             conventions" with the seven mandatory rules
FELT_TUTORIAL.md           — human-readable tutorial (this file's companion)
FELT_AGENT_GUIDE.md        — this file

~/.claude/plugins/marketplaces/cailmdaley-felt/claude-plugin/skills/felt/
    SKILL.md               — canonical agent skill (load first!)
    references/*.md        — protocol references (constitution, maintenance,
                             archiving, transcripts, mining, …)
```

When an agent enters a felt-tracked session, the load order is:

1. **Load the canonical skill** (`SKILL.md`) — that's the CLI surface and
   protocol baseline.
2. **Read this file** for project-specific operating conventions.
3. **Read the relevant CLAUDE.md sections** — the project rules including
   the seven conventions.
4. **`felt ls`** for current campaign state.
5. **Read the constitution** of whatever campaign you're touching, including
   the Loop Status stanza, before launching any work.

Then proceed with the work. If at any point you're tempted to do something
that violates one of the seven conventions, push back on the user — don't
silently route around the convention. The conventions were earned with real
wasted effort; they exist for reasons.
