# felt — a short tutorial

A practical introduction to *felt*, the persistent-context tool we use on this
project. Read this once and you'll know enough to start using it on your own
work. Roughly a 15-minute read.

This document is for **humans**. The agent-facing operating conventions for
this project live in [FELT_AGENT_GUIDE.md](FELT_AGENT_GUIDE.md). The canonical
agent skill body (CLI surface, frontmatter spec, maintenance protocol) lives
inside the felt plugin at `~/.claude/plugins/.../skills/felt/SKILL.md`.

---

## 1. What felt is

felt is two things bundled together:

1. **A directory of markdown files** — `.felt/` at your repo root, with one
   subfolder per concern (called a *fiber*). Each fiber is a single markdown
   file with YAML frontmatter and a body.
2. **A small CLI** (`felt`) that lets you create, list, edit, link, and close
   fibers without hand-editing files when you don't want to.

A *fiber* is one concern: an open question, a decision you made, an experiment
that's running, a finding you discovered, a bug you noticed, a sub-task you're
deferring. Fibers persist across sessions. Agents and humans both read and
write them. Six months from now you (or someone else) can open the same `.felt/`
tree and pick up where you left off.

The point of felt is **research substrate**, not project management. The thing
it does well that a TODO list doesn't is hold prose, evidence, and links —
"here's what we tried, here's why it didn't work, here's what we learned, here's
what to try next" — in a way that survives the lifespan of any single session.

## 2. When felt actually helps

felt earns its overhead when one or more of these is true:

- **The work spans more than one session.** You close the laptop tonight and
  want a different version of yourself (or a different agent) to pick up
  cleanly tomorrow.
- **There are many parallel arms.** A multi-seed campaign, a multi-variant
  experiment, a survey of conditions — each arm gets a fiber, the whole thing
  has a parent constitution fiber.
- **You want a prose paper trail.** Not just numbers but the *reasoning* — what
  hypothesis was this testing, what would falsify it, what did we conclude.
  felt is good at this because the body is just markdown.
- **Multiple agents need to coordinate.** Ralph loops in particular use felt
  as shared state: each iteration reads the current fiber set, runs work,
  appends evidence, closes or opens fibers.

Concrete examples that fit:

- A long-running parameter scan (multi-arm campaign) with a prose comparison
  report at the end.
- Tracking the open questions that emerged from a paper draft.
- Maintaining a "what I've ruled out" log on a hard debugging problem that
  spans days.
- Coordinating an overnight autonomous-loop run where you want to wake up to a
  readable status report.

## 3. When felt is the wrong tool

felt has overhead. Adding a fiber, writing the outcome, linking to neighbours,
keeping the constitution coherent — none of this is free. Don't pay it when
something else is genuinely cheaper.

- **Pure hyperparameter optimization.** Use [optuna](https://optuna.org/) +
  a SLURM array (or a hand-rolled shell loop). felt's narrative-and-memory
  layer doesn't add value when you just want a CSV at the end.
- **Bug-hunting in code.** A single sharp Claude Code session with the codebase
  in front of it will out-perform a Ralph loop iterating around a bug. The bug
  needs *depth of thought per iteration*, not breadth and persistence.
- **Anything that fits in one PR.** If you can fix it in one sitting, fix it.
  Don't ceremoniously open a fiber for a one-line change.
- **Decisions about whether the question is well-posed.** Those want a human
  in the loop, not a long-running agent.
- **Casual notes you'll throw away.** A scratch buffer or a single markdown
  file is fine.

Useful rule of thumb: **if you can describe the task as "iterate 30 times on
something that doesn't change after each iteration", felt is the wrong tool.**

## 4. Vocabulary you need

- **Fiber** — a single concern, stored as `.felt/path/to/<slug>/<slug>.md`.
- **Constitution** — a special fiber that defines the *objective* of a campaign
  or workstream, including the primary metric, the budget, the done condition.
  Everything else in the campaign nests under it.
- **Status** — `open` (filed, not yet worked), `active` (being worked), or
  `closed` (resolved, with an outcome). Use one explicitly.
- **Outcome** — a one-line summary of what happened with this fiber. Lives in
  frontmatter, surfaced everywhere. Update this whenever evidence lands.
- **Tags** — short labels for filtering: `experiment`, `decision`, `bug`,
  `deferred-question`, `ship-blocker`, etc.
- **Wikilink** — `[[other-fiber-slug]]` inside a body, creates a narrative link
  to another fiber. `felt show <id> --citations` traces backlinks.
- **History** — felt keeps an append-only event log per fiber; `felt history
  <id>` shows it.

## 5. Step-by-step: starting a new piece of work

Walk through this once and you'll have the muscle memory.

### Step 0 — does this work need felt at all?

Ask: "will I be working on this across more than one session, with more than
one branch of investigation, where I'd want a prose paper trail to survive me
forgetting it?" If no, just write a single markdown file in the repo and stop.
If yes, continue.

### Step 1 — initialize felt in your repo (once per repo)

```bash
felt init
```

Creates a `.felt/` directory. Safe to run again; idempotent.

### Step 2 — file the constitution fiber

The constitution is the contract for the whole workstream. Write it
deliberately — it's the document every future iteration anchors against.

```bash
felt add my-workstream "Push CNN auto-only FoM3 toward L1 auto+cross" \
    -t autoresearch -t sbi \
    -o "Objective: close the auto-only FoM3 gap between CNN and L1 by varying \
        compressor architecture and training schedule. Primary metric: \
        3-seed pooled FoM3 on the fiducial cosmology. Budget: 30 iterations. \
        Plateau-stop: 3 consecutive iters within ±5% of current best."
```

Then open the file at `.felt/my-workstream/my-workstream.md` and write the
body. Include at minimum:

- **Objective** — one paragraph, what success looks like.
- **Primary metric** — *one* metric, declared explicitly. Not "headline X but
  decided on Y" — that mismatch is a known failure mode.
- **Done condition** — both a budget (max iterations) AND a plateau-stop.
- **Loop status (live)** — what's currently blocking progress, what would
  unblock it.

See [FELT_AGENT_GUIDE.md](FELT_AGENT_GUIDE.md) §3 for the full set of conventions
we enforce on this project. The agent skill body's `constitution.md` reference
covers the general protocol.

### Step 3 — file fibers as concerns crystallize

Every time something becomes worth tracking — an experiment to run, a
question to defer, a finding to record, a bug to chase — file a fiber.

```bash
# An experiment that's about to run:
felt add my-workstream/exp-iter1-cdim20 "Iter 1: cdim=20 baseline" \
    -t experiment -t my-workstream \
    -o "Pending. 3 seeds @120k compressor steps, default LR. Anchor for \
        the rest of the campaign."

# An open question that emerged but isn't being worked yet:
felt add my-workstream/q-is-data-limited "Is the model data-limited or capacity-limited?" \
    -t deferred-question -t my-workstream

# A bug noticed in passing:
felt add my-workstream/bug-checkpoint-policy "Compressor returns last-step, not best-val ckpt" \
    -t bug -t ship-blocker -t my-workstream
```

The slug after `add` is both a folder name and the fiber id. Keep slugs
descriptive: `q-is-data-limited` is better than `q1` because it survives
context loss.

### Step 4 — update outcomes as evidence lands

After each experiment runs, **update the fiber's outcome with what you
learned**, not just "done". The outcome is the thing future-you reads first.

```bash
felt edit my-workstream/exp-iter1-cdim20 \
    --status closed \
    --outcome "Iter 1 NULL — pooled FoM3 18,400 ± 1.2k across 3 seeds. \
               Baseline established. Next: vary cdim ∈ {10, 32, 64}."
```

Good outcomes have:

- **A verdict** (NULL / WIN / LOSE / inconclusive).
- **A number** (the metric value, ideally with uncertainty).
- **A pointer** (where the artifacts live on disk, if relevant).
- **A next move** (what the next fiber should be).

Bad outcomes are "done", "ran successfully", or "see logs".

### Step 5 — link with wikilinks

When one fiber's reasoning depends on another, link them in the body using
`[[slug]]` syntax. This builds the narrative graph that lets future-you walk
back through how you got where you are.

```markdown
The cdim=20 baseline pooled at FoM3 = 18,400 (see [[exp-iter1-cdim20]]).
We're now testing whether deeper cdim helps — but flagging that the
[[bug-checkpoint-policy]] ship-blocker may be masking real gains.
```

`felt show <id> --citations` will then surface where this fiber is linked from.

### Step 6 — find and read what's already there

A few useful commands:

```bash
felt ls                                # open + active fibers (default)
felt ls -s all                         # everything including closed
felt ls "checkpoint" -s all            # search names/outcomes by substring
felt ls --body "cdim=20"               # search bodies (FTS5)
felt tree                              # show the containment hierarchy
felt show <id>                         # render one fiber
felt show <id> -d compact              # one-line summary
felt history <id>                      # event log
```

Use `felt ls` at the start of any session to remember what state you're in.

### Step 7 — sweep periodically

Once you have ~20-30 fibers, the tree starts to drift. A *sweep* is a
deliberate pass where you:

1. Read every open fiber.
2. Close anything resolved.
3. Update outcomes that have drifted.
4. Add wikilinks where you spot missing connections.
5. Re-tag or rename anything misclassified.

The agent skill has a `references/maintenance.md` that documents this in more
detail. A sweep takes ~30 minutes for a 30-fiber tree. Do one every ~10
iterations of substantive work, not on a clock.

### Step 8 — close the workstream

When the campaign hits its done condition (budget reached, plateau triggered,
question resolved), close the constitution with a final outcome:

```bash
felt edit my-workstream \
    --status closed \
    --outcome "Campaign closed 2026-05-19 at iter 88. Ceiling triply-confirmed \
               at FoM3 ≈ 25,000. Architecture not the lever; data-limited. \
               Final report: scripts/sbi/results/exploratory/X/REPORT.md"
```

Also close every still-open sub-fiber underneath it — either resolve them or
explicitly mark them as *deferred to a future workstream* with a pointer to
where they're now tracked.

### Step 9 — exit interview (optional but valuable)

After closing a substantial workstream, write a short retrospective in the
constitution body answering:

- **What worked.**
- **What was wasted effort.**
- **What we'd do differently.**
- **What conventions, if any, should we adopt to prevent the wasted effort?**

This is the entry point through which felt usage *improves* over time. The
"Felt / Ralph operating conventions" stanza in our project CLAUDE.md was
written from one of these.

## 6. The 5-second decision tree

When you're not sure whether to open / update / close a fiber, use these:

**Should I open a fiber for this?**
- Will this concern outlive this session? → yes, open it.
- Is this a sub-question of an existing fiber I should link instead? → open as
  nested, link with `[[parent]]`.
- Is this a one-line fix I'm doing right now? → no, just fix it.

**Should I update an outcome?**
- Did new evidence land? → yes, update.
- Did my interpretation of existing evidence change? → yes, update.
- Did I just spend 10 minutes thinking and conclude nothing changed? → don't
  update for the sake of updating.

**Should I close this fiber?**
- Is there a verdict + a number? → yes, close.
- Am I about to spawn a follow-up fiber? → close this one with a pointer to
  the new one.
- Is it "kind of done but I might come back to it"? → close anyway, with the
  outcome stating the deferred work.

## 7. Common mistakes (we've made them)

- **Mixing primary metrics inside one campaign.** The keep-rule uses pooled
  FoM3, the headline cites per-seed-mean FoM3, the overlay anchors on a third
  number. Declare *one* primary metric in the constitution and stick to it.
- **No done condition.** A campaign without a plateau-stop runs until you
  notice you've been grinding in a band of ±5% for 12 iterations. Declare a
  budget AND a plateau-stop up front.
- **Letting ship-blockers sit.** A bug fiber tagged `ship-blocker` should
  *pause new training*, not be a footnote. We sat on a checkpoint-policy bug
  for an entire 20-iteration campaign because it wasn't tagged.
- **Outcome = "done".** Future-you can't act on "done". Write what you learned.
- **No `loop_review.md`.** When using felt as a Ralph-loop substrate, every
  ~5 iterations write a "should this loop continue" verdict. Two consecutive
  "no" verdicts auto-close the campaign.

## 8. Where to go next

- **Agent perspective on our project specifically**:
  [FELT_AGENT_GUIDE.md](FELT_AGENT_GUIDE.md)
- **Canonical agent skill** (CLI surface, all references):
  `~/.claude/plugins/marketplaces/cailmdaley-felt/claude-plugin/skills/felt/SKILL.md`
- **Project operating conventions** (what we add on top of vanilla felt):
  the "Felt / Ralph operating conventions" section of
  [CLAUDE.md](CLAUDE.md).
- **The Ralph loop pattern** (felt + autonomous iteration):
  `~/.claude/plugins/.../skills/ralph/SKILL.md`.

If something is unclear, the answer is probably to read the existing `.felt/`
tree of a closed workstream end-to-end. Closed campaigns are the best teacher.
