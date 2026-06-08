# felt — a practical day-in-the-life workflow

A hands-on companion to `FELT_TUTORIAL.md` (concepts) and `FELT_AGENT_GUIDE.md`
(agent conventions). This one answers: *"What do I actually do, step by step, to
run a campaign with felt on this project?"* — using the live
`definitive-l1-vs-cnn-2026-05` campaign as the worked example.

---

## 0. The question everyone asks first: do I drive felt inside or outside Claude?

**felt is a standalone command-line tool. It has nothing to do with Claude.**

- It's a binary at `~/.local/bin/felt`. You run it in any terminal: `felt ls`,
  `felt add …`, `felt edit …`.
- The data is just **markdown files** under `.felt/` (one folder per fiber) plus a
  search index (`.felt/index.db`). You can open/read/edit those `.md` files in vim,
  VS Code, anything.
- During a Claude Code session you normally **let Claude run the felt commands** (it
  has a shell). But "Claude's felt" and "your felt" are the same thing — both just
  call the `felt` CLI on the same `.felt/` files. No conflict, no separate state.

So there are three equivalent ways to touch felt, and you'll use all of them:

| You want to… | Do this |
|---|---|
| Glance at campaign state | `felt ls` / `felt tree` in a terminal (no Claude needed) |
| Have the day's work tracked as it happens | tell Claude "this is a felt campaign, keep the tree honest" — it runs the CLI |
| Make a campaign-level decision | edit the constitution's `.md` body yourself (vim/VS Code), or have Claude do it |
| Quick metadata change (status, outcome) | `felt edit <id> --status … --outcome "…"` — you or Claude |

**Rule of thumb:** *structure and metadata* (creating fibers, status, outcome, tags)
→ use the **CLI** (`felt add`/`felt edit`), because it keeps the event log + search
index consistent. *Prose bodies* → edit the `.md` file directly with a text editor.
Both are fine; just don't hand-edit the YAML frontmatter at the top of a fiber file
(use `felt edit` for that).

---

## 1. What a fiber actually is (so the commands make sense)

A **fiber** is one concern, stored as `.felt/<path>/<slug>/<slug>.md`. Open one and
you'll see two parts:

```markdown
---                         # <- YAML frontmatter (metadata; edit via `felt edit`)
name: Multi-perm obs recompress used the wrong flag
status: closed              # open | active | closed
tags: [bug, definitive]
created-at: 2026-06-01T…
outcome: 'FIXED: used --no-train (that is for the FLOW)… Next: per-perm-average …'
---

The gated design meant the wrong flag cost no compute. Part of                # <- body
[[definitive-l1-vs-cnn-2026-05]]; see also [[refine-phase-c-perm-matched]].    #    (prose; edit in any editor)
```

The vocabulary you need:

- **Fiber** — one concern (a question, decision, experiment, finding, bug, deferred task).
- **Constitution** — the special fiber that defines a whole *campaign*: its objective,
  the **one** primary metric, the budget, the done-condition. Everything else nests
  under it. Ours is `definitive-l1-vs-cnn-2026-05`.
- **status** — `open` (filed, not worked), `active` (being worked), `closed` (resolved,
  has an outcome). Shown as ○ (open/active) and ● (closed) in `felt tree`.
- **outcome** — the one-line conclusion. The single most important field — it's what
  future-you reads first. Must have **verdict + number + pointer + next** (see §6).
- **tags** — labels for filtering (`felt ls -t bug`). On this project: `experiment`,
  `bug`, `finding`, `decision`, `deferred-question`, `ship-blocker`, plus the campaign
  tag (`definitive`).
- **wikilink** — `[[other-fiber-slug]]` in a body, links one fiber to another. **Only
  for fibers** — a memory or a `.md` doc is NOT a fiber, so write its name in plain
  prose. `felt check` flags `[[non-fiber]]` as a broken reference.

---

## 2. The day-in-the-life

Marked **[you]** vs **[Claude]** so it's clear who does what. Most mechanics are
Claude's job; yours is scope, steering, and a sanity check.

### 0. [you] Scope-check — the only decision that really matters

Before opening or continuing a campaign, ask: *"Is this multi-session, multi-arm,
and worth a prose paper trail?"*

- **Yes** (the definitive comparison; a multi-seed sweep) → it's a felt campaign.
- **No** ("fix this bug", "make this plot", "is this question even well-posed") →
  **don't touch felt.** Open a sharp Claude session and just do it. Opening a fiber
  for a one-off is the #1 over-use mistake (your own retrospective established this).

### 1. [you] Connect and land in the repo
```bash
ssh <cluster>
cd ~/software/cnn_sbi          # felt lives in ./.felt/
```

### 2. [you] See where things stand — *before* any work
```bash
felt ls                                  # open + active fibers
felt tree                                # the hierarchy (constitution + children)
felt show definitive-l1-vs-cnn-2026-05   # render the constitution
```
This 2-minute read replaces re-onboarding. Right now you'd see the constitution
(open) with three ● closed fibers and one ○ open fiber — that open one is the task.

### 3. [you] Start Claude and point it at the campaign
Launch Claude Code, paste the init prompt (in the handoff). The prompt makes Claude
load the felt skill, run `felt ls`, read the constitution + the **open** fiber, verify
state, and **propose a plan**. You read the plan → approve or redirect.

### 4. [Claude, you approve] The open fiber *is* the task
```bash
felt show definitive-l1-vs-cnn-2026-05/refine-phase-c-perm-matched
```
Its body/outcome holds the to-do list. Claude works it; you supervise.

### 5. [Claude, *as you go* — not batched to the end] File fibers as concerns crystallize

This is the habit my session got wrong (I filed everything at the very end). The
right rhythm: **the moment something becomes worth tracking, give it a fiber.**

Anatomy of the command:
```bash
felt add  <slug>  <name>  [flags]
#         ^path/id ^title
#  flags:  -t <tag> (repeatable)   -s <status>   -o "<outcome>"   -b "<body>"
```
- **`<slug>`** is the id *and* the folder name. `campaign/sub-slug` **nests** it under
  the campaign. Make it descriptive so it survives context loss, and prefix by type:
  `exp-…`, `bug-…`, `q-…`, `finding-…`. (`bug-multiperm-no-train-flag`, not `bug1`.)
- **`<name>`** is a human sentence — the title you'll see in `felt ls`.
- **`-t`** tags it (repeatable). Always include the campaign tag (`definitive`).
- **`-s`** sets status. Omit → `open`. A finding you *already have* → `-s closed`.
- **`-o`** is the outcome (for a pending experiment: `"pending: <what it tests>"`; for
  something already resolved: the full verdict — see §6).

Worked examples **for this campaign**, with what each is for:

```bash
# (a) An experiment you're ABOUT to run — file it open/pending so the plan is on record:
felt add definitive-l1-vs-cnn-2026-05/exp-perm-averaged-table \
    "Re-aggregate Phase C with per-perm-averaged multi-perm" \
    -t experiment -t definitive \
    -o "pending: fix aggregate_all_arms.py to per-perm-average (not pool) the multi-perm arms."
#   -> creates .felt/definitive-l1-vs-cnn-2026-05/exp-perm-averaged-table/…
#      shows up in `felt ls` as open. Close it (step 6) once it runs.

# (b) A BUG you notice in passing — tag it ship-blocker if it invalidates the metric:
felt add definitive-l1-vs-cnn-2026-05/bug-<slug> \
    "Compressor cache fingerprint ignores X" \
    -t bug -t ship-blocker -t definitive \
    -o "Noticed during the rerun. Effect on primary metric: <quantified>. Until fixed, NEW training paused."
#   -> `ship-blocker` is a CONVENTION (not enforced by felt): while a ship-blocker
#      fiber is open, you don't launch new training until it's fixed or explicitly
#      demoted with a reason. It's the thing that stops a real bug from silently
#      poisoning a whole campaign.

# (c) A QUESTION that emerged but you're NOT working yet — park it so it isn't lost:
felt add definitive-l1-vs-cnn-2026-05/q-within-route-l1-vs-cnn \
    "Does the L1-vs-CNN verdict hold within a single data route?" \
    -t deferred-question -t definitive \
    -o "Filed 2026-06-01. Hypothesis: route confound inflates the gap. Falsifier: a within-route run. Tier: 2."

# (d) A FINDING that just landed — file it already-closed, with the full result:
felt add definitive-l1-vs-cnn-2026-05/finding-<slug> \
    "Standardization is neutral" \
    -t finding -t definitive -s closed \
    -o "NULL: std FoM3 24281 vs 26748, sigma marginally tighter -> z-scoring does not destroy info. Artifacts: SUMMARY_DEFINITIVE.md. Next: none."
```

### 6. [Claude] Close fibers with *real* outcomes as evidence lands
```bash
felt edit definitive-l1-vs-cnn-2026-05/exp-perm-averaged-table \
    --status closed \
    --outcome "DONE: per-perm-average implemented; multi-perm now comparable to perm-0 rows. Artifacts: PHASE_C_2026_05_31/SUMMARY_DEFINITIVE.md. Next: [[finding-...]]."
```
A good outcome has all four: **VERDICT** (WIN/LOSE/NULL/FIXED) · **a NUMBER** ·
**a POINTER** (where the artifact lives) · **a NEXT** (slug or "none"). Never `"done"` —
future-you can't act on "done".

### 7. [Claude] Link the narrative
In bodies, `[[fiber-slug]]` connects reasoning (e.g. the bug fiber links the refinement
fiber that fixes it). `felt show <id> --citations` shows what links *to* a fiber.
Wikilinks are **for fibers only**; memories/docs go in plain prose.

### 8. [you] Steer via the constitution, not the chat
When you make a campaign-level call — "the primary metric is now σ/2D", "CEILING
CONFIRMED", "deprioritize the clean rerun" — **write it into the constitution body /
its "Loop Status (live)" stanza** (edit the `.md`, or have Claude do it). A decision
said only in chat evaporates; one in the constitution is the contract every future
session reads.

### 9. [you, ~1 min at end of session] The honesty check — *expanded*

The goal: someone running `felt ls` + reading the constitution should know **exactly**
where things are, with zero surprises. Concretely, run `felt ls` and ask:

1. **Any fiber marked `open`/`active` that's actually finished?** e.g. you see
   `exp-perm-averaged-table` still `active` but the experiment ran this afternoon →
   close it: `felt edit <id> --status closed --outcome "…"`. (Stale-open fibers are
   the most common drift.)
2. **Did you do real work that has NO fiber?** e.g. you discovered a finding but never
   filed it → `felt add … -s closed -o "…"` now, while you remember it.
3. **Is the constitution's "Loop Status (live)" describing *today*'s state**, or a
   stale snapshot from three days ago? Update the one or two lines that drifted
   (what's done, what's the current task, what's blocking).

That's it — three questions, a minute or two. It's the difference between a tree that
helps the next session and one that lies to it.

### 10. [~every 10 substantive iterations] The sweep — *expanded*

The daily honesty check is light; the **sweep** is the periodic deeper tidy that stops
a 20–30-fiber tree from rotting. `felt ls -s all` (everything, including closed), then:

1. **Close resolved fibers.** Anything done-but-still-open → `felt edit <id> --status
   closed --outcome "<verdict+number+pointer+next>"`.
2. **Fix stale outcomes.** A fiber whose outcome is now *wrong* because later evidence
   changed the conclusion (e.g. "L1 leads" but a later run overturned it) → rewrite the
   outcome so it tells the current truth. `felt edit <id> --outcome "…"`.
3. **Add missing wikilinks.** You spot that fiber A's logic depends on fiber B but
   they're not connected → add `[[B]]` in A's body. This is what makes the tree a
   *graph* you can walk, not a flat list.
4. **Re-tag/rename misclassified fibers** (`felt edit <id> --tag …`; rename is rare).
5. **`felt check`** — the linter. It catches broken wikilinks (`[[thing]]` that isn't a
   fiber), missing outcomes on closed fibers, and other structural issues. **It must be
   clean.** Fix what it reports (usually: un-bracket a memory name, or write an outcome).

Takes ~30 min for a 30-fiber tree. Do it on *work cadence* (every ~10 iters), not a
clock. After a sweep, the tree is trustworthy again.

### 11. [you, when the campaign is truly done] Close it
```bash
felt edit definitive-l1-vs-cnn-2026-05 --status closed \
    --outcome "<final verdict + headline number + path to the report>"
```
Close any still-open sub-fibers (resolve them, or mark them deferred-to-a-future-
workstream with a pointer). Then append an **Exit Interview** to the constitution body:

```markdown
## Exit interview (YYYY-MM-DD)
**What worked:** …
**What was wasted effort:** …
**What we'd do differently:** …
**Convention updates proposed:** …  (these feed back into CLAUDE.md)
```

That retrospective is the loop through which your felt usage *improves* — the
"Felt / Ralph operating conventions" in `CLAUDE.md` were written from one.

---

## 3. Command cheat-sheet

```bash
# look around
felt ls                          # open + active fibers
felt ls -s all                   # include closed
felt ls -t bug                   # filter by tag
felt ls "checkpoint" -s all      # search names/outcomes
felt ls --body "cdim=20"         # full-text search bodies
felt tree                        # containment hierarchy
felt show <id>                   # render a fiber
felt show <id> -d compact        # one-line
felt show <id> --citations       # what links to it
felt history <id>                # append-only event log

# create / change
felt add <campaign>/<slug> "Name" -t tag1 -t tag2 -s open   -o "pending: …"
felt add <campaign>/<slug> "Name" -t finding      -s closed -o "VERDICT: … Next: …"
felt edit <id> --status closed --outcome "VERDICT: number. interp. Artifacts: path. Next: slug"
felt edit <id> --tag newtag                       # add a tag
felt nest <id> <new-parent>                       # move a fiber under another
felt rm <id>                                      # delete (rare)

# health
felt check                       # lint — must be clean (broken [[links]], missing outcomes…)
```
Body prose is edited in a **text editor** (vim/VS Code) on the `.md` file — the CLI
manages frontmatter + the event log, not the body.

---

## 4. When NOT to use felt (worth re-stating)

Don't open a fiber / don't run it as a felt campaign for:

- **Pure hyperparameter optimization** → optuna + a SLURM array + one analysis notebook.
- **Bug-hunting in code** → one sharp Claude session; the bug needs depth, not breadth.
- **Anything that fits in one PR / one sitting** → just do it.
- **"Is this question even well-posed?"** → that wants you in the loop, not a fiber.

Rule of thumb: *"iterate 30 times on something that doesn't change after each iteration"*
→ felt is the wrong tool.

---

## 5. The whole thing in one breath

**`felt ls` to start → file a fiber the moment a concern appears → close it with a real
outcome the moment evidence lands → put campaign-level decisions in the constitution →
sweep every ~10 iters → exit-interview at the end.**

And the part that should relax you: **you mostly steer.** Your real jobs are the
scope-check (don't over-use it), writing campaign decisions into the constitution, and
the 1-minute honesty check. The CLI mechanics — `felt add`/`felt edit` as work happens —
you can hand to Claude; just tell it "this is a felt campaign, keep the tree honest as
you go," and check `felt ls`/`felt tree` now and then to confirm it did.
