# Talk best practices — distilled, and applied to "Do Baryons Break HOS?"

Research pass (2026-06-14) into how to give a strong scientific conference talk, **narrative side and
slide-design side**, then mapped onto *this* 30-min talk. Use this as the standard when we build the
deck and rework the plots. Sources at the bottom — the synthesis is grounded, not invented.

The three frameworks that matter here, in one line each:
- **Doumont** (*Trees, Maps & Theorems*): one main message, top-down; opening (hook → need → message)
  / body / closing; signpost relentlessly; visuals are *redundant, stand-alone, visual*.
- **Alley assertion-evidence** (*Craft of Scientific Presentations*): each slide's title is a
  **full-sentence assertion** (the message), supported by **visual evidence**, *not* a bullet list.
- **Hull** (*How to give a great talk*, astronomy-specific): practice + timing discipline, KISS plots,
  make images HUGE, be colorblind-safe, state conclusions early and often.

---

## A. Narrative / story

1. **One main message, stated early.** Decide the single sentence you want them to remember and build
   everything to support it (Doumont). Ours is locked by the title: *"No — baryons don't break HOS:
   they're resilient to cheap scale cuts, and even BNT's apparent break is a recoverable frame
   artifact."* State it in the opening; don't withhold it for suspense (Hull: withholding the
   conclusion "is a recipe for disaster").

2. **Structure as ABT (And–But–Therefore)** (Olson) — our spine maps cleanly:
   - **AND:** weak lensing has rich non-Gaussian information; the starlet ℓ1-norm captures it and beats
     P(k) ~3× even on baryon-safe scales.
   - **BUT:** baryons threaten small scales, and the clean mitigation (BNT) *breaks* per-bin HOS —
     contours inflate; a Euclid forecast says the lost SNR is hard to recover.
   - **THEREFORE:** a channel-mixing learned compressor (or a single fixed rotation) recovers it — the
     information was never lost; and that same learned summary is, in the standard basis, already
     within ~15% of the analytical optimum and fully calibrated.
   Avoid the "AAA" failure mode (and… and… and…, a list of results with no tension) and the "DHY"
   failure mode (despite… however… yet…, too many turns). One BUT, one THEREFORE.

3. **Doumont's opening, deliberately built.** Attention-getter (relevant, slightly unexpected) → the
   *need* (Stage IV pushes us into the non-Gaussian regime where baryons bite) → the *task* (compare
   analytical vs learned HOS under SBI, and test BNT) → the *main message*. Then a one-line **preview**
   of the two acts. The attention-getter is the first impression — prepare it most carefully.

4. **Signpost relentlessly** (Doumont): preview at the end of the opening; explicit **transitions**
   between Act 1 and Act 2 and between M1/M2/M3 that both *separate* and *link* ("baryons don't break
   HOS on safe scales — but our cleanest fix does; why?"); a **review** before the conclusion. Listeners
   can't skim or rewind, so redundancy is a feature, not padding.

5. **Rule of thirds for a mixed-expert room** (Hull): first ~third understandable by everyone (why
   non-Gaussian information matters), middle for HOS/SBI experts (the comparison + BNT), last part for
   the handful of close collaborators (the mechanism / whitening) — but keep most of the room engaged
   throughout. *This is a specialist meeting (everyone does HOS/SBI/wavelets), so the "general third"
   can be compressed — but the round-table crowd spans WL and 21cm, so don't assume everyone knows
   VMIM, TARP, or BNT; define them in a line each.*

6. **State conclusions early and repeat them** (Hull, Doumont): put the answer up front, restate at
   transitions, and land it again at the close. People periodically tune out — give them multiple
   chances to catch the message.

7. **End forcefully; loop the loop** (Doumont): the close ties back to the opening question ("So — do
   baryons break HOS?"), signalled by a clear change of tone. Not a trailing "…and that's it."

8. **Tell people things they already know** (Hull): a specialist audience *enjoys* a confident recap of
   shared foundations. Don't rush the setup as if it's beneath the room.

---

## B. Slide design

1. **Assertion-evidence: the title is a sentence, the body is a picture** (Alley). Replace topic titles
   ("Results", "BNT") with a full-sentence *message* that states the so-what, supported by one visual.
   Empirically this improves audience comprehension and recall of complex material (Alley & Neeley).
   Draft headlines for our result slides are in §D below.

2. **One message per slide, one slide per message** (Doumont). If a slide carries two ideas, split it.
   Remove anything that doesn't serve that slide's single message ("eliminate the noise").

3. **Minimal text.** The audience cannot read one text and listen to another at once (Doumont, Hull).
   Few words per slide; let the figure carry the argument and *you* narrate it. "Pixels are finite;
   slides are infinite" (Hull) — split dense content across slides rather than cramming.

4. **Visuals must be redundant + stand-alone + visual** (Doumont): a deaf attendee should get the
   message from the slide alone; a blind one from your words alone. A *poor* visual is worse than none.

5. **Build complex figures progressively** (Hull, astrobites): draw axes, state their meaning, add data
   one element at a time, then the trend/interpretation. For our corner plots: consider revealing the
   ℓ1 contour first, then overlaying the CNN, so the "they nearly coincide" beat lands. Split overlays
   into multiple slides (also survives a PDF-fallback; avoid 14 overlapping fade-ins).

6. **Typography:** a single sans-serif typeface, **≤3 sizes** per figure/slide; nothing below ~7 pt at
   display size (figure-design guides). Equations in serif/Computer Modern only if shown at all (see B8).

7. **Color, deliberately and colorblind-safe** (Hull, Doumont, figure guides). Design in black-and-white
   first, then add color in light touches for emphasis/identification. **Avoid red-green; prefer
   blue-orange / blue-yellow / purple-green.** Where colors must be similar, differentiate by symbol or
   linestyle. High contrast text vs background (never red-on-black, yellow-on-white). **This matters
   acutely here: Alan Heavens (round table) and ~10% of the room are colorblind.**

8. **Don't show equations** (Hull): "θ ~ λ/D is okay, nothing more complicated." For us: show FoM₃ =
   1/√det(C₃) *once*, simply, as the definition of the metric; keep the ℓ1-norm and VMIM conceptual /
   visual, not as displayed formulae.

9. **Cite every figure on-slide** (Hull): our reused panels (Paper I figures, Zürcher, Vinciguerra,
   CosmoGridV1, Zeghal/Learn2Map) get a small attribution. Talk files end up online.

10. **The conclusion slide keeps representative plots**, not just bullet text (Hull); animate bullets
    independently so the audience isn't reading ahead.

---

## C. Figures / plots — the redo standard (this is the next work item)

Hull's KISS rule is the governing principle: *plots that are good for papers are usually bad for
talks.* Our result figures were built paper-first; for the talk they need:

- **Huge fonts.** Axis labels, ticks, legends sized for the back row, not for an A&A column. Crop
  paper-style axes and relabel intuitively (Hull: replace "erg/s/cm²/Hz" with "Brightness"). For us:
  spell out "Ω_m, σ8, w0", label the metric "constraining power (FoM₃)", avoid raw `1/√det C_3` on an
  axis.
- **One message per figure.** Each plot makes exactly the point of its slide; crop/cover irrelevant
  detail.
- **A consistent color language across the whole deck — LOCKED (Wong palette; see `TALK_FIGURE_AUDIT.md §0`):**
  - *Method axis = color:* **CNN = Wong blue `#0072B2`, L1 = Wong vermillion `#D55E00`**, in *every*
    figure. (Colorblind-safe; matches the stitched figure Andreas likes.)
  - *Basis axis (no-BNT / BNT / whitened) = texture:* **no-BNT solid, BNT hatched** (same hue), not a
    clashing color — so blue never doubles as "no-BNT."
  - Secondary grayscale cue: corners → L1 solid / CNN dashed; method-only bars → L1 hatched.
- **Progressive reveal where it helps the story** (M1 corner: ℓ1 then CNN; the BNT bars: L1 collapse,
  then CNN holds).
- **Labels near the data, not in a distant legend**, where feasible.
- **Self-contained titles** consistent with the assertion headline of the slide they live on.
- **Gotcha — never combine `hatch` with `alpha`** for bars/patches: the PNG (Agg) and PDF backends
  composite alpha-over-hatch differently, so the *same* figure looks different in `.png` vs `.pdf`
  (hit on the BNT bars). Fix: bake the "paler" look into an explicit lightened *fill* color and put the
  hatch lines in the saturated hue (a `lighten()` helper), with `rcParams["hatch.linewidth"]` fixed —
  no alpha. Renders identically in both backends. (getdist filled *contours* use alpha and are fine.)

Per-figure redo notes are tracked in `TALK_NONGAUSSIAN_CONTENT.md §5`; the new `p2_M2_cross_deleaking`
already follows most of this and is the template.

---

## D. Assertion-style headlines for the key slides (draft — bake into the deck)

Topic title → assertion (the "so what"). These are the spoken-message form; tighten for the deck.

| slide | assertion headline (draft) |
|---|---|
| S5 | "Unmodeled baryons bias cosmology more the larger the survey — catastrophically (>3σ) at full sky." |
| S6 | "Higher-order statistics shed the baryon bias by cutting a single finest scale; P(k) must cut far more." |
| S7 | "Even on baryon-safe scales, the ℓ1-norm constrains ~3× tighter than the power spectrum." |
| S9 | "BNT nulls the baryons — but its noise mixing inflates the per-bin HOS contours." |
| S13 | "An optimized learned compressor beats the analytical ℓ1-norm by only ~15% — and both are calibrated." |
| S16 | "The contours are trustworthy: every arm passes the same TARP + SBC calibration battery." |
| S17 | "A patch-local cross-map adds ~20% (the κᵢκⱼ product); the full-sphere build's 4× gain was leakage." |
| S18 | "Under BNT the per-bin ℓ1-norm collapses — but the channel-mixing CNN is lossless." |
| S19 | "The collapse is a frame artifact: one fixed rotation recovers the full ℓ1 information." |
| S22 | "Do baryons break HOS? No — and even BNT's break is recoverable. Prioritize calibrated HOS." |

---

## E. Delivery & logistics (for Andreas, not the deck)

- **Practice ≥2–3 full run-throughs** (Hull): the first is always rough; the second is much better.
  Time the second. Practice once within 24 h, and click silently through all slides a few hours before.
- **Timing discipline:** 30-min slot → plan ~25 min of material, leave room for questions/overruns;
  *do not run over* — and never gloss the conclusions because you ran long (Hull). Conclusions stated
  early hedge against this.
- **Know your slide order** so you can punctuate transitions (and jump to backup slides by number).
- **Backup slides** for detailed questions: lever decomposition (NDE +36% / arch +6%), the cross-map
  leakage scale decomposition, prior-shrinkage / sharpness-vs-calibration, the parked M4/M5/2D-1D work.
- **Q&A** (Doumont): listen → repeat/rephrase → pause/think → answer briefly to the whole room. It's
  fine to say "I don't know — good avenue, let's talk after" (Hull) — honest > bluffing with this crowd.
- **Tech:** bring dongles + a USB PDF fallback; disable notifications; have your own clicker; **green**
  laser, not red (Hull, colorblindness).
- **Delivery (Doumont):** speak extemporaneously from a memorized outline (not a script); kill filler
  words with silence; eye contact; slow down (many non-native English speakers in an astro audience).

---

## Sources
- Jean-luc Doumont, *Effective oral presentations* (summary of *Trees, Maps, and Theorems*),
  principiae.be — [PDF mirror](https://www.cs.tufts.edu/~nr/cs257/archive/jean-luc-doumont/oral.pdf).
- Michael Alley, *The Craft of Scientific Presentations* / the assertion-evidence approach —
  [assertion-evidence.com](https://www.assertion-evidence.com/), and Alley & Neeley, "Rethinking the
  design of presentation slides: A case for sentence headlines and visual evidence"
  ([PDF](http://writing.engr.psu.edu/2005_alley_neeley.pdf)).
- Chat Hull, *How to give a great talk* (astronomy-specific) —
  [arXiv:1712.08088](https://arxiv.org/abs/1712.08088).
- Randy Olson, *Houston, We Have a Narrative* — the ABT framework
  ([overview](https://abtagenda.substack.com/)).
- astrobites, "Speak your science: How to give a better conference talk"
  ([Part 1](https://astrobites.org/2018/02/10/speak-your-science-part-1/),
  [Part 2](https://astrobites.org/2018/02/17/speak-your-science-part-2/)).
- "One Message per Slide. One Slide per Message."
  ([AstroBetter](https://www.astrobetter.com/blog/2011/01/21/learning-to-give-better-talks-one-message-per-slide-one-slide-per-message/)).
- Figure-design specifics (font ≥7 pt, ≤3 sizes, sans-serif, blue-orange over red-green, minimalism):
  [Nature/Science figure guidelines](https://conceptviz.app/blog/how-to-make-figures-for-nature-science-journals),
  [ColorBrewer](https://colorbrewer2.org).
