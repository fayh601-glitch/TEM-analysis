# AI workflow guide — scientific software validation sprints

How to prompt an AI assistant (Cursor, ChatGPT, etc.) for **honest validation
work** on microscopy analysis code — not feature churn.

---

## 1. Start with context

Tell the AI:

- **Goal:** e.g. "Validate TEM rod length/width against Enright 2018 Figure S2."
- **Audience:** lab intern, PI, or paper reviewer.
- **Done (science):** independent hand labels agree within X% on N rods per panel.
- **Done (engineering):** regression tests pass, docs state limitations, CI runs
  pytest on every PR.

Science "done" and engineering "done" are different. State both explicitly.

---

## 2. Ask for an honest audit first

Before requesting features, use a prompt like:

> Pretend you are a strict mentor reviewing this repo for a methods paper.
> List every place we overclaim validation, use circular references (pipeline
> output called ground truth), or lack tests. Be harsh. No fixes yet — audit only.

Require the AI to cite **files and metrics**, not vague praise.

---

## 3. Separate prompts (do not combine)

| Phase | Prompt focus | Output |
|---|---|---|
| **Audit** | What's wrong, what's missing | Bullet list with evidence |
| **Plan** | Prioritized sprint, no code yet | Ordered tasks + acceptance criteria |
| **Implement** | **One** sprint item at a time | Code + docs + test for that item only |

Combining "audit + fix everything" produces shallow changes and hidden regressions.

---

## 4. Always require

Every implementation prompt should include:

1. **Before/after metrics** — e.g. `paper_comparison.csv` length and width error %.
2. **No validation without human labels** — pipeline-vs-pipeline is reproducibility,
   not accuracy.
3. **Regression tests as acceptance criteria** — "S2A mean length within 15% of
   paper value; test fails if not."
4. **Document limitations** — which panels fail and by how much.

---

## 5. Example prompt templates

### Validation sprint (full)

```
Context: TEM rod analyzer, Enright 2018 S2 panels in data/curated/.
Science done = Fiji hand labels on 30+ rods/panel within 10% of pipeline.
Engineering done = pytest regression harness + honest docs.

Phase 1 — Audit only: find overclaims, circular validation, missing width metrics.
Phase 2 — Plan: prioritized list, no code.
Phase 3 — Implement one item at a time with acceptance criteria.

Do not add new segmentation algorithms this sprint.
Match existing code style. Do not commit unless I ask.
```

### Root-cause one panel

```
S2B pipeline mean length is 66 nm vs paper 44.4 nm (~49% error).
Audit segment.py + enright_rods preset on data/curated/s2_B_30min.png.
Report: rod count, merge/split behavior, overlay review.
Propose ONE parameter change with predicted metric impact.
Acceptance: re-run paper_comparison row for S2B; length error must drop OR
document why it cannot without algorithm change.
```

### Fiji labeling workflow

```
Create docs/LABELING_GUIDE.md and manual_v2.template.csv for Fiji hand
measurement of 30–50 rods on S2A/S2B/S2D.
Columns: particle_id, image, class, length_nm, width_nm, measured_by, notes.
Include calibration steps from data/calibration.csv.
Do not call manual_v1.csv ground truth.
```

### CI setup

```
Add .github/workflows/ci.yml: Python 3.10+, pip install -e ., pytest on push/PR.
Skip or xfail tests that require curated images if missing.
Document in scripts/README.md.
```

---

## 6. Anti-patterns (tell the AI to avoid)

| Anti-pattern | Why it's bad |
|---|---|
| "Make segmentation better" | No metric, no acceptance test |
| Adding presets without measuring | More knobs, same validation gap |
| Calling `manual_v1.csv` ground truth | Circular — it's pipeline output |
| "Validated against the paper" using only `paper_comparison.csv` | Pipeline vs published **means**, not per-rod labels |
| Skipping width error | Length can look OK while width is 130%+ off |
| One giant PR for audit + docs + tests + algorithm | Hard to review; easy to hide dishonesty |

---

## 7. Review checklist (you or your mentor)

After an AI sprint, verify:

- [ ] README / MANUAL / PROJECT_STATUS say what `manual_v1.csv` actually is.
- [ ] `paper_comparison.csv` includes `width_error_pct`.
- [ ] Regression test runs on curated images and fails when you break S2A.
- [ ] Known failures (S2B, S2D) are documented, not hidden.
- [ ] No commit message says "improved validation" without new human labels.

---

## 8. When to escalate beyond classical CV

Only after:

1. Fiji labels exist for all target panels.
2. Error is characterized (length vs width, count vs dimension).
3. Parameter tuning plateaus (documented in `paper_comparison.csv`).

Then — and only then — consider watershed tuning, ML segmentation, or new presets
with **measured** justification.
