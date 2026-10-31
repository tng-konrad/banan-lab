Write a post characterizing 

A reusable specification for producing a new post in the exact voice, structure, and
formatting of the existing series on time series https://konradb.substack.com/. 
Given **(a)** a subject and **(b)** a companion notebook with the implementation, follow this recipe to stay consistent.

The subject is a replication of the results of a 

---

## 1. The one-sentence brief



Everything below is in service of that brief.


## 3. The overall arc (section skeleton)

Follow this macro-structure. Section *names* vary with content, but the **shape** is
constant:

1. **Hook + framing** (2–4 short paragraphs, no header).
2. **Companion notebook line** (the 📓 link).
3. **Groundwork / theory sections** — one section per "building block", ordered by
   increasing complexity (the ladder). Each introduces a method, its intuition, its
   equation, its knobs, and often a picture.
4. **Application section** — put the pieces together on a real dataset, following a
   visible workflow (EDA → model choice → fit → diagnostics → error metrics).
5. **A method-comparison or "can we do better?" beat** — honest results, often the
   turning point to the next rung.
6. **`Closing time`** — always this exact header. Recap the ladder; state the throughline.
7. **(Optional) `Key Takeaways`** — a numbered list (used when the post is long or has
   multiple competing methods, e.g. episode 4).
8. **Subscribe footer** (verbatim block, see §11).

> Section headers in the source render *after* their body text in the PDF extraction
> (an artifact of the export). When authoring, place each header **before** its content
> as normal Markdown.

---

## 4. The opening hook — patterns to rotate

The first 2–4 paragraphs must earn the reader's attention before any math. Pick one of
these proven openers (each appears in the corpus):

- **"You'd expect X, but actually Y" / historical reframing.**
  *"When you think about forecasting the future, you might imagine complex neural
  networks… but some of the most elegant tools emerged in the 1950s."*
- **"These were the workhorses until…" (heritage + still-relevant).**
  *"Until the 1980s, ARIMA models were the workhorses of time series analysis."*
- **"This structure is everywhere" (ubiquity).**
  *"Hierarchical structures show up everywhere: organizations, geographies, product
  catalogs…"*
- **"We usually ask X; this asks a different question."**
  *"In most time series problems, we ask what happens next. Survival analysis asks…
  when will it happen?"*
- **"The dangerous assumptions are the ones you don't notice."**
  *"Almost everything we've covered so far quietly assumes that something happens every
  period… Then you open a real retail catalog and that assumption dies on contact."*

**Then**, in the same opening, always:
- State the **core principle** of the post in one bolded or set-apart line
  (e.g. *"recent observations matter more than older ones."*).
- Preview the journey: name the methods you'll climb through, in order.
- Reassure on scope: *"The goal is not mathematical rigor, but practical understanding."*

---

## 5. The companion-notebook line

Immediately after the hook, drop the notebook link on its own, in this exact form:

```
📓 All code lives in the companion notebook:
https://github.com/tng-konrad/time-series-with-Konrad/blob/main/mNN-<slug>.ipynb
```

Use the zero-padded module number and a short hyphenated slug matching the topic
(`m02-smoothing-methods`, `m03-arima-family`, `m04-hierarchical-forecasting`,
`m05-survival-analysis`, `m06-sales-and-demand`).

---

## 6. How to write a single "rung" (theory building block)

Each method gets its own section that follows this internal template. Do **not** include
every element every time, but this is the menu and the order:

1. **Motivate it from the previous rung's failure.** Each method exists because the last
   one broke. *"Flat forecasts break down as soon as the data starts trending. To fix
   that, we need to explicitly model the trend."* → next section is Double Smoothing.
2. **One-line definition** in plain language, then the name(s) it's known by
   (*"Double exponential smoothing, a.k.a. the Holt method"*).
3. **The equation(s)**, presented cleanly as display math. Keep notation minimal and
   consistent across the whole series (see §8).
4. **"Let's unpack" / "In plain English"** — immediately translate the equation into a
   sentence a practitioner would say out loud. Never leave an equation un-narrated.
5. **A short bullet list of properties / intuitions** (3–6 bullets, each ~1 sentence).
6. **What the knobs do** — for each smoothing/tuning parameter, contrast the extremes:
   - `High α`: … (responsive but noisy)
   - `Low α`: … (smooth but sluggish)
7. **A picture** and a one-paragraph reading of it (*"As you can see from the graphs
   above…"*).
8. **Optional callouts**: a `Pro tip:` (italicized) for a real-world gotcha, or a
   `caveat` ("always worth repeating").

Keep each rung tight. The reader should feel the method click, not get a textbook.

---

## 7. Voice & tone rules

The narrator is an experienced, generous practitioner talking to a smart peer. Concretely:

- **First-person plural for the work** ("we simulate", "we split the data"), **second
  person for the reader** ("you might imagine", "if you've ever done actuarial work").
- **Practical over rigorous.** Explicitly wave off heavy theory when it would exceed the
  promised level (*"The proper analysis… involves characteristic polynomials, which is
  above the math level I promised"*).
- **Honest about results.** Report when the sophisticated method underperforms:
  *"Minor improvement at the lowest level, but overall inferior to TopDown."*
  *"adding exogenous variables gives mixed results… Takeaway: more complexity does not
  always mean better forecasts."* This honesty is a signature — never oversell.
- **Light, dry humor.** Emoticons `;-)` and `:-(` and `:-/`; the occasional meme
  reaction (Gandalf/"I was there 3000 years ago", Bob the Builder "YES WE CAN"). Roughly
  one comedic beat per major section, not more.
- **Rhetorical questions as pivots.** *"Can we do better?"*, *"What about the
  prediction?"*, *"How does it work?"* — used to hand off between sections.
- **Relatable admissions.** *"(at least it wasn't to me when I learned it)"*. Lowers the
  barrier; keep them parenthetical and brief.
- **Parenthetical / italic asides** carry the personality. Use them for jokes, caveats,
  and cross-references — not for load-bearing content.
- **Analogies that transfer intuition**, ideally to a domain the reader may know:
  Bühlmann credibility, "borrow strength", "borrow a voice", "instantaneous churn
  pressure", "the NA of survival analysis".
- **No hype, no filler.** Sentences are direct. Avoid "genuinely/honestly/actually" tics.

---

## 8. Notation & math conventions

- Present equations as **clean display math**, then unpack in prose. One idea per
  equation.
- Reuse the series' standing notation so returning readers aren't re-learning symbols:
  - `X_t` observation, `S_t` smoothed level/state, `b_t` trend, `c_t` seasonal component.
  - `α` level smoothing, `β` trend smoothing, `γ` seasonal smoothing; all in (0,1).
  - Forecast: `X̂_{t+h}`; horizon `h`; season length `L` (and note the `> 2L` sample
    rule when seasonality appears).
  - AR/MA: `φ` (AR coefficients, order `p`), `θ` (MA coefficients, order `q`), `ε_t`
    white noise. ARIMA `(p,d,q)`, SARIMA `(p,d,q)(P,D,Q)[m]`.
  - Survival: `T` event time, `S(t)` survival function, `h(t)`/`λ(t)` hazard,
    `Λ(t)` cumulative hazard.
  - Hierarchical: `y_t = S b_t` (summing matrix `S`), reconciliation `ỹ = S G ŷ`.
- When you introduce a **named model class**, bold it on first use (**Single Exponential
  Smoothing**, **ARMA(p,q)**, **MinT**, **Croston**).
- If a convention is ambiguous in the wild, flag it (episode 2's `Pro tip` about whether
  small α means heavy or light smoothing).

---

## 9. The experiment / application workflow

When you reach the "put it together on real data" section, walk the standard workflow
explicitly and in order. Readers should be able to reproduce the notebook's logic from
the prose:

1. **Introduce the dataset** in one line with units and provenance (*"daily US energy
   consumption (in billion kWh)"*; *"quarterly changes in U.S. aggregate savings from
   FRED"*; *"M5 competition data (daily Walmart unit sales, California stores)"*;
   Kaggle telco churn). Reuse datasets across episodes when natural and say so
   (*"introduced in the previous episode"*).
2. **Quick EDA** — usually a seasonal decomposition (compare additive vs multiplicative
   and justify the pick via residual stability), and/or a raw plot.
3. **Test assumptions formally when relevant** — e.g. ADF/Dickey–Fuller for stationarity,
   reported as:
   ```
   ADF Statistic: -18.705760
   p-value: 0.000000
   ```
   and translated (*"No red flags… this means we don't see an overall linear trend"*).
4. **Choose order/params** — show the ACF/PACF read, then automate
   (stepwise AIC search; `auto_arima`-style output snippets are fine to paste).
5. **Train/validation split** with an explicit chronological cutoff (*"Our cutoff point
   will be 2005"*). Stress: **never shuffle a time series**.
6. **Fit, then diagnostics** — residual ACF/PACF, the four-panel statsmodels diagnostic
   (standardized residuals, histogram+KDE vs N(0,1), Q–Q, correlogram). Read them
   honestly (*"diagnostics show remaining seasonal structure"*).
7. **Report error metrics** as a literal dict, low-key, in monospace:
   ```
   {'MAE': 40.46, 'RMSE': 48.52}
   ```
   Then one sentence of context (*"reasonable given the model's simplicity and the long
   forecast horizon"*). For hierarchical/level-wise results, use a small
   `level | baseline_rmse | reconciled_rmse | delta_rmse | improved` table.
8. **Interpret the forecast plot** — name what it does right and where it fails, and tie
   the failure to a model assumption (this failure usually motivates the next rung or the
   next episode).

---

## 10. Formatting conventions

- **Headers:** sentence/near-title case, short. Signature recurring header: `Closing
  time`. Some episodes also use `Key Takeaways`, `Groundwork`, `Baseline`.
- **Bold** for: first use of a named method/term, and for the single most important
  takeaway sentence in a passage (*"recent observations matter more than older ones."*,
  *"more complexity does not always mean better forecasts"*).
- **Italics** for: asides, cross-references, book/paper titles (*Forecasting: Principles
  and Practice* by Hyndman & Athanasopoulos), and gentle emphasis.
- **Bulleted lists** for properties, advantages/disadvantages, strengths/limitations.
  A very common device is paired **Advantages: / Disadvantages:** (or **Strengths /
  Limitations**) blocks with bolded lead-ins per bullet:
  - **Granularity preservation:** …
  - **Signal-to-Noise ratio:** …
- **Numbered lists** for sequential mechanisms ("Mechanism:" steps) and for final
  takeaways.
- **Monospace / code style** for raw tool output: metric dicts, ADF stats, model-search
  lines (`ARIMA(0,1,0)(0,0,0)[0] intercept`), and inline code (`cph.plot_partial_effects_on_outcome(...)`).
- **Cross-references** to earlier episodes are inline links on a short anchor word
  (*"the single exponential smoothing method (reminder: here)"*, *"check out Episode 1
  for a refresher"*). Add these whenever you reuse a concept.
- Keep paragraphs short (2–5 sentences). White space is part of the voice.

---

## 11. Closing conventions

**`Closing time`** section, every episode:
- Recap the ladder in one paragraph — frame the whole post as a single idea getting
  steadily more capable (*"we explored exponential smoothing as a progression of
  increasingly expressive models: from estimating a stable level, to tracking trends,
  and finally to capturing recurring seasonal patterns."*).
- State the honest trade-off / throughline (*"a sparse series can't speak for itself, so
  you let it borrow a voice…"*).
- **Optionally** forward-reference the next episode (*"In the next installment, we'll
  move beyond smoothing to explore ARIMA models…"*).
- If the post is method-heavy, add a numbered **Key Takeaways** block (see episode 4:
  "No Free Lunch", "Heuristics are still useful", etc.). The **"no free lunch"** motif —
  no method dominates everywhere — is a recurring closer.

**Subscribe footer** — paste verbatim at the very end:

```
United States of Banan is a reader-supported publication. To receive new posts
and support my work, consider becoming a free or paid subscriber.

[ ✓ Subscribed ]
```

---

## 12. Visual / image conventions

- **Header image / interior art:** stylized, abstract portrait-of-a-woman-meets-data-viz
  pieces (line-art faces fused with candlesticks, waveforms, or bar charts; high-contrast
  palettes on black). One at the top and/or bottom.
- **Reaction memes:** one, occasionally, at a comedic pivot ("Can we do better?" → meme).
  Keep tasteful and sparse.
- **Plots** are the real payload: observed-vs-smoothed/forecast line charts (with a
  legend: *Observed / Smoothed(param) / Forecast*), seasonal-decomposition panels,
  ACF/PACF stem plots, four-panel diagnostics, heatmaps of the G/summing matrix,
  survival/hazard curves with confidence bands, Brier-score comparisons, feature-importance
  bars, shrinkage scatter (posterior vs MLE with a y=x reference line).
- Every non-trivial plot gets a **one-paragraph reading** right after it. Never drop a
  figure without telling the reader what to see in it.

---

## 13. Fill-in-the-blank skeleton

Copy this and populate from the subject + notebook.

```markdown
# Time Series with Konrad: episode {N}
{playful one-line subtitle}

KONRAD BANACHEWICZ
{MON DD, YYYY}

{Hook: 2–4 short paragraphs using one opener from §4. End on the core
principle (bold/standalone line) and a preview of the methods to come.}

📓 All code lives in the companion notebook:
https://github.com/tng-konrad/time-series-with-Konrad/blob/main/m{NN}-{slug}.ipynb

## {Building block 1}
{Motivate from prior failure → one-line def + a.k.a. → equation → "in plain
English" → property bullets → knob extremes (High/Low) → picture + reading.}

## {Building block 2}
{Same template. Each rung fixes what the last one couldn't.}

## {…more rungs as needed…}

## {Building / applying a model}
{Dataset one-liner → EDA (decomposition, plots) → formal test (ADF etc.) →
order selection (ACF/PACF + AIC search) → chronological split → fit →
diagnostics → error dict → forecast-plot interpretation tying failure to an
assumption.}

## Closing time
{Recap the ladder as one evolving idea → honest throughline / no-free-lunch →
optional next-episode teaser.}

## Key Takeaways   ← (optional; use for long/multi-method posts)
1. **{Principle}:** {one sentence}.
2. …

United States of Banan is a reader-supported publication. To receive new posts
and support my work, consider becoming a free or paid subscriber.

[ ✓ Subscribed ]
```

---

## 14. Pre-publish checklist

- [ ] Title is `Time Series with Konrad: episode N`; subtitle set.
- [ ] Hook uses one opener pattern and states the core principle up front.
- [ ] 📓 notebook link present, correct `mNN-slug`.
- [ ] Methods are ordered as a ladder; each rung is motivated by the previous one's
      failure.
- [ ] Every equation is immediately unpacked in plain English.
- [ ] Every tuning parameter shown with High/Low contrast.
- [ ] Standing notation reused consistently (α/β/γ, S_t/b_t/c_t, φ/θ/ε, etc.).
- [ ] Real dataset named with units/source; workflow shown (EDA → test → order → split →
      fit → diagnostics → metrics).
- [ ] Error metrics reported as a literal dict; results reported **honestly**, including
      losses.
- [ ] Every figure has a one-paragraph reading.
- [ ] Advantages/Disadvantages (or Strengths/Limitations) blocks where methods are
      compared.
- [ ] At least one aside/joke/emoticon, but sparingly (~1 per section).
- [ ] Cross-links to earlier episodes where concepts recur.
- [ ] `Closing time` recaps the ladder + throughline; optional next-episode teaser.
- [ ] Subscribe footer pasted verbatim.
- [ ] Header/footer art in the series' portrait-meets-dataviz style.
```
