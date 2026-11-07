# Independent Post-Outline Critique and Dispositions

Reviewer: independent `research` subagent (`tails_critic`, conversation `0557d31c-eacb-42ce-ad36-5080d30ebab4`)
Reviewed artifact: [post-plan.md](post-plan.md) cross-checked against [primary-comparison.json](primary-comparison.json), [asset-manifest.json](asset-manifest.json), all retained CSV tables, `runs/20261107T155047630280+0100/run.json`, [fresh-kernel-validation.json](fresh-kernel-validation.json), and rendered PNGs.

## Overall Verdict

**Clean Pass (`0 P0`, `0 P1`, `1 P2`, `4 P3`)** — All numerical values, signs, confidence intervals, p-values, counts, runtimes, heading parity, 13-asset sequencing, and statistical claim boundaries verified cleanly against retained evidence.

## Findings and Dispositions

1. **`[P2]` Align Graph 08's step-episode years with `graphs/graph_08_0711.csv` and `graph_08_0711.png`**
   - *Finding*: In `graph_08_0711.csv`, the initial calm upward drift peaks in August `2014` (`+1.06`), the four largest single-day downward steps occur in autumn `2014` (`Sep–Nov 2014`) continuing into early `2015`, followed by a second upward drift across `2017–2019` (`-0.22` to `+0.58`) before sharp downward steps in `2020` and `2022`.
   - *Disposition*: **Accepted and incorporated** in both [post-plan.md](post-plan.md) and [post.md](post.md).

2. **`[P3]` Interior grid start date (`2003-01-17` to `2021-12-08`)**
   - *Finding*: `u = 0.15` maps to `2003-01-17` (early/January 2003, not spring 2003) and `u = 0.85` maps to `2021-12-08` (late/December 2021).
   - *Disposition*: **Accepted and incorporated** in [post-plan.md](post-plan.md) and [post.md](post.md).

3. **`[P3]` Harmonize `-0.0000015` (`-0.015e-4`) vs Table 11's two-decimal `-0.01` (`e-4`) rounding on EUR Rolling-2,500**
   - *Finding*: Mentioning both `-0.0000015` and `-0.01e-4` without noting two-decimal table display could look like a rounding discrepancy.
   - *Disposition*: **Accepted and incorporated** in [post-plan.md](post-plan.md) and [post.md](post.md), explicitly noting `-0.015 × 10⁻⁴` (displayed as `-0.01` at two decimals in Table 11).

4. **`[P3]` Keep EUR/USD's temporal pseudo-holdout status explicit in opening prose**
   - *Finding*: Because `EUR/USD` appeared in the earlier exploratory study, prose should match Table 01 row 7 by noting that `EUR/USD` serves as a chronological continuity / temporal pseudo-out-of-sample benchmark alongside previously uninspected `GBP` and `JPY`.
   - *Disposition*: **Accepted and incorporated** in [post-plan.md](post-plan.md) and [post.md](post.md).

5. **`[P3]` Title string parity and reader-facing bookkeeping hygiene**
   - *Finding*: Align post title `# Trick of the tail: testing and forecasting currency extremes` with the notebook title cell and keep internal JSON/run-directory filenames out of reader-facing `post.md` prose (delegating reproducibility manifest paths to [validation.md](validation.md)).
   - *Disposition*: **Accepted and incorporated** in [post-plan.md](post-plan.md) and [post.md](post.md).

