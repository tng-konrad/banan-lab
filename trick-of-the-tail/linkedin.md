Draft — article publication URL: [add after publication].

When a rolling tail-risk chart spikes during a currency sell-off, did the tail's shape actually steepen, or did volatility clustering and local scale shifts do the talking?

In a follow-up Python notebook and article on Li and Peng's “Time-Varying Tails: Estimation, Inference, and Risk Measurement”, we put their Bartlett HAC tail-stability test and one-sided Weissman quantile forecaster to work on 27 years of daily Federal Reserve H.10 FX data (EUR/USD, GBP/USD, and inverted JPY/USD, 1999–2025; 6,769 signed log losses per currency).

Separating retrospective inference from chronological prediction keeps the story honest:

* **Tail-stability inference (9-slot Holm family across 3 currencies × raw, GARCH-t, and GJR-GARCH-t targets):** **0 of 9** tests reject constant tail shape at nominal familywise alpha = 0.05 under 5,000-draw pivotal Pareto calibration. Two unadjusted GBP signals (raw p = 0.0328; GJR |z| p = 0.0200) do not survive multiplicity adjustment (Holm p = 0.2623 and 0.1800), and EUR stays quiet across six (m, h, J) sensitivity perturbations.
* **Scale adaptation dominates real-time risk forecasting (4,253 daily 1% VaR forecasts, 2009–2025):** Updating the local intermediate threshold daily while holding 2008's tail index fixed (Arm B vs static Arm A) cuts EUR pinball loss by −0.006673 (95% block-bootstrap CI [−0.008087, −0.005061]).
* **Updating tail shape daily is much smaller and primary-inconclusive:** Holding the daily kernel threshold identical, re-estimating gamma_t every day (Arm C vs Arm B) changes EUR expanding-window 1% pinball loss by −0.000224, with a paired L = 20 moving-block 95% interval [−0.000678, +0.000204] that straddles zero (and slightly favors constant shape on JPY).

All empirical stability p-values are framed explicitly as nominal paper-calibrated pivotal quantities because this real-data study omits synthetic dependent-null Monte Carlo screening.

Article: [publication URL pending]
Companion notebook: [local draft](trick_of_the_tail.ipynb)

#ExtremeValueTheory #TimeSeries #RiskManagement #Econometrics #Python

