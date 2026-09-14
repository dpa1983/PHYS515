"""
week1_lab_bayes_detector.py

Week 1 Lab — Bayes' Theorem for a Simulated Detector
Course: Data Analysis Methods for Physics and Astronomy

This script reproduces and extends the Week 1 worked example: a detector
("search pipeline") flags a "trigger" when its statistic crosses some
threshold. Given a true-positive rate, a false-alarm rate, and a prior
probability that any given time window contains a real signal, we:

  1. Compute the analytic answer via Bayes' theorem.
  2. Verify it against a direct Monte Carlo simulation.
  3. Study how sensitive the answer is to the false-alarm rate and the
     prior (the point of the lab, per the Week 1 slides).
  4. Show Monte Carlo convergence toward the analytic answer.

All inputs and outputs are saved to disk (CSV/JSON) alongside the plots,
so every number in this write-up is reproducible from this one file.

Run:
    python week1_lab_bayes_detector.py
Requires:
    numpy, pandas, matplotlib
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# =======================================================================
# 0. Style (kept consistent with the Week 1 slide deck)
# =======================================================================
NAVY = "#1E2761"
NAVY_DARK = "#141B47"
ICEBLUE_DARK = "#9FB8E8"
GOLD = "#E9B44C"
GRAY = "#6B7280"

plt.rcParams.update({
    "font.size": 12,
    "axes.edgecolor": GRAY,
    "axes.labelcolor": NAVY_DARK,
    "xtick.color": GRAY,
    "ytick.color": GRAY,
})

RNG_SEED = 42

# =======================================================================
# 1. INPUTS — the parameters of the worked example (Week 1 slides 14-15)
# =======================================================================
INPUTS = {
    "p_trigger_given_signal": 0.95,   # true-positive rate: P(trigger | real signal)
    "p_trigger_given_noise": 0.001,   # false-alarm rate:   P(trigger | noise only)
    "p_signal": 1e-4,                 # prior:              P(real signal present)
    "n_trials": 2_000_000,            # Monte Carlo sample size for the baseline run
}

print("=" * 70)
print("INPUTS")
print("=" * 70)
for k, v in INPUTS.items():
    print(f"  {k:28s} = {v}")


# =======================================================================
# 2. Analytic answer via Bayes' theorem
# =======================================================================
def analytic_posterior(p_tp, p_fa, p_signal):
    """
    P(signal | trigger) via Bayes' theorem, with the denominator expanded
    using the Law of Total Probability:

        P(trigger) = P(trigger|signal) P(signal) + P(trigger|noise) P(noise)
        P(signal | trigger) = P(trigger|signal) P(signal) / P(trigger)
    """
    p_noise = 1 - p_signal
    p_trigger = p_tp * p_signal + p_fa * p_noise
    p_signal_given_trigger = (p_tp * p_signal) / p_trigger
    return p_signal_given_trigger, p_trigger


p_signal_given_trigger_analytic, p_trigger_analytic = analytic_posterior(
    INPUTS["p_trigger_given_signal"], INPUTS["p_trigger_given_noise"], INPUTS["p_signal"]
)

print("\n" + "=" * 70)
print("ANALYTIC RESULT (Bayes' theorem)")
print("=" * 70)
print(f"  P(trigger)            = {p_trigger_analytic:.6f}")
print(f"  P(signal | trigger)   = {p_signal_given_trigger_analytic:.6f}"
      f"  ({p_signal_given_trigger_analytic*100:.3f}%)")


# =======================================================================
# 3. Monte Carlo simulation
# =======================================================================
def simulate_detector(p_tp, p_fa, p_signal, n_trials, rng):
    """
    Simulate n_trials independent time windows. In each window a real
    signal is present with probability p_signal; the detector then fires
    a trigger with probability p_tp (if signal present) or p_fa (if not).
    Returns a DataFrame with one row per trial: signal_present, triggered.
    """
    signal_present = rng.random(n_trials) < p_signal
    trigger_prob = np.where(signal_present, p_tp, p_fa)
    triggered = rng.random(n_trials) < trigger_prob
    return pd.DataFrame({"signal_present": signal_present, "triggered": triggered})


rng = np.random.default_rng(RNG_SEED)
trials = simulate_detector(
    INPUTS["p_trigger_given_signal"], INPUTS["p_trigger_given_noise"],
    INPUTS["p_signal"], INPUTS["n_trials"], rng,
)

n_triggers = trials["triggered"].sum()
n_true_positive_triggers = (trials["triggered"] & trials["signal_present"]).sum()
n_false_alarm_triggers = (trials["triggered"] & ~trials["signal_present"]).sum()
p_signal_given_trigger_empirical = n_true_positive_triggers / n_triggers

print("\n" + "=" * 70)
print("MONTE CARLO RESULT")
print("=" * 70)
print(f"  n_trials                        = {INPUTS['n_trials']:,}")
print(f"  total triggers observed         = {n_triggers:,}")
print(f"    - real-signal triggers        = {n_true_positive_triggers:,}")
print(f"    - false-alarm triggers        = {n_false_alarm_triggers:,}")
print(f"  empirical P(signal | trigger)   = {p_signal_given_trigger_empirical:.6f}"
      f"  ({p_signal_given_trigger_empirical*100:.3f}%)")
print(f"  analytic  P(signal | trigger)   = {p_signal_given_trigger_analytic:.6f}"
      f"  ({p_signal_given_trigger_analytic*100:.3f}%)")
print(f"  absolute difference             = "
      f"{abs(p_signal_given_trigger_empirical - p_signal_given_trigger_analytic):.6f}")


# =======================================================================
# 4. OUTPUT — baseline results summary (saved to disk)
# =======================================================================
baseline_results = {
    **INPUTS,
    "p_trigger_analytic": p_trigger_analytic,
    "p_signal_given_trigger_analytic": p_signal_given_trigger_analytic,
    "n_triggers_observed": int(n_triggers),
    "n_true_positive_triggers": int(n_true_positive_triggers),
    "n_false_alarm_triggers": int(n_false_alarm_triggers),
    "p_signal_given_trigger_empirical": float(p_signal_given_trigger_empirical),
}
with open("lab01_baseline_results.json", "w") as f:
    json.dump(baseline_results, f, indent=2)
print("\nSaved baseline_results -> lab01_baseline_results.json")

trigger_counts_df = pd.DataFrame({
    "category": ["Real-signal trigger (true positive)", "False-alarm trigger"],
    "count": [n_true_positive_triggers, n_false_alarm_triggers],
})
trigger_counts_df.to_csv("lab01_trigger_counts.csv", index=False)
print("Saved trigger counts       -> lab01_trigger_counts.csv")


# =======================================================================
# 5. Sensitivity analysis: vary the prior P(signal)
# =======================================================================
priors = np.logspace(-6, -1, 60)  # from 1e-6 to 1e-1
posterior_vs_prior = [
    analytic_posterior(INPUTS["p_trigger_given_signal"], INPUTS["p_trigger_given_noise"], p)[0]
    for p in priors
]
sensitivity_prior_df = pd.DataFrame({
    "prior_p_signal": priors,
    "posterior_p_signal_given_trigger": posterior_vs_prior,
})
sensitivity_prior_df.to_csv("lab01_sensitivity_prior.csv", index=False)
print("Saved prior sensitivity    -> lab01_sensitivity_prior.csv")

# =======================================================================
# 6. Sensitivity analysis: vary the false-alarm rate
# =======================================================================
false_alarm_rates = np.logspace(-5, -1, 60)
posterior_vs_fa = [
    analytic_posterior(INPUTS["p_trigger_given_signal"], fa, INPUTS["p_signal"])[0]
    for fa in false_alarm_rates
]
sensitivity_fa_df = pd.DataFrame({
    "false_alarm_rate": false_alarm_rates,
    "posterior_p_signal_given_trigger": posterior_vs_fa,
})
sensitivity_fa_df.to_csv("lab01_sensitivity_false_alarm.csv", index=False)
print("Saved false-alarm sensitivity -> lab01_sensitivity_false_alarm.csv")


# =======================================================================
# 7. Monte Carlo convergence: empirical estimate vs. number of trials
# =======================================================================
def running_posterior_estimate(trials_df):
    """Cumulative empirical P(signal|trigger) as a function of sample size."""
    triggered = trials_df["triggered"].to_numpy()
    true_pos = (trials_df["triggered"] & trials_df["signal_present"]).to_numpy()
    cum_triggers = np.cumsum(triggered)
    cum_true_pos = np.cumsum(true_pos)
    with np.errstate(invalid="ignore", divide="ignore"):
        running_est = np.where(cum_triggers > 0, cum_true_pos / cum_triggers, np.nan)
    return running_est


# Use a checkpoint schedule (log-spaced) rather than every single trial,
# both for plotting clarity and to keep the output file small.
checkpoints = np.unique(np.logspace(2, np.log10(INPUTS["n_trials"]), 200).astype(int))
running_est_full = running_posterior_estimate(trials)
convergence_df = pd.DataFrame({
    "n_trials": checkpoints,
    "empirical_p_signal_given_trigger": running_est_full[checkpoints - 1],
})
convergence_df["analytic_p_signal_given_trigger"] = p_signal_given_trigger_analytic
convergence_df.to_csv("lab01_convergence.csv", index=False)
print("Saved MC convergence data  -> lab01_convergence.csv")


# =======================================================================
# 8. PLOTS
# =======================================================================

# --- Plot 1: trigger composition bar chart ---
fig, ax = plt.subplots(figsize=(6.5, 5))
bars = ax.bar(
    trigger_counts_df["category"], trigger_counts_df["count"],
    color=[NAVY, GOLD], edgecolor=NAVY_DARK, width=0.55,
)
for bar, count in zip(bars, trigger_counts_df["count"]):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() * 1.01,
            f"{count:,}\n({count / n_triggers:.1%} of triggers)",
            ha="center", va="bottom", fontsize=11)
ax.set_ylabel("Number of triggers")
ax.set_title(f"Composition of {n_triggers:,} observed triggers\n"
             f"(n_trials = {INPUTS['n_trials']:,}, prior = {INPUTS['p_signal']:.0e})")
ax.spines[["top", "right"]].set_visible(False)
plt.xticks(rotation=8)
fig.tight_layout()
fig.savefig("lab01_fig1_trigger_composition.png", dpi=150)
plt.close(fig)
print("\nSaved plot -> lab01_fig1_trigger_composition.png")

# --- Plot 2: posterior vs prior (the base-rate sensitivity) ---
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(priors, posterior_vs_prior, color=NAVY, linewidth=2.5)
ax.axvline(INPUTS["p_signal"], color=GRAY, linestyle="--", linewidth=1.3)
ax.plot(INPUTS["p_signal"], p_signal_given_trigger_analytic, "o", color=GOLD,
        markersize=10, zorder=5)
ax.annotate(
    f"baseline: prior={INPUTS['p_signal']:.0e}\nP(signal|trigger)={p_signal_given_trigger_analytic:.1%}",
    xy=(INPUTS["p_signal"], p_signal_given_trigger_analytic),
    xytext=(3e-6, 0.6), fontsize=10.5, color=NAVY_DARK,
    arrowprops=dict(arrowstyle="->", color=GRAY),
)
ax.set_xscale("log")
ax.set_xlabel("Prior probability of a real signal, P(signal)")
ax.set_ylabel("P(signal | trigger)")
ax.set_title("Sensitivity to the prior (base-rate effect)\n"
             f"[fixed: true-positive rate={INPUTS['p_trigger_given_signal']}, "
             f"false-alarm rate={INPUTS['p_trigger_given_noise']}]")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("lab01_fig2_sensitivity_prior.png", dpi=150)
plt.close(fig)
print("Saved plot -> lab01_fig2_sensitivity_prior.png")

# --- Plot 3: posterior vs false-alarm rate ---
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(false_alarm_rates, posterior_vs_fa, color=NAVY, linewidth=2.5)
ax.axvline(INPUTS["p_trigger_given_noise"], color=GRAY, linestyle="--", linewidth=1.3)
ax.plot(INPUTS["p_trigger_given_noise"], p_signal_given_trigger_analytic, "o", color=GOLD,
        markersize=10, zorder=5)
ax.annotate(
    f"baseline: false-alarm rate={INPUTS['p_trigger_given_noise']:.0e}\n"
    f"P(signal|trigger)={p_signal_given_trigger_analytic:.1%}",
    xy=(INPUTS["p_trigger_given_noise"], p_signal_given_trigger_analytic),
    xytext=(3e-5, 0.6), fontsize=10.5, color=NAVY_DARK,
    arrowprops=dict(arrowstyle="->", color=GRAY),
)
ax.set_xscale("log")
ax.set_xlabel("False-alarm rate, P(trigger | noise only)")
ax.set_ylabel("P(signal | trigger)")
ax.set_title("Sensitivity to the false-alarm rate\n"
             f"[fixed: true-positive rate={INPUTS['p_trigger_given_signal']}, "
             f"prior={INPUTS['p_signal']:.0e}]")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("lab01_fig3_sensitivity_false_alarm.png", dpi=150)
plt.close(fig)
print("Saved plot -> lab01_fig3_sensitivity_false_alarm.png")

# --- Plot 4: Monte Carlo convergence toward the analytic answer ---
fig, ax = plt.subplots(figsize=(7.5, 5))
ax.plot(convergence_df["n_trials"], convergence_df["empirical_p_signal_given_trigger"],
        color=NAVY, linewidth=1.8, label="Empirical (Monte Carlo)")
ax.axhline(p_signal_given_trigger_analytic, color=GOLD, linewidth=2.2, linestyle="--",
           label="Analytic (Bayes' theorem)")
ax.set_xscale("log")
ax.set_xlabel("Number of simulated trials")
ax.set_ylabel("P(signal | trigger)")
ax.set_title("Monte Carlo convergence to the analytic Bayes' theorem answer")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("lab01_fig4_convergence.png", dpi=150)
plt.close(fig)
print("Saved plot -> lab01_fig4_convergence.png")

print("\nAll done. Outputs written to the current working directory.")


# =======================================================================
# 9. DISCUSSION — connecting this toy example to real LIGO/Virgo numbers
# =======================================================================
"""
The structure of this lab is exactly the structure behind a real
gravitational-wave search pipeline (and the reason the Gravity Spy
glitch-classification project exists): a matched-filter search produces a
"trigger" when its detection statistic crosses threshold, and the
question "is this trigger a real signal or noise?" is answered with
exactly the same Bayes' theorem machinery used above.

The NUMBERS in the toy example (true-positive rate 0.95, false-alarm
rate 0.001, prior 1e-4), however, are illustrative teaching values, not
real pipeline figures. Below, real published false-alarm rates (FARs)
for several actual LIGO/Virgo events are tabulated directly from their
discovery papers, for comparison.

IMPORTANT UNITS CAVEAT: a real search's FAR is reported as a *rate*
(events per year of equivalent background data, estimated empirically
via detector time-slides), not a dimensionless per-trial probability
like this toy lab's p_fa. The two are not directly interchangeable
without additional assumptions about how many independent "trials"
occur per year -- which real analyses deliberately avoid needing, by
working with time-slides instead of a discrete trial count. The
comparison below is therefore shown with units kept explicit throughout,
and is meant to convey *scale* (how many orders of magnitude apart these
are), not a literal apples-to-apples probability comparison.
"""

print("\n" + "=" * 70)
print("DISCUSSION: REAL LIGO/VIRGO PIPELINE NUMBERS (for comparison)")
print("=" * 70)

# Published false-alarm rates, pulled directly from the discovery papers.
# Sources:
#   GW150914:  Abbott et al. 2016, PRL 116, 061102 (arXiv:1602.03839)
#   GW170814:  Abbott et al. 2017, PRL 119, 141101 (arXiv:1709.09660)
#   GW190814:  Abbott et al. 2020, ApJL 896, L44   (arXiv:2006.12611)
#   Real-event rate: LIGO/Virgo/KAGRA O3b (GWTC-3) catalog release notes
real_events = pd.DataFrame([
    {
        "event": "GW150914",
        "pipeline": "matched-filter (combined)",
        "far_value": 1, "far_years": 203_000,
        "far_per_year": 1 / 203_000,
        "significance_sigma": 5.1,
        "source": "Abbott et al. 2016, PRL 116, 061102 (arXiv:1602.03839)",
    },
    {
        "event": "GW170814",
        "pipeline": "matched-filter, pipeline A",
        "far_value": 1, "far_years": 140_000,
        "far_per_year": 1 / 140_000,
        "significance_sigma": np.nan,
        "source": "Abbott et al. 2017, PRL 119, 141101 (arXiv:1709.09660)",
    },
    {
        "event": "GW170814",
        "pipeline": "matched-filter, pipeline B",
        "far_value": 1, "far_years": 27_000,
        "far_per_year": 1 / 27_000,
        "significance_sigma": np.nan,
        "source": "Abbott et al. 2017, PRL 119, 141101 (arXiv:1709.09660)",
    },
    {
        "event": "GW170814",
        "pipeline": "coherent unmodeled search (3-detector)",
        "far_value": 1, "far_years": 5_900,
        "far_per_year": 1 / 5_900,
        "significance_sigma": np.nan,
        "source": "Abbott et al. 2017, PRL 119, 141101 (arXiv:1709.09660)",
    },
    {
        "event": "GW190814",
        "pipeline": "GstLAL",
        "far_value": 1, "far_years": 1_300,
        "far_per_year": 1 / 1_300,
        "significance_sigma": np.nan,
        "source": "Abbott et al. 2020, ApJL 896, L44 (arXiv:2006.12611)",
    },
    {
        "event": "GW190814",
        "pipeline": "extended PyCBC",
        "far_value": 1, "far_years": 8.1,
        "far_per_year": 1 / 8.1,
        "significance_sigma": np.nan,
        "source": "Abbott et al. 2020, ApJL 896, L44 (arXiv:2006.12611)",
    },
])
real_events.to_csv("lab01_real_ligo_comparison.csv", index=False)
print("\nPublished false-alarm rates (from discovery papers):")
for _, row in real_events.iterrows():
    sig = f", significance {row['significance_sigma']}\u03c3" if not np.isnan(row["significance_sigma"]) else ""
    print(f"  {row['event']:10s} [{row['pipeline']:38s}]  "
          f"FAR < 1 per {row['far_years']:>8,.1f} yr  "
          f"(\u2248 {row['far_per_year']:.2e} yr\u207b\u00b9){sig}")

print(f"\n  For comparison, this lab's toy false-alarm rate: "
      f"{INPUTS['p_trigger_given_noise']} (per trial, NOT per year)")
print("  Real pipelines achieve false-alarm rates many orders of magnitude")
print("  smaller than this toy value -- precisely BECAUSE real merger events")
print("  are also extremely rare, so a much lower false-alarm rate is needed")
print("  before a single trigger can be trusted (cf. the base-rate effect")
print("  explored in Section 5/6 above).")

# Real-world context for the PRIOR side of the comparison: how often do
# real events actually occur? From the GWTC-3 catalog release: 90 confident
# events accumulated from the start of O1 (Sept 2015) through the end of
# O3b (March 2020); within O3 alone (~350 days, Apr 2019-Mar 2020, with at
# least one detector observing ~96% of the time), 79 confident events
# (44 in O3a + 35 in O3b) were reported.
o3_days = 350
o3_events = 44 + 35
approx_o3_rate_per_day = o3_events / o3_days
print(f"\n  Real detection RATE (not FAR) for context: during O3 "
      f"({o3_days} days), {o3_events} confident events were reported")
print(f"  (LIGO/Virgo/KAGRA GWTC-3 catalog) -- roughly one confident "
      f"detection every {1/approx_o3_rate_per_day:.1f} days on average.")
print("  This is only a rough average (real duty cycle and sensitivity")
print("  varied through the run) but it shows that, while real merger")
print("  events are rare on a per-second basis, they are far from")
print("  vanishingly rare over a year of good observing -- which is why")
print("  the false-alarm rate has to be pushed down so aggressively.")

print("\n  Where Gravity Spy fits in: it doesn't change the Bayes' theorem")
print("  logic at all -- it improves the LIKELIHOOD term. By classifying")
print("  glitches into known categories (with machine learning and citizen")
print("  scientists), Gravity Spy lets a pipeline recognize and veto known")
print("  glitch morphologies, which lowers the effective false-alarm rate")
print("  P(trigger | noise) used in exactly the same formula as Section 2.")

print("\nSaved real-pipeline comparison data -> lab01_real_ligo_comparison.csv")


# =======================================================================
# 10. PLOT 5 — real published FARs vs. the toy lab's false-alarm rate
# =======================================================================
fig, ax = plt.subplots(figsize=(9, 5.5))
labels = [f"{r.event}\n({r.pipeline})" for r in real_events.itertuples()]
values = real_events["far_per_year"].to_numpy()
colors = [GOLD if "GW190814" in lbl else NAVY for lbl in labels]
bars = ax.barh(labels, values, color=colors, edgecolor=NAVY_DARK)
ax.set_xscale("log")
ax.axvline(INPUTS["p_trigger_given_noise"], color="crimson", linestyle="--", linewidth=2,
           label=f"toy lab false-alarm rate = {INPUTS['p_trigger_given_noise']} (per TRIAL, different units)")
ax.set_xlabel("False-alarm rate (yr$^{-1}$) — log scale")
ax.set_title("Published LIGO/Virgo false-alarm rates vs. this lab's toy value\n"
             "(different units — shown together only to illustrate scale)")
ax.legend(frameon=False, loc="lower right", fontsize=9.5)
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout()
fig.savefig("lab01_fig5_real_ligo_comparison.png", dpi=150)
plt.close(fig)
print("Saved plot -> lab01_fig5_real_ligo_comparison.png")

print("\nDiscussion section complete.")
