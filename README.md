# E-Commerce A/B Test Analysis — Landing Page Conversion Experiment

A rigorous statistical analysis of an A/B test where an e-commerce company tested a **new landing page design** against their existing page to determine its impact on **purchase conversion rate**.

This project goes beyond "p < 0.05" — it includes power analysis, effect size estimation, revenue impact modeling, segmented analysis with multiple comparison corrections, and a Bayesian comparison.

## Business Context

An e-commerce company redesigned their landing page with the goal of increasing purchase conversions. Before rolling out the new design to all users, they ran a controlled A/B test:

- **Control group**: Saw the existing (old) landing page
- **Treatment group**: Saw the redesigned (new) landing page
- **Primary metric**: Purchase conversion rate
- **Decision**: Launch the new page only if there is both a statistically significant AND practically meaningful improvement

## Key Findings

*(Run the analysis notebook to generate specific results from your chosen dataset)*

The analysis covers:
- Whether the new page significantly improved conversion rates
- The practical size of the effect (if any)
- Estimated monthly revenue impact
- Whether the result holds across user segments
- Both frequentist and Bayesian perspectives on the evidence

## Analysis Structure

The notebook follows the structure of a report you'd present to a product team:

1. **Data Overview & Sanity Checks** — Sample ratio mismatch, duplicates, data quality
2. **Descriptive Statistics** — Conversion rates with confidence intervals, temporal trends
3. **Frequentist Hypothesis Testing** — Z-test (manually derived + scipy validation)
4. **Power Analysis** — Was the sample large enough to detect the observed effect?
5. **Effect Size & Revenue Impact** — Cohen's h, practical significance, dollar estimates
6. **Segmented Analysis** — Device, time-of-day, new vs returning users + Bonferroni correction
7. **Bayesian Analysis** — Beta-Binomial model, posterior distributions, P(B > A)
8. **Recommendation** — Ship or don't ship, with clear reasoning

## Quick Start

### 1. Clone and Install

```bash
git clone https://github.com/VarunLautkar/ecommerce-ab-test-analysis.git
cd ecommerce-ab-test-analysis
pip install -r requirements.txt
```

### 2. Get the Dataset

Download one of these from Kaggle and place in `data/`:

- [E-Commerce A/B Testing 2022](https://www.kaggle.com/datasets/putdejudomthai/ecommerce-ab-testing-2022-dataset1)
- [A/B Testing Dataset](https://www.kaggle.com/datasets/amirmotefaker/ab-testing-dataset)

### 3. Run the Analysis

```bash
python ab_test_analysis.py
```

This generates:
- All statistical results printed to console
- Visualization plots saved to `figures/`
- Summary metrics saved to `results/analysis_results.json`

## Project Structure

```
ecommerce-ab-test-analysis/
├── README.md
├── requirements.txt
├── ab_test_analysis.py          # Full analysis script (report-style)
├── data/
│   └── ab_test_data.csv         # Dataset (download separately)
├── figures/                     # Generated plots
│   ├── conversion_rates_ci.png
│   ├── conversion_over_time.png
│   ├── power_analysis.png
│   ├── bayesian_posteriors.png
│   └── segment_analysis.png
└── results/
    └── analysis_results.json    # All computed metrics
```

## Statistical Methods Used

- **Two-proportion Z-test** — for testing the difference in conversion rates
- **Confidence intervals** — 95% CI for the treatment effect using the normal approximation
- **Cohen's h** — standardized effect size for proportions
- **Power analysis** — using the observed effect to determine adequacy of sample size
- **Bonferroni correction** — for controlling family-wise error rate in segmented analysis
- **Bayesian Beta-Binomial model** — posterior distributions and direct probability comparison

## Key Design Decisions

- **Two-sided test**: We test for *any* difference, not just improvement — the new page could be worse.
- **Piecewise RUL cap**: Practical significance threshold set at 0.5% absolute conversion lift (below this, engineering cost of shipping outweighs gains).
- **Multiple comparison correction**: With 4+ segments, uncorrected p-values inflate false positive risk — Bonferroni is conservative but honest.
- **Revenue modeling**: Converts statistical effect to dollar impact using assumed AOV — makes results actionable for stakeholders.

## What I Learned

- Statistical significance ≠ practical significance. A tiny lift can be "significant" with enough data.
- Power analysis should happen *before* the test, not after — but retroactive power analysis still reveals whether results are trustworthy.
- Bayesian analysis provides more intuitive answers ("82% chance treatment is better") vs frequentist ("we reject the null at α=0.05").
- Segmented analysis can reveal that an overall null result hides real effects in subgroups (Simpson's Paradox).
