"""
E-Commerce A/B Test Analysis — Landing Page Conversion Experiment
=================================================================

A rigorous statistical analysis of whether a redesigned landing page
improves purchase conversion rates for an e-commerce company.

This script performs:
  1. Data loading, validation, and sanity checks
  2. Descriptive statistics with confidence intervals
  3. Frequentist hypothesis testing (manual + scipy)
  4. Statistical power analysis
  5. Effect size and revenue impact estimation
  6. Segmented analysis with multiple comparison correction
  7. Bayesian Beta-Binomial analysis
  8. Final recommendation

Run:
    python ab_test_analysis.py

Output:
    - Console: Full analysis report
    - figures/: All visualization plots
    - results/analysis_results.json: Computed metrics
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.stats.power import NormalIndPower

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
FIGURES_DIR = Path("figures")
RESULTS_DIR = Path("results")
DATA_DIR = Path("data")
FIGURES_DIR.mkdir(exist_ok=True)
RESULTS_DIR.mkdir(exist_ok=True)

# Business assumptions
MONTHLY_VISITORS = 500_000
AVG_ORDER_VALUE = 75.0  # USD
SIGNIFICANCE_LEVEL = 0.05
PRACTICAL_THRESHOLD = 0.005  # 0.5% absolute lift required to be "worth it"

# Plot styling
plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
})
COLORS = {"control": "#3498db", "treatment": "#e74c3c", "neutral": "#95a5a6"}


# ===========================================================================
# SECTION 1: DATA LOADING & VALIDATION
# ===========================================================================

def load_data() -> pd.DataFrame:
    """
    Load the A/B test dataset.

    Attempts to load from data/ directory. If not found, generates a
    realistic synthetic dataset for demonstration purposes.
    """
    csv_files = list(DATA_DIR.glob("*.csv"))

    if csv_files:
        print(f"Loading dataset: {csv_files[0].name}")
        df = pd.read_csv(csv_files[0])
        df = standardize_columns(df)
    else:
        print("No dataset found in data/ — generating synthetic data for demo")
        df = generate_synthetic_data()

    return df


def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize column names to a consistent format.

    Different Kaggle datasets use different naming conventions.
    We normalize to: user_id, group, converted, device, timestamp.
    """
    df.columns = df.columns.str.lower().str.strip().str.replace(" ", "_")

    # Common renames across different dataset formats
    rename_map = {}
    for col in df.columns:
        if "user" in col and "id" in col:
            rename_map[col] = "user_id"
        elif col in ("group", "variant", "ab_group", "test_group", "landing_page"):
            rename_map[col] = "group"
        elif col in ("converted", "conversion", "purchased", "purchase"):
            rename_map[col] = "converted"
        elif col in ("device", "device_type", "platform"):
            rename_map[col] = "device"
        elif col in ("timestamp", "date", "time", "visit_date"):
            rename_map[col] = "timestamp"

    df = df.rename(columns=rename_map)

    # Standardize group labels
    if "group" in df.columns:
        group_map = {
            "control": "control", "old_page": "control", "A": "control",
            "ad": "control", "0": "control", 0: "control",
            "treatment": "treatment", "new_page": "treatment", "B": "treatment",
            "psa": "treatment", "1": "treatment", 1: "treatment",
        }
        df["group"] = df["group"].map(group_map).fillna(df["group"])

    # Parse timestamp column to datetime if present (CSV loads it as string)
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
        except Exception:
            pass

    return df


def generate_synthetic_data(
    n_control: int = 150_000, n_treatment: int = 150_000, seed: int = 42
) -> pd.DataFrame:
    """
    Generate a realistic synthetic A/B test dataset.

    The treatment has a small positive effect (0.3% lift), which creates
    an interesting analytical challenge: statistically significant with
    this sample size, but is it practically meaningful?
    """
    rng = np.random.default_rng(seed)

    # Base conversion rates
    control_rate = 0.120   # 12.0%
    treatment_rate = 0.123  # 12.3% — small but real lift

    # Generate users
    n_total = n_control + n_treatment
    groups = np.array(["control"] * n_control + ["treatment"] * n_treatment)

    # Device distribution (mobile converts lower)
    devices = rng.choice(
        ["desktop", "mobile", "tablet"],
        size=n_total,
        p=[0.45, 0.40, 0.15],
    )

    # Device-specific conversion modifiers
    device_modifier = np.where(devices == "desktop", 1.1,
                      np.where(devices == "mobile", 0.85, 1.0))

    # Time features — simulate 30 days
    days = rng.integers(0, 30, size=n_total)
    hours = rng.integers(0, 24, size=n_total)
    timestamps = pd.to_datetime("2024-01-01") + pd.to_timedelta(days, unit="D")

    # User type
    user_type = rng.choice(
        ["new", "returning"], size=n_total, p=[0.6, 0.4]
    )
    user_modifier = np.where(user_type == "returning", 1.15, 1.0)

    # Generate conversions with all modifiers
    base_rates = np.where(groups == "control", control_rate, treatment_rate)
    adjusted_rates = np.clip(base_rates * device_modifier * user_modifier, 0, 1)
    conversions = rng.binomial(1, adjusted_rates)

    df = pd.DataFrame({
        "user_id": range(1, n_total + 1),
        "group": groups,
        "converted": conversions,
        "device": devices,
        "timestamp": timestamps,
        "hour": hours,
        "user_type": user_type,
    })

    return df


def run_sanity_checks(df: pd.DataFrame) -> dict:
    """
    Perform pre-analysis sanity checks.

    These checks catch common data quality issues that would invalidate
    the experiment before we even get to hypothesis testing.
    """
    print("\n" + "=" * 60)
    print("SECTION 1: DATA OVERVIEW & SANITY CHECKS")
    print("=" * 60)

    results = {}

    # Basic info
    print(f"\nDataset shape: {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")

    # --- Check 1: Duplicate users ---
    if "user_id" in df.columns:
        n_duplicates = df["user_id"].duplicated().sum()
        print(f"\nDuplicate users: {n_duplicates}")
        if n_duplicates > 0:
            print("  ⚠ WARNING: Users appear in multiple rows. Deduplicating...")
            df = df.drop_duplicates(subset="user_id", keep="first")
        results["duplicate_users"] = int(n_duplicates)

    # --- Check 2: Sample Ratio Mismatch (SRM) ---
    group_counts = df["group"].value_counts()
    n_control = group_counts.get("control", 0)
    n_treatment = group_counts.get("treatment", 0)
    n_total = n_control + n_treatment

    print(f"\nGroup sizes:")
    print(f"  Control:   {n_control:>8,} ({n_control/n_total:.1%})")
    print(f"  Treatment: {n_treatment:>8,} ({n_treatment/n_total:.1%})")

    # Chi-square test for 50/50 split
    chi2_stat, srm_pvalue = stats.chisquare([n_control, n_treatment])
    results["srm_chi2"] = float(chi2_stat)
    results["srm_pvalue"] = float(srm_pvalue)

    if srm_pvalue < 0.01:
        print(f"  ⚠ SAMPLE RATIO MISMATCH DETECTED (χ²={chi2_stat:.2f}, p={srm_pvalue:.4f})")
        print("    The groups are not balanced — this could indicate a bug in randomization.")
    else:
        print(f"  ✓ No sample ratio mismatch (χ²={chi2_stat:.2f}, p={srm_pvalue:.4f})")

    # --- Check 3: Missing values ---
    missing = df.isnull().sum()
    if missing.sum() > 0:
        print(f"\nMissing values:\n{missing[missing > 0]}")
    else:
        print("\n✓ No missing values")

    # --- Check 4: Conversion value validation ---
    unique_conversions = df["converted"].unique()
    print(f"\nConversion values: {sorted(unique_conversions)}")
    assert set(unique_conversions).issubset({0, 1}), "Conversion should be binary (0/1)"

    results["n_control"] = int(n_control)
    results["n_treatment"] = int(n_treatment)

    return results


# ===========================================================================
# SECTION 2: DESCRIPTIVE STATISTICS & VISUALIZATION
# ===========================================================================

def descriptive_analysis(df: pd.DataFrame) -> dict:
    """
    Compute and visualize descriptive statistics for both groups.
    """
    print("\n" + "=" * 60)
    print("SECTION 2: DESCRIPTIVE STATISTICS")
    print("=" * 60)

    results = {}

    control = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    # Conversion rates
    cr_control = control["converted"].mean()
    cr_treatment = treatment["converted"].mean()
    absolute_diff = cr_treatment - cr_control
    relative_lift = (cr_treatment - cr_control) / cr_control * 100

    print(f"\nConversion Rates:")
    print(f"  Control:   {cr_control:.4f} ({cr_control:.2%})")
    print(f"  Treatment: {cr_treatment:.4f} ({cr_treatment:.2%})")
    print(f"  Absolute difference: {absolute_diff:+.4f} ({absolute_diff:+.2%})")
    print(f"  Relative lift: {relative_lift:+.2f}%")

    results["cr_control"] = float(cr_control)
    results["cr_treatment"] = float(cr_treatment)
    results["absolute_diff"] = float(absolute_diff)
    results["relative_lift_pct"] = float(relative_lift)

    # --- Plot 1: Conversion rates with 95% CI error bars ---
    fig, ax = plt.subplots(figsize=(8, 6))

    groups = ["Control\n(Old Page)", "Treatment\n(New Page)"]
    rates = [cr_control, cr_treatment]

    # Wilson confidence intervals (better than normal approx for proportions)
    ci_control = _wilson_ci(control["converted"].sum(), len(control))
    ci_treatment = _wilson_ci(treatment["converted"].sum(), len(treatment))

    errors = [
        [rates[0] - ci_control[0], rates[1] - ci_treatment[0]],
        [ci_control[1] - rates[0], ci_treatment[1] - rates[1]],
    ]

    bars = ax.bar(groups, rates, color=[COLORS["control"], COLORS["treatment"]],
                  width=0.5, edgecolor="black", linewidth=0.8)
    ax.errorbar(groups, rates, yerr=errors, fmt="none", color="black",
                capsize=8, capthick=2, linewidth=2)

    ax.set_ylabel("Conversion Rate")
    ax.set_title("Conversion Rate by Group (with 95% CI)")

    # Position labels ABOVE the upper CI bound to avoid overlapping error bars
    ci_tops = [ci_control[1], ci_treatment[1]]
    y_label_offset = 0.003  # fixed gap above error bar cap
    for bar, rate, ci_top in zip(bars, rates, ci_tops):
        ax.text(bar.get_x() + bar.get_width() / 2, ci_top + y_label_offset,
                f"{rate:.2%}", ha="center", va="bottom", fontweight="bold", fontsize=13)

    # Expand ylim to show labels above error bars
    y_min = min(rates) * 0.97
    y_max = max(ci_tops) + y_label_offset * 4
    ax.set_ylim(bottom=y_min, top=y_max)


    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "conversion_rates_ci.png", dpi=150)
    plt.close(fig)
    print("\n  [Saved: figures/conversion_rates_ci.png]")

    # --- Plot 2: Conversion over time (novelty effect check) ---
    if "timestamp" in df.columns:
        fig, ax = plt.subplots(figsize=(12, 5))

        daily = df.groupby([pd.Grouper(key="timestamp", freq="D"), "group"])["converted"].mean()
        daily = daily.reset_index()

        for group, color in [("control", COLORS["control"]), ("treatment", COLORS["treatment"])]:
            group_data = daily[daily["group"] == group]
            ax.plot(group_data["timestamp"], group_data["converted"],
                    marker="o", markersize=4, color=color, label=group.title(), linewidth=2)

        ax.set_xlabel("Date")
        ax.set_ylabel("Daily Conversion Rate")
        ax.set_title("Conversion Rate Over Time — Checking for Novelty Effects")
        ax.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "conversion_over_time.png", dpi=150)
        plt.close(fig)
        print("  [Saved: figures/conversion_over_time.png]")

    return results


def _wilson_ci(successes: int, n: int, confidence: float = 0.95) -> tuple:
    """Wilson score confidence interval for a proportion."""
    z = stats.norm.ppf(1 - (1 - confidence) / 2)
    p_hat = successes / n
    denominator = 1 + z**2 / n
    center = (p_hat + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) + z**2 / (4 * n)) / n) / denominator
    return (center - margin, center + margin)


# ===========================================================================
# SECTION 3: FREQUENTIST HYPOTHESIS TESTING
# ===========================================================================

def frequentist_test(df: pd.DataFrame) -> dict:
    """
    Perform a two-proportion Z-test, both manually and with scipy.

    Hypotheses:
        H₀: p_treatment = p_control   (no difference)
        H₁: p_treatment ≠ p_control   (two-sided)
    """
    print("\n" + "=" * 60)
    print("SECTION 3: FREQUENTIST HYPOTHESIS TESTING")
    print("=" * 60)

    results = {}

    control = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    n_c = len(control)
    n_t = len(treatment)
    x_c = control["converted"].sum()
    x_t = treatment["converted"].sum()
    p_c = x_c / n_c
    p_t = x_t / n_t

    # --- Manual calculation (showing the work) ---
    print("\n--- Manual Z-Test Derivation ---")

    # Pooled proportion under H₀
    p_pool = (x_c + x_t) / (n_c + n_t)
    print(f"\n  Pooled proportion (under H₀): {p_pool:.6f}")

    # Standard error of the difference
    se = np.sqrt(p_pool * (1 - p_pool) * (1/n_c + 1/n_t))
    print(f"  Standard error: {se:.6f}")

    # Z-statistic
    z_stat = (p_t - p_c) / se
    print(f"  Z-statistic: {z_stat:.4f}")

    # Two-sided p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))
    print(f"  P-value (two-sided): {p_value:.6f}")

    # 95% confidence interval for the difference
    se_diff = np.sqrt(p_c * (1 - p_c) / n_c + p_t * (1 - p_t) / n_t)
    z_crit = stats.norm.ppf(1 - SIGNIFICANCE_LEVEL / 2)
    ci_lower = (p_t - p_c) - z_crit * se_diff
    ci_upper = (p_t - p_c) + z_crit * se_diff
    print(f"\n  95% CI for difference: [{ci_lower:.6f}, {ci_upper:.6f}]")
    print(f"  (We are 95% confident the true difference in conversion")
    print(f"   rates lies between {ci_lower:.2%} and {ci_upper:.2%})")

    # --- Scipy validation ---
    print("\n--- Scipy Validation ---")
    z_scipy, p_scipy = proportions_ztest(
        count=[x_t, x_c], nobs=[n_t, n_c], alternative="two-sided"
    )
    print(f"  Z-statistic (scipy): {z_scipy:.4f}")
    print(f"  P-value (scipy):     {p_scipy:.6f}")

    # --- Decision ---
    print("\n--- Decision ---")
    if p_value < SIGNIFICANCE_LEVEL:
        print(f"  ✓ REJECT H₀ at α = {SIGNIFICANCE_LEVEL}")
        print(f"    The difference is statistically significant.")
        decision = "reject_null"
    else:
        print(f"  ✗ FAIL TO REJECT H₀ at α = {SIGNIFICANCE_LEVEL}")
        print(f"    We cannot conclude there is a meaningful difference.")
        decision = "fail_to_reject"

    results.update({
        "z_statistic": float(z_stat),
        "p_value": float(p_value),
        "ci_lower": float(ci_lower),
        "ci_upper": float(ci_upper),
        "decision": decision,
        "z_scipy": float(z_scipy),
        "p_scipy": float(p_scipy),
    })

    return results


# ===========================================================================
# SECTION 4: POWER ANALYSIS
# ===========================================================================

def power_analysis(df: pd.DataFrame) -> dict:
    """
    Retroactive power analysis: was the sample large enough?

    While power analysis is ideally done BEFORE the test, a retroactive
    check reveals whether the results are trustworthy.
    """
    print("\n" + "=" * 60)
    print("SECTION 4: POWER ANALYSIS")
    print("=" * 60)

    results = {}

    control = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    p_c = control["converted"].mean()
    p_t = treatment["converted"].mean()
    n_c = len(control)
    n_t = len(treatment)

    # Cohen's h — effect size for proportions
    cohens_h = 2 * np.arcsin(np.sqrt(p_t)) - 2 * np.arcsin(np.sqrt(p_c))
    print(f"\n  Observed effect size (Cohen's h): {cohens_h:.4f}")

    interpretation = "negligible"
    if abs(cohens_h) >= 0.8:
        interpretation = "large"
    elif abs(cohens_h) >= 0.5:
        interpretation = "medium"
    elif abs(cohens_h) >= 0.2:
        interpretation = "small"
    print(f"  Interpretation: {interpretation}")

    # Achieved power with current sample
    power_analyzer = NormalIndPower()
    avg_n = (n_c + n_t) / 2
    achieved_power = power_analyzer.power(
        effect_size=abs(cohens_h),
        nobs1=avg_n,
        alpha=SIGNIFICANCE_LEVEL,
        ratio=n_t / n_c,
        alternative="two-sided",
    )
    print(f"\n  Achieved statistical power: {achieved_power:.2%}")

    if achieved_power >= 0.80:
        print("  ✓ Adequate power (≥80%). Results are trustworthy.")
    else:
        print(f"  ⚠ UNDERPOWERED ({achieved_power:.0%} < 80%).")
        print(f"    There's a {1-achieved_power:.0%} chance we'd miss a real effect of this size.")

    # Required sample size for 80% power
    required_n = power_analyzer.solve_power(
        effect_size=abs(cohens_h) if abs(cohens_h) > 0.001 else 0.01,
        power=0.80,
        alpha=SIGNIFICANCE_LEVEL,
        ratio=1.0,
        alternative="two-sided",
    )
    print(f"\n  Required sample size per group (for 80% power): {int(required_n):,}")
    print(f"  Actual sample size per group: ~{int(avg_n):,}")

    # --- Plot: Power curve ---
    fig, ax = plt.subplots(figsize=(10, 6))

    sample_sizes = np.arange(1000, 500_001, 5000)
    powers = [
        power_analyzer.power(
            effect_size=abs(cohens_h) if abs(cohens_h) > 0.001 else 0.01,
            nobs1=n, alpha=SIGNIFICANCE_LEVEL, ratio=1.0, alternative="two-sided"
        )
        for n in sample_sizes
    ]

    ax.plot(sample_sizes, powers, linewidth=2, color="#2196F3")
    ax.axhline(y=0.80, color="red", linestyle="--", linewidth=1.5, label="80% Power Threshold")
    ax.axvline(x=avg_n, color="green", linestyle="--", linewidth=1.5, label=f"Our Sample (n={int(avg_n):,})")

    ax.set_xlabel("Sample Size per Group")
    ax.set_ylabel("Statistical Power")
    ax.set_title("Power Curve — How Sample Size Affects Detection Ability")
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.05)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "power_analysis.png", dpi=150)
    plt.close(fig)
    print("\n  [Saved: figures/power_analysis.png]")

    results.update({
        "cohens_h": float(cohens_h),
        "effect_interpretation": interpretation,
        "achieved_power": float(achieved_power),
        "required_n_per_group": int(required_n),
        "actual_n_per_group": int(avg_n),
    })

    return results


# ===========================================================================
# SECTION 5: EFFECT SIZE & REVENUE IMPACT
# ===========================================================================

def revenue_impact(df: pd.DataFrame) -> dict:
    """
    Translate the statistical effect into business terms.

    A statistically significant result is meaningless if the actual
    revenue impact doesn't justify the engineering cost of shipping.
    """
    print("\n" + "=" * 60)
    print("SECTION 5: EFFECT SIZE & REVENUE IMPACT")
    print("=" * 60)

    results = {}

    control = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]
    p_c = control["converted"].mean()
    p_t = treatment["converted"].mean()
    diff = p_t - p_c

    print(f"\n  Assumptions:")
    print(f"    Monthly visitors: {MONTHLY_VISITORS:,}")
    print(f"    Average order value: ${AVG_ORDER_VALUE:.2f}")
    print(f"    Practical significance threshold: {PRACTICAL_THRESHOLD:.1%} absolute lift")

    # Monthly conversions change
    current_monthly_conversions = MONTHLY_VISITORS * p_c
    new_monthly_conversions = MONTHLY_VISITORS * p_t
    additional_conversions = new_monthly_conversions - current_monthly_conversions

    # Revenue impact
    monthly_revenue_change = additional_conversions * AVG_ORDER_VALUE
    annual_revenue_change = monthly_revenue_change * 12

    print(f"\n  Current monthly conversions: {current_monthly_conversions:,.0f}")
    print(f"  Projected monthly conversions: {new_monthly_conversions:,.0f}")
    print(f"  Additional conversions/month: {additional_conversions:+,.0f}")
    print(f"\n  Monthly revenue impact: ${monthly_revenue_change:+,.2f}")
    print(f"  Annual revenue impact:  ${annual_revenue_change:+,.2f}")

    # Practical significance check
    print(f"\n--- Practical Significance ---")
    if abs(diff) >= PRACTICAL_THRESHOLD:
        print(f"  ✓ Effect ({diff:.2%}) exceeds practical threshold ({PRACTICAL_THRESHOLD:.1%})")
        practical_sig = True
    else:
        print(f"  ✗ Effect ({diff:.2%}) is below practical threshold ({PRACTICAL_THRESHOLD:.1%})")
        print(f"    The lift may not justify the engineering effort to ship.")
        practical_sig = False

    results.update({
        "additional_monthly_conversions": float(additional_conversions),
        "monthly_revenue_impact_usd": float(monthly_revenue_change),
        "annual_revenue_impact_usd": float(annual_revenue_change),
        "practically_significant": practical_sig,
    })

    return results


# ===========================================================================
# SECTION 6: SEGMENTED ANALYSIS
# ===========================================================================

def segmented_analysis(df: pd.DataFrame) -> dict:
    """
    Test whether the effect holds across user segments.

    Important: testing multiple segments inflates the false positive rate.
    With 4 segments at α=0.05, there's a ~18.5% chance of at least one
    false positive. We apply Bonferroni correction to be honest about this.
    """
    print("\n" + "=" * 60)
    print("SECTION 6: SEGMENTED ANALYSIS")
    print("=" * 60)

    results = {}
    segments = []

    # Define segments to test
    segment_configs = []

    if "device" in df.columns:
        for device in df["device"].unique():
            segment_configs.append(("device", device, df["device"] == device))

    if "user_type" in df.columns:
        for utype in df["user_type"].unique():
            segment_configs.append(("user_type", utype, df["user_type"] == utype))

    if "hour" in df.columns:
        df["time_period"] = pd.cut(
            df["hour"], bins=[0, 6, 12, 18, 24],
            labels=["night", "morning", "afternoon", "evening"],
            include_lowest=True,
        )
        for period in ["morning", "afternoon", "evening", "night"]:
            mask = df["time_period"] == period
            if mask.sum() > 100:
                segment_configs.append(("time_period", period, mask))

    n_segments = len(segment_configs)
    bonferroni_alpha = SIGNIFICANCE_LEVEL / n_segments if n_segments > 0 else SIGNIFICANCE_LEVEL

    print(f"\n  Testing {n_segments} segments")
    print(f"  Bonferroni-corrected α: {bonferroni_alpha:.4f} (original: {SIGNIFICANCE_LEVEL})")
    print(f"\n  {'Segment':<25} {'CR Control':>12} {'CR Treatment':>12} {'Diff':>10} {'P-value':>10} {'Sig?':>6}")
    print("  " + "-" * 80)

    for seg_type, seg_value, mask in segment_configs:
        seg_df = df[mask]
        seg_control = seg_df[seg_df["group"] == "control"]
        seg_treatment = seg_df[seg_df["group"] == "treatment"]

        if len(seg_control) < 30 or len(seg_treatment) < 30:
            continue

        cr_c = seg_control["converted"].mean()
        cr_t = seg_treatment["converted"].mean()
        diff = cr_t - cr_c

        # Z-test for this segment
        z, p = proportions_ztest(
            count=[seg_treatment["converted"].sum(), seg_control["converted"].sum()],
            nobs=[len(seg_treatment), len(seg_control)],
            alternative="two-sided",
        )

        sig = "✓" if p < bonferroni_alpha else "✗"
        label = f"{seg_type}={seg_value}"
        print(f"  {label:<25} {cr_c:>12.4f} {cr_t:>12.4f} {diff:>+10.4f} {p:>10.4f} {sig:>6}")

        segments.append({
            "segment": label,
            "n_control": len(seg_control),
            "n_treatment": len(seg_treatment),
            "cr_control": float(cr_c),
            "cr_treatment": float(cr_t),
            "difference": float(diff),
            "p_value": float(p),
            "significant_after_correction": p < bonferroni_alpha,
        })

    # Simpson's Paradox check
    print(f"\n--- Simpson's Paradox Check ---")
    overall_diff = df[df["group"] == "treatment"]["converted"].mean() - df[df["group"] == "control"]["converted"].mean()
    direction_changes = sum(
        1 for s in segments if np.sign(s["difference"]) != np.sign(overall_diff) and abs(s["difference"]) > 0.001
    )

    if direction_changes > 0:
        print(f"  ⚠ {direction_changes} segment(s) show the OPPOSITE direction from the overall result!")
        print("    This could indicate Simpson's Paradox — the aggregated result may be misleading.")
    else:
        print("  ✓ All segments agree in direction with the overall result.")

    # --- Segment visualization ---
    if segments:
        fig, ax = plt.subplots(figsize=(12, 6))

        seg_labels = [s["segment"] for s in segments]
        seg_diffs = [s["difference"] * 100 for s in segments]  # percentage
        seg_colors = ["#27ae60" if s["significant_after_correction"] else "#bdc3c7" for s in segments]

        bars = ax.barh(seg_labels, seg_diffs, color=seg_colors, edgecolor="black", linewidth=0.5)
        ax.axvline(x=0, color="black", linewidth=1)
        ax.set_xlabel("Conversion Rate Difference (percentage points)")
        ax.set_title("Treatment Effect by Segment (green = significant after Bonferroni)")

        plt.tight_layout()
        fig.savefig(FIGURES_DIR / "segment_analysis.png", dpi=150)
        plt.close(fig)
        print("\n  [Saved: figures/segment_analysis.png]")

    results["segments"] = segments
    results["bonferroni_alpha"] = float(bonferroni_alpha)
    results["simpsons_paradox_segments"] = direction_changes

    return results


# ===========================================================================
# SECTION 7: BAYESIAN ANALYSIS
# ===========================================================================

def bayesian_analysis(df: pd.DataFrame) -> dict:
    """
    Bayesian Beta-Binomial analysis.

    Instead of asking "is the difference statistically significant?",
    Bayesian analysis asks "what is the probability that treatment
    is better than control?" — a much more intuitive question for
    business stakeholders.

    Model:
        Prior: Beta(1, 1) — uninformative (uniform) prior
        Likelihood: Binomial
        Posterior: Beta(1 + successes, 1 + failures)
    """
    print("\n" + "=" * 60)
    print("SECTION 7: BAYESIAN ANALYSIS")
    print("=" * 60)

    results = {}

    control = df[df["group"] == "control"]
    treatment = df[df["group"] == "treatment"]

    # Prior parameters (uninformative)
    alpha_prior, beta_prior = 1, 1

    # Posterior parameters
    alpha_c = alpha_prior + control["converted"].sum()
    beta_c = beta_prior + (len(control) - control["converted"].sum())

    alpha_t = alpha_prior + treatment["converted"].sum()
    beta_t = beta_prior + (len(treatment) - treatment["converted"].sum())

    print(f"\n  Prior: Beta({alpha_prior}, {beta_prior}) — uninformative")
    print(f"\n  Posterior (Control):   Beta({alpha_c}, {beta_c})")
    print(f"    Mean: {alpha_c / (alpha_c + beta_c):.6f}")
    print(f"  Posterior (Treatment): Beta({alpha_t}, {beta_t})")
    print(f"    Mean: {alpha_t / (alpha_t + beta_t):.6f}")

    # Monte Carlo simulation: P(treatment > control)
    n_simulations = 100_000
    samples_control = np.random.beta(alpha_c, beta_c, size=n_simulations)
    samples_treatment = np.random.beta(alpha_t, beta_t, size=n_simulations)

    prob_treatment_better = (samples_treatment > samples_control).mean()
    print(f"\n  P(Treatment > Control): {prob_treatment_better:.4f} ({prob_treatment_better:.1%})")

    expected_lift = (samples_treatment - samples_control).mean()
    lift_ci = np.percentile(samples_treatment - samples_control, [2.5, 97.5])
    print(f"  Expected lift: {expected_lift:.4%}")
    print(f"  95% Credible interval: [{lift_ci[0]:.4%}, {lift_ci[1]:.4%}]")

    # Interpretation
    print(f"\n--- Interpretation ---")
    if prob_treatment_better > 0.95:
        print(f"  Strong evidence that treatment is better ({prob_treatment_better:.1%} probability).")
    elif prob_treatment_better > 0.80:
        print(f"  Moderate evidence that treatment is better ({prob_treatment_better:.1%} probability).")
    elif prob_treatment_better > 0.50:
        print(f"  Weak evidence — treatment might be slightly better ({prob_treatment_better:.1%}),")
        print(f"  but not enough certainty to make a confident decision.")
    else:
        print(f"  Evidence suggests control is actually better ({1-prob_treatment_better:.1%} probability).")

    # --- Plot: Posterior distributions ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Posterior PDFs
    x = np.linspace(
        min(samples_control.min(), samples_treatment.min()) * 0.998,
        max(samples_control.max(), samples_treatment.max()) * 1.002,
        1000,
    )

    axes[0].plot(x, stats.beta.pdf(x, alpha_c, beta_c),
                 color=COLORS["control"], linewidth=2.5, label="Control")
    axes[0].plot(x, stats.beta.pdf(x, alpha_t, beta_t),
                 color=COLORS["treatment"], linewidth=2.5, label="Treatment")
    axes[0].set_xlabel("Conversion Rate")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Posterior Distributions of Conversion Rate")
    axes[0].legend(fontsize=11)

    # Lift distribution
    lift_samples = (samples_treatment - samples_control) * 100
    axes[1].hist(lift_samples, bins=80, color="#9b59b6", alpha=0.7, edgecolor="black", linewidth=0.3)
    axes[1].axvline(x=0, color="red", linestyle="--", linewidth=2, label="No Effect")
    axes[1].set_xlabel("Treatment - Control (percentage points)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title(f"Distribution of Lift — P(Treatment > Control) = {prob_treatment_better:.1%}")
    axes[1].legend(fontsize=11)

    plt.tight_layout()
    fig.savefig(FIGURES_DIR / "bayesian_posteriors.png", dpi=150)
    plt.close(fig)
    print("\n  [Saved: figures/bayesian_posteriors.png]")

    results.update({
        "prob_treatment_better": float(prob_treatment_better),
        "expected_lift": float(expected_lift),
        "credible_interval_95_lower": float(lift_ci[0]),
        "credible_interval_95_upper": float(lift_ci[1]),
    })

    return results


# ===========================================================================
# SECTION 8: RECOMMENDATION
# ===========================================================================

def make_recommendation(all_results: dict) -> str:
    """
    Synthesize all analyses into a clear ship/don't-ship recommendation.
    """
    print("\n" + "=" * 60)
    print("SECTION 8: FINAL RECOMMENDATION")
    print("=" * 60)

    freq = all_results.get("frequentist", {})
    power = all_results.get("power", {})
    revenue = all_results.get("revenue", {})
    bayesian = all_results.get("bayesian", {})
    descriptive = all_results.get("descriptive", {})

    stat_sig = freq.get("decision") == "reject_null"
    practical_sig = revenue.get("practically_significant", False)
    adequate_power = power.get("achieved_power", 0) >= 0.80
    bayes_confident = bayesian.get("prob_treatment_better", 0) > 0.90

    print(f"\n  Summary of Evidence:")
    print(f"    Statistically significant:   {'Yes' if stat_sig else 'No'}")
    print(f"    Practically significant:     {'Yes' if practical_sig else 'No'}")
    print(f"    Adequate statistical power:  {'Yes' if adequate_power else 'No'}")
    print(f"    Bayesian confidence (>90%):  {'Yes' if bayes_confident else 'No'}")

    annual = revenue.get("annual_revenue_impact_usd", 0)

    print(f"\n    Estimated annual revenue impact: ${annual:+,.0f}")
    print(f"    Relative lift: {descriptive.get('relative_lift_pct', 0):+.2f}%")

    # Decision logic
    if stat_sig and practical_sig and adequate_power:
        recommendation = "SHIP"
        reasoning = (
            "The new landing page shows a statistically significant improvement "
            "that exceeds our practical significance threshold, with adequate "
            "statistical power to trust the result."
        )
    elif stat_sig and not practical_sig:
        recommendation = "DON'T SHIP"
        reasoning = (
            "While the result is statistically significant, the effect size is "
            "below our practical significance threshold. The improvement is too "
            "small to justify the engineering effort and risk of deployment."
        )
    elif not stat_sig and adequate_power:
        recommendation = "DON'T SHIP"
        reasoning = (
            "With adequate statistical power, the failure to reach significance "
            "suggests the new page does not meaningfully improve conversion rates."
        )
    elif not stat_sig and not adequate_power:
        recommendation = "EXTEND THE TEST"
        reasoning = (
            "The test was underpowered — we don't have enough data to make a "
            "confident decision either way. Extend the test to reach the required "
            f"sample size of ~{power.get('required_n_per_group', 0):,} per group."
        )
    else:
        recommendation = "REVIEW FURTHER"
        reasoning = "Mixed signals. Recommend further analysis before deciding."

    print(f"\n  ╔══════════════════════════════════════╗")
    print(f"  ║  RECOMMENDATION: {recommendation:<20s}║")
    print(f"  ╚══════════════════════════════════════╝")
    print(f"\n  Reasoning: {reasoning}")

    # Next steps
    print(f"\n  Suggested Next Steps:")
    if recommendation == "SHIP":
        print("    1. Monitor conversion rate for 2 weeks post-launch for regression")
        print("    2. Track secondary metrics (bounce rate, time-on-site, AOV)")
        print("    3. Consider testing further optimizations on the new design")
    elif recommendation == "EXTEND THE TEST":
        print(f"    1. Continue the test until ~{power.get('required_n_per_group', 0):,} users per group")
        print("    2. Set up sequential testing to allow early stopping if effect is clear")
        print("    3. Review test infrastructure for any issues causing the small sample")
    else:
        print("    1. Gather qualitative user feedback on the new design")
        print("    2. Consider testing a more differentiated redesign")
        print("    3. Investigate whether specific segments showed promise")

    return recommendation


# ===========================================================================
# MAIN
# ===========================================================================

def main():
    """Run the complete A/B test analysis."""
    print("╔══════════════════════════════════════════════════════════╗")
    print("║    E-COMMERCE A/B TEST ANALYSIS                        ║")
    print("║    Landing Page Conversion Experiment                  ║")
    print("╚══════════════════════════════════════════════════════════╝")

    # Load data
    df = load_data()

    # Run all sections
    all_results = {}
    all_results["sanity"] = run_sanity_checks(df)
    all_results["descriptive"] = descriptive_analysis(df)
    all_results["frequentist"] = frequentist_test(df)
    all_results["power"] = power_analysis(df)
    all_results["revenue"] = revenue_impact(df)
    all_results["segments"] = segmented_analysis(df)
    all_results["bayesian"] = bayesian_analysis(df)

    recommendation = make_recommendation(all_results)
    all_results["recommendation"] = recommendation

    # Save results
    results_path = RESULTS_DIR / "analysis_results.json"
    with open(results_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n  All results saved to {results_path}")

    print("\n" + "=" * 60)
    print("ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"  Figures saved to: {FIGURES_DIR}/")
    print(f"  Results saved to: {results_path}")


if __name__ == "__main__":
    main()
