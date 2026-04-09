"""
Statistics Basics for Data Science
=====================================
Covers: descriptive statistics, probability distributions, correlation,
        confidence intervals, and hypothesis testing.
"""

import math
import numpy as np
from scipy import stats

# ---------------------------------------------------------
# 1. Descriptive Statistics
# ---------------------------------------------------------
print("=== Descriptive Statistics ===")

data = [23, 45, 67, 12, 89, 34, 56, 78, 90, 23, 45, 67, 34, 56, 78]

mean    = np.mean(data)
median  = np.median(data)
mode_res = stats.mode(data, keepdims=True)
variance = np.var(data, ddof=1)
std_dev  = np.std(data, ddof=1)
q1, q3  = np.percentile(data, [25, 75])
iqr      = q3 - q1
skewness = stats.skew(data)
kurtosis = stats.kurtosis(data)

print(f"Data:      {data}")
print(f"Mean:      {mean:.2f}")
print(f"Median:    {median:.2f}")
print(f"Mode:      {mode_res.mode[0]} (count={mode_res.count[0]})")
print(f"Variance:  {variance:.2f}")
print(f"Std Dev:   {std_dev:.2f}")
print(f"Q1:        {q1:.2f}")
print(f"Q3:        {q3:.2f}")
print(f"IQR:       {iqr:.2f}")
print(f"Skewness:  {skewness:.4f}")
print(f"Kurtosis:  {kurtosis:.4f}")

# ---------------------------------------------------------
# 2. Probability Distributions
# ---------------------------------------------------------
print("\n=== Probability Distributions ===")

# Normal distribution
mu, sigma = 100, 15
normal_dist = stats.norm(mu, sigma)
x_val = 115
print(f"Normal(μ={mu}, σ={sigma})")
print(f"  P(X = {x_val}): PDF = {normal_dist.pdf(x_val):.6f}")
print(f"  P(X ≤ {x_val}): CDF = {normal_dist.cdf(x_val):.4f}")
print(f"  90th percentile: {normal_dist.ppf(0.90):.2f}")

# Binomial distribution
n_trials, p_success = 10, 0.3
binom_dist = stats.binom(n_trials, p_success)
k = 4
print(f"\nBinomial(n={n_trials}, p={p_success})")
print(f"  P(X = {k}): {binom_dist.pmf(k):.4f}")
print(f"  P(X ≤ {k}): {binom_dist.cdf(k):.4f}")
print(f"  Mean: {binom_dist.mean():.2f}  Std: {binom_dist.std():.4f}")

# Poisson distribution
lambda_val = 3.5
poisson_dist = stats.poisson(lambda_val)
print(f"\nPoisson(λ={lambda_val})")
for k_val in range(6):
    print(f"  P(X = {k_val}): {poisson_dist.pmf(k_val):.4f}")

# ---------------------------------------------------------
# 3. Correlation & Covariance
# ---------------------------------------------------------
print("\n=== Correlation & Covariance ===")

rng = np.random.default_rng(seed=42)
x = rng.normal(50, 10, 50)
y = 2 * x + rng.normal(0, 5, 50)
z = rng.normal(50, 10, 50)

pearson_xy, p_val  = stats.pearsonr(x, y)
pearson_xz, p_val2 = stats.pearsonr(x, z)
spearman_xy, _     = stats.spearmanr(x, y)

print(f"Pearson  r(x, y) = {pearson_xy:.4f}  (p={p_val:.4e})  → strong positive")
print(f"Pearson  r(x, z) = {pearson_xz:.4f}  (p={p_val2:.4f})  → near zero")
print(f"Spearman r(x, y) = {spearman_xy:.4f}")

cov_matrix = np.cov(x, y)
print(f"Covariance matrix:\n{cov_matrix.round(2)}")

# ---------------------------------------------------------
# 4. Confidence Intervals
# ---------------------------------------------------------
print("\n=== Confidence Intervals ===")

sample = rng.normal(loc=170, scale=10, size=30)
confidence = 0.95
ci = stats.t.interval(confidence, df=len(sample)-1,
                       loc=np.mean(sample), scale=stats.sem(sample))
print(f"Sample mean:  {np.mean(sample):.2f}")
print(f"Sample std:   {np.std(sample, ddof=1):.2f}")
print(f"95% CI: ({ci[0]:.2f}, {ci[1]:.2f})")

# ---------------------------------------------------------
# 5. Hypothesis Testing
# ---------------------------------------------------------
print("\n=== Hypothesis Testing ===")

# One-sample t-test: is the mean different from 170?
t_stat, p_value = stats.ttest_1samp(sample, popmean=170)
print(f"\nOne-Sample t-test  (H₀: μ = 170)")
print(f"  t-statistic = {t_stat:.4f}")
print(f"  p-value     = {p_value:.4f}")
print(f"  Result: {'Reject H₀' if p_value < 0.05 else 'Fail to reject H₀'} at α=0.05")

# Two-sample t-test
group_a = rng.normal(75, 10, 40)
group_b = rng.normal(80, 10, 40)
t2, p2 = stats.ttest_ind(group_a, group_b)
print(f"\nTwo-Sample t-test  (H₀: μ_A = μ_B)")
print(f"  Group A mean = {group_a.mean():.2f}  |  Group B mean = {group_b.mean():.2f}")
print(f"  t-statistic  = {t2:.4f}")
print(f"  p-value      = {p2:.4f}")
print(f"  Result: {'Reject H₀' if p2 < 0.05 else 'Fail to reject H₀'} at α=0.05")

# Chi-square test of independence
observed = np.array([[30, 20], [15, 35]])
chi2, p_chi, dof, expected = stats.chi2_contingency(observed)
print(f"\nChi-Square Test of Independence")
print(f"  χ² = {chi2:.4f}  df = {dof}  p-value = {p_chi:.4f}")
print(f"  Result: {'Reject H₀' if p_chi < 0.05 else 'Fail to reject H₀'} at α=0.05")

# ---------------------------------------------------------
# 6. Z-scores & Normalization
# ---------------------------------------------------------
print("\n=== Z-scores & Normalization ===")

raw_scores = np.array([55, 70, 85, 60, 90, 45, 80, 75])
z_scores = stats.zscore(raw_scores)
print("Raw scores:", raw_scores)
print("Z-scores:  ", z_scores.round(3))

min_max = (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min())
print("Min-Max:   ", min_max.round(3))

print("\nDone! Statistics basics covered successfully.")
