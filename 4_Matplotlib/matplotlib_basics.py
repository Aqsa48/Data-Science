"""
Matplotlib Basics for Data Science
=====================================
Covers: line plots, bar charts, histograms, scatter plots, box plots,
        subplots, and plot customization.

Note: Figures are saved as PNG files (no interactive display needed).
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")  # non-interactive backend for script execution

# ---------------------------------------------------------
# Helper: save figure and report
# ---------------------------------------------------------
OUTPUT_DIR = "."


def save(name):
    path = f"{OUTPUT_DIR}/{name}.png"
    plt.savefig(path, dpi=100, bbox_inches="tight")
    plt.close()
    print(f"  Saved → {name}.png")


# ---------------------------------------------------------
# 1. Line Plot
# ---------------------------------------------------------
print("=== Line Plot ===")

x = np.linspace(0, 2 * np.pi, 300)
fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(x, np.sin(x), label="sin(x)", linewidth=2, color="royalblue")
ax.plot(x, np.cos(x), label="cos(x)", linewidth=2, color="tomato", linestyle="--")
ax.set_title("Sine and Cosine Functions")
ax.set_xlabel("x (radians)")
ax.set_ylabel("Amplitude")
ax.legend()
ax.grid(True, linestyle=":", alpha=0.7)
save("01_line_plot")

# ---------------------------------------------------------
# 2. Bar Chart
# ---------------------------------------------------------
print("=== Bar Chart ===")

departments = ["Engineering", "Marketing", "HR", "Finance", "Sales"]
avg_salaries = [90_000, 65_000, 60_000, 75_000, 70_000]

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.bar(departments, avg_salaries, color="steelblue", edgecolor="navy", alpha=0.85)
ax.bar_label(bars, fmt="${:,.0f}", padding=5, fontsize=9)
ax.set_title("Average Salary by Department")
ax.set_ylabel("Salary (USD)")
ax.set_ylim(0, 110_000)
ax.grid(axis="y", linestyle="--", alpha=0.6)
save("02_bar_chart")

# ---------------------------------------------------------
# 3. Histogram
# ---------------------------------------------------------
print("=== Histogram ===")

rng = np.random.default_rng(seed=42)
heights = rng.normal(loc=170, scale=10, size=500)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(heights, bins=25, color="mediumseagreen", edgecolor="white", alpha=0.9)
ax.axvline(heights.mean(), color="crimson", linestyle="--", linewidth=2, label=f"Mean: {heights.mean():.1f}")
ax.set_title("Distribution of Heights")
ax.set_xlabel("Height (cm)")
ax.set_ylabel("Frequency")
ax.legend()
save("03_histogram")

# ---------------------------------------------------------
# 4. Scatter Plot
# ---------------------------------------------------------
print("=== Scatter Plot ===")

study_hours = rng.uniform(1, 10, 80)
exam_score = 50 + 5 * study_hours + rng.normal(0, 5, 80)

fig, ax = plt.subplots(figsize=(7, 6))
sc = ax.scatter(study_hours, exam_score, c=exam_score, cmap="viridis", s=60, alpha=0.8, edgecolors="grey", linewidths=0.4)
plt.colorbar(sc, ax=ax, label="Exam Score")
# Trend line
m, b = np.polyfit(study_hours, exam_score, 1)
x_line = np.linspace(study_hours.min(), study_hours.max(), 100)
ax.plot(x_line, m * x_line + b, color="crimson", linewidth=2, label=f"Trend: y={m:.1f}x+{b:.1f}")
ax.set_title("Study Hours vs Exam Score")
ax.set_xlabel("Study Hours")
ax.set_ylabel("Exam Score")
ax.legend()
save("04_scatter_plot")

# ---------------------------------------------------------
# 5. Box Plot
# ---------------------------------------------------------
print("=== Box Plot ===")

np.random.seed(0)
groups = {
    "Group A": rng.normal(70, 10, 50),
    "Group B": rng.normal(80, 15, 50),
    "Group C": rng.normal(65, 8, 50),
}

fig, ax = plt.subplots(figsize=(8, 5))
ax.boxplot(groups.values(), tick_labels=groups.keys(), patch_artist=True,
           boxprops=dict(facecolor="lightblue", color="navy"),
           medianprops=dict(color="crimson", linewidth=2))
ax.set_title("Score Distribution by Group")
ax.set_ylabel("Score")
ax.grid(axis="y", linestyle="--", alpha=0.5)
save("05_box_plot")

# ---------------------------------------------------------
# 6. Subplots Grid
# ---------------------------------------------------------
print("=== Subplots Grid ===")

x_data = np.linspace(-3, 3, 200)

fig, axes = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Common Mathematical Functions", fontsize=14, fontweight="bold")

axes[0, 0].plot(x_data, x_data ** 2, color="royalblue")
axes[0, 0].set_title("Quadratic  y = x²")

axes[0, 1].plot(x_data, np.abs(x_data), color="orange")
axes[0, 1].set_title("Absolute Value  y = |x|")

axes[1, 0].plot(x_data, np.where(x_data >= 0, x_data, 0), color="mediumseagreen")
axes[1, 0].set_title("ReLU  y = max(0, x)")

axes[1, 1].plot(x_data, 1 / (1 + np.exp(-x_data)), color="tomato")
axes[1, 1].set_title("Sigmoid  y = 1/(1+e⁻ˣ)")

for ax in axes.flat:
    ax.axhline(0, color="grey", linewidth=0.5)
    ax.axvline(0, color="grey", linewidth=0.5)
    ax.grid(True, linestyle=":", alpha=0.6)

plt.tight_layout()
save("06_subplots_grid")

print("\nDone! All Matplotlib plots saved as PNG files.")
