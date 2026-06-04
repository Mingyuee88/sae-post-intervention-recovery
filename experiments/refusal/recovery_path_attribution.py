import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

# -----------------------------
# Reset style to avoid gray background from ggplot/seaborn-like themes
# -----------------------------
plt.style.use("default")

mpl.rcParams["figure.facecolor"] = "white"
mpl.rcParams["axes.facecolor"] = "white"
mpl.rcParams["savefig.facecolor"] = "white"
mpl.rcParams["savefig.edgecolor"] = "white"
mpl.rcParams["savefig.transparent"] = False

# -----------------------------
# Data
# -----------------------------
encoder_proj_drift = 0.00
encoder_proj_rate = 90 / 91 * 100   # 98.90%

no_proj_drift = 0.333
no_proj_rate = 91 / 91 * 100        # 100.00%

# -----------------------------
# Figure setup
# -----------------------------
fig, ax = plt.subplots(figsize=(12, 9), dpi=150, facecolor="white")
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

# Desired region (light blue vertical band)
ax.axvspan(-0.002, 0.028, color="#dbe4ef", alpha=0.9, zorder=0)

# Grid
ax.grid(True, which="major", color="#d0d5dd", linewidth=1.8, alpha=0.9)
ax.set_axisbelow(True)

# -----------------------------
# Scatter points
# -----------------------------
ax.scatter(
    encoder_proj_drift, encoder_proj_rate,
    s=900, color="#144f99", edgecolor="white", linewidth=2.5, zorder=5
)

ax.scatter(
    no_proj_drift, no_proj_rate,
    s=520, marker="s", color="#b9403b", edgecolor="white", linewidth=2.0, zorder=5
)

# -----------------------------
# Annotations
# -----------------------------
ax.annotate(
    "Encoder proj.\n90/91",
    xy=(encoder_proj_drift, encoder_proj_rate),
    xytext=(0.058, 98.78),
    fontsize=26,
    color="#144f99",
    fontweight="bold",
    ha="left",
    va="center",
    arrowprops=dict(
        arrowstyle="-",
        lw=2.0,
        color="#144f99",
        shrinkA=8,
        shrinkB=8
    ),
)

ax.annotate(
    "No projection\n91/91",
    xy=(no_proj_drift, no_proj_rate),
    xytext=(0.258, 99.92),
    fontsize=26,
    color="#b9403b",
    ha="right",
    va="center",
    arrowprops=dict(
        arrowstyle="-",
        lw=2.0,
        color="#b9403b",
        shrinkA=8,
        shrinkB=8
    ),
)

ax.text(
    0.000, 97.25,
    "desired:\nhigh recovery,\nlow drift",
    fontsize=24,
    color="#144f99",
    fontweight="bold",
    ha="left",
    va="center"
)

ax.text(
    0.362, 97.12,
    "same 91 flips\nsame post-hoc evaluator",
    fontsize=22,
    color="#4a5568",
    ha="right",
    va="center"
)

# -----------------------------
# Axes, ticks, labels
# -----------------------------
ax.set_xlim(-0.012, 0.38)
ax.set_ylim(96.6, 100.08)
ax.set_xticks(np.arange(0.00, 0.36, 0.05))
ax.set_yticks([97, 98, 99, 100])

ax.set_xlabel("Post-hoc clamp-feature drift (abs. L2)", fontsize=28, labelpad=16)
ax.set_ylabel("Recovery rate (%)", fontsize=28, labelpad=14)
ax.set_title("Recovery under active SAE clamp", fontsize=30, pad=14)

# Spine style
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_linewidth(2.3)
    ax.spines[side].set_color("black")

ax.tick_params(axis="both", labelsize=22, width=2.3, length=12, pad=10, colors="black")

plt.tight_layout()
plt.show()

# Save white background explicitly
plt.savefig(
    "recovery_under_active_sae_clamp_white_bg.png",
    dpi=300,
    bbox_inches="tight",
    facecolor="white",
    edgecolor="white",
    transparent=False
)