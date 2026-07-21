"""
Global AI Content Impact - Chart Generator
============================================
Regenerates every chart used in the corrected "Global AI Content Impact"
deck, directly from the underlying dataset.

Requirements:
    pip install pandas numpy matplotlib

Usage:
    1. Put this script and "ai_impact_data.csv" in the same folder
       (or edit CSV_PATH below to point at the dataset).
    2. Run:  python generate_all_charts.py
    3. Output PNGs are written to a "charts" subfolder next to this script.

Charts produced (17 PNG files):
    fig1_adoption_industry.png   - AI adoption rate by industry (bar + std err bars)
    fig_broad_data.png           - Industry adoption, 5-country breakdown vs global avg
    fig2_adoption_country.png    - AI adoption rate by country (horizontal bar)
    fig_trend_years.png          - Adoption trend 2020-2025, 5 countries + global avg
    fig3_jobloss_box.png         - Job loss distribution by industry (box plot)
    fig_radar_jobloss.png        - Job loss by industry, radar chart per country
    fig4_revenue_scatter.png     - Revenue increase vs AI adoption (scatter + trend)
    fig_sector_corr_pos.png      - Sector correlations, positive (revenue vs adoption)
    fig_sector_corr_neg.png      - Sector correlations, negative (revenue vs adoption)
    fig11_jobloss_revenue.png    - Job loss vs revenue by industry (grouped bar)
    fig_volatility.png           - Job loss & revenue by industry, year-by-year small multiples
    fig6_tools.png                - AI tool usage frequency (horizontal bar)
    fig8_marketshare.png         - Market share of AI companies by country
    fig10_content_volume.png     - AI-generated content volume trend (line + std band)
    fig5_trust_regulation.png    - Consumer trust by regulation status (bar + err bars)
    fig7_collab_country.png      - Human-AI collaboration rate by country
    fig9_corr_matrix.png         - Correlation heatmap of all 7 numerical variables
    fig_trust_heatmap.png        - Consumer trust, country x industry (heatmap)

All figures use a colorblind-safe, ten-color qualitative palette, avoid pie
charts and 3D effects, and encode quantities with position/length rather
than angle or area, per standard data-visualization best practice.
"""

import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, "ai_impact_data.csv")
OUT = os.path.join(BASE_DIR, "charts")
os.makedirs(OUT, exist_ok=True)

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.edgecolor"] = "#4A4A4A"
plt.rcParams["axes.linewidth"] = 0.8

df = pd.read_csv(CSV_PATH)
df.columns = ['Country', 'Year', 'Industry', 'Adoption', 'ContentVol', 'JobLoss', 'Revenue',
              'Collab', 'Tool', 'Regulation', 'Trust', 'MktShare']

# Colorblind-safe qualitative palette (10 categories)
PAL10 = ["#3B6FA8", "#E1812C", "#3E9651", "#CC4C4C", "#8072B2",
         "#8C6D31", "#CE79A6", "#767676", "#BCA136", "#4FA6A6"]
NAVY = "#233245"
INK = "#233245"
TOP5 = ["USA", "China", "Germany", "India", "Australia"]
TOP5_COL = {"USA": "#3B6FA8", "China": "#CC4C4C", "Germany": "#E1812C",
            "India": "#3E9651", "Australia": "#8072B2"}


def style_ax(ax, grid_axis="y"):
    """Strip chart-junk: no top/right/left spines, light gridlines only."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.tick_params(colors=INK, labelsize=10)
    ax.grid(axis=grid_axis, color="#DADFE3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


# ---------------------------------------------------------------------------
# Figure 1 - AI Adoption Rate by Industry (mean +/- 1 std)
# ---------------------------------------------------------------------------
g = df.groupby("Industry")["Adoption"].agg(["mean", "std"]).sort_values("mean", ascending=False)
fig, ax = plt.subplots(figsize=(8.6, 5.0), dpi=200)
ax.bar(g.index, g["mean"], yerr=g["std"], capsize=4, color=PAL10[:len(g)],
       error_kw=dict(ecolor="#5B6570", lw=1.2))
for i, (idx, row) in enumerate(g.iterrows()):
    ax.text(i, row["mean"] + row["std"] + 2, f"{row['mean']:.1f}%", ha="center", fontsize=9, color=INK)
style_ax(ax)
ax.set_ylabel("AI Adoption Rate (%)", fontsize=10)
ax.set_ylim(0, 95)
plt.xticks(rotation=28, ha="right")
ax.set_title("Figure 1 - AI Adoption Rate by Industry (mean +/- 1 std)", fontsize=12, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig1_adoption_industry.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure - The Danger of Broad Data (industry x top-5-country breakdown)
# ---------------------------------------------------------------------------
piv = df[df.Country.isin(TOP5)].pivot_table(index="Industry", columns="Country", values="Adoption", aggfunc="mean")
piv = piv.reindex(g.index)
global_avg = g["mean"]
fig, ax = plt.subplots(figsize=(11.5, 5.4), dpi=200)
x = np.arange(len(piv.index))
w = 0.15
ax.bar(x, global_avg.reindex(piv.index), width=0.82, color="#E4E7EA", label="Global avg (all 10 countries)", zorder=1)
for i, c in enumerate(TOP5):
    ax.bar(x + (i - 2) * w, piv[c], width=w * 0.9, color=TOP5_COL[c], label=c, zorder=2)
style_ax(ax)
ax.set_xticks(x)
ax.set_xticklabels(piv.index, rotation=28, ha="right")
ax.set_ylabel("AI Adoption Rate (%)", fontsize=10)
ax.legend(ncol=6, fontsize=8.5, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.16))
ax.set_title("The Danger of Broad Data - Adoption by Industry, 5 Countries vs. Global Average",
             fontsize=11.5, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_broad_data.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2 - AI Adoption Rate by Country
# ---------------------------------------------------------------------------
gc = df.groupby("Country")["Adoption"].mean().sort_values(ascending=True)
med = gc.median()
colors = ["#3B6FA8" if v >= med else "#3E9651" for v in gc.values]
fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)
ax.barh(gc.index, gc.values, color=colors, zorder=2)
for i, v in enumerate(gc.values):
    ax.text(v + 0.6, i, f"{v:.1f}%", va="center", fontsize=9.5, color=INK)
style_ax(ax, grid_axis="x")
ax.set_xlabel("Average AI Adoption Rate (%)", fontsize=10)
ax.set_xlim(0, 75)
ax.set_title("Figure 2 - Average AI Adoption Rate by Country", fontsize=12, weight="bold", loc="left")
leg = [Patch(color="#3B6FA8", label="Above median"), Patch(color="#3E9651", label="Below median")]
ax.legend(handles=leg, loc="lower right", frameon=False, fontsize=9)
plt.tight_layout()
plt.savefig(f"{OUT}/fig2_adoption_country.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure - Adoption trend 2020-2025 (top-5 countries + global average)
# ---------------------------------------------------------------------------
yearly_global = df.groupby("Year")["Adoption"].mean()
piv_y = df[df.Country.isin(TOP5)].pivot_table(index="Year", columns="Country", values="Adoption", aggfunc="mean")
fig, ax = plt.subplots(figsize=(10.6, 5.2), dpi=200)
ax.bar(yearly_global.index, yearly_global.values, color="#E4E7EA", width=0.6,
       label="Global avg (all countries)", zorder=1)
for c in TOP5:
    ax.plot(piv_y.index, piv_y[c], marker="o", lw=2.2, color=TOP5_COL[c], label=c, zorder=3)
style_ax(ax)
ax.set_ylabel("AI Adoption Rate (%)", fontsize=10)
ax.set_xticks(list(yearly_global.index))
ax.set_ylim(0, 100)
ax.legend(ncol=6, fontsize=9, frameon=False, loc="upper center", bbox_to_anchor=(0.5, 1.14))
ax.set_title("Tracking Adoption Trends, 2020-2025", fontsize=12, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig_trend_years.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3 - Job Loss Distribution by Industry (box plot)
# ---------------------------------------------------------------------------
order = df.groupby("Industry")["JobLoss"].median().sort_values(ascending=False).index
data = [df[df.Industry == i]["JobLoss"].values for i in order]
fig, ax = plt.subplots(figsize=(9.2, 5.2), dpi=200)
bp = ax.boxplot(data, patch_artist=True, tick_labels=list(order), widths=0.55,
                 medianprops=dict(color="white", lw=2),
                 flierprops=dict(marker="o", markerfacecolor="#CC4C4C", markersize=4, markeredgecolor="none"))
for patch, col in zip(bp["boxes"], PAL10[:len(order)]):
    patch.set_facecolor(col)
    patch.set_edgecolor("none")
style_ax(ax)
ax.set_ylabel("Job Loss Due to AI (%)", fontsize=10)
plt.xticks(rotation=28, ha="right")
ax.set_title("Figure 3 - Job Loss Distribution by Industry", fontsize=12, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig3_jobloss_box.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure - Job Loss by Industry, radar chart per country (top 5)
# ---------------------------------------------------------------------------
industries_all = sorted(df.Industry.unique())
piv_r = df[df.Country.isin(TOP5)].pivot_table(index="Country", columns="Industry", values="JobLoss", aggfunc="mean")
piv_r = piv_r.reindex(columns=industries_all)
angles = np.linspace(0, 2 * np.pi, len(industries_all), endpoint=False).tolist()
angles += angles[:1]
fig, axes = plt.subplots(1, 5, figsize=(16.5, 3.6), dpi=200, subplot_kw=dict(polar=True))
for ax, c in zip(axes, TOP5):
    vals = piv_r.loc[c].values
    vals = np.nan_to_num(vals, nan=0.0).tolist()
    vals += vals[:1]
    ax.plot(angles, vals, color=TOP5_COL[c], lw=1.8)
    ax.fill(angles, vals, color=TOP5_COL[c], alpha=0.28)
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(industries_all, fontsize=6.3)
    ax.set_ylim(0, 50)
    ax.set_yticks([25, 50])
    ax.set_yticklabels(["25%", "50%"], fontsize=6)
    ax.set_title(c, fontsize=11, weight="bold", color=TOP5_COL[c], pad=14)
    ax.grid(color="#CBD2D8", linewidth=0.6)
fig.suptitle("Job Loss Due to AI (%) by Industry - Country Radars (blank = no data for that cell)",
             fontsize=11.5, weight="bold", y=1.05)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_radar_jobloss.png", facecolor="white", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 4 - Revenue Increase vs AI Adoption Rate (scatter + trend line)
# ---------------------------------------------------------------------------
r = df["Adoption"].corr(df["Revenue"])
z = np.polyfit(df["Adoption"], df["Revenue"], 1)
xs = np.linspace(df["Adoption"].min(), df["Adoption"].max(), 50)
fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)
ax.scatter(df["Adoption"], df["Revenue"], s=26, color="#4FA6A6", alpha=0.75, edgecolor="none", zorder=2)
ax.plot(xs, np.polyval(z, xs), "--", color="#CC4C4C", lw=1.8, label=f"Trend line (slope={z[0]:.3f})", zorder=3)
ax.text(0.03, 0.94, f"r = {r:.3f}", transform=ax.transAxes, fontsize=12, weight="bold", color=INK)
style_ax(ax)
ax.set_xlabel("AI Adoption Rate (%)", fontsize=10)
ax.set_ylabel("Revenue Increase Due to AI (%)", fontsize=10)
ax.legend(frameon=False, fontsize=9, loc="upper right")
ax.set_title("Figure 4 - Revenue Increase vs AI Adoption Rate (all 200 records)",
             fontsize=12, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig4_revenue_scatter.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure - Sector-specific correlations (revenue vs adoption), top-5 countries
# ---------------------------------------------------------------------------
pos_ind = ["Marketing", "Gaming", "Legal", "Media", "Education", "Automotive", "Manufacturing"]
neg_ind = ["Finance", "Healthcare", "Retail"]


def scatter_panel(ax, ind):
    sub = df[(df.Industry == ind) & (df.Country.isin(TOP5))]
    rr = sub["Adoption"].corr(sub["Revenue"])
    for c in TOP5:
        s2 = sub[sub.Country == c]
        ax.scatter(s2["Adoption"], s2["Revenue"], s=22, color=TOP5_COL[c], edgecolor="none")
    if len(sub) >= 2:
        zz = np.polyfit(sub["Adoption"], sub["Revenue"], 1)
        xxs = np.linspace(sub["Adoption"].min(), sub["Adoption"].max(), 20)
        ax.plot(xxs, np.polyval(zz, xxs), "--", color="#888888", lw=1.2)
    ax.set_title(f"{ind}\nr = {rr:+.3f}  (n={len(sub)})", fontsize=8.6)
    ax.tick_params(labelsize=6.5)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    style_ax(ax)


fig, axes = plt.subplots(1, 7, figsize=(16.5, 2.7), dpi=200)
fig.suptitle("Positive correlation - AI adoption tracks with revenue growth (r > 0)   |   "
             "USA, China, Germany, India, Australia, 2020-2025", fontsize=9.5, weight="bold", color="#2E7D46")
for ax, ind in zip(axes, pos_ind):
    scatter_panel(ax, ind)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_sector_corr_pos.png", facecolor="white", bbox_inches="tight")
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.7), dpi=200)
fig.suptitle("Negative correlation - AI adoption does NOT track with revenue growth (r < 0)",
             fontsize=9.5, weight="bold", color="#B23A3A")
for ax, ind in zip(axes, neg_ind):
    scatter_panel(ax, ind)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_sector_corr_neg.png", facecolor="white", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 11 - Job Loss vs Revenue Increase by Industry (grouped bar)
# ---------------------------------------------------------------------------
jv = df.groupby("Industry")[["JobLoss", "Revenue"]].mean().sort_values("Revenue", ascending=False)
fig, ax = plt.subplots(figsize=(9.4, 5.2), dpi=200)
x = np.arange(len(jv.index))
w = 0.38
ax.bar(x - w / 2, jv["JobLoss"], width=w, color="#CC7A4C", label="Job Loss (%)")
ax.bar(x + w / 2, jv["Revenue"], width=w, color="#3E9651", label="Revenue Increase (%)")
style_ax(ax)
ax.set_xticks(x)
ax.set_xticklabels(jv.index, rotation=28, ha="right")
ax.set_ylabel("Percentage (%)", fontsize=10)
ax.legend(frameon=False, fontsize=9.5)
ax.set_title("Figure 11 - Job Loss vs Revenue Increase by Industry (avg)", fontsize=12, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig11_jobloss_revenue.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure - Economic volatility by industry, year-by-year small multiples
# ---------------------------------------------------------------------------
piv_jl = df.pivot_table(index="Year", columns="Industry", values="JobLoss", aggfunc="mean")
piv_rv = df.pivot_table(index="Year", columns="Industry", values="Revenue", aggfunc="mean")
inds = list(jv.index)
fig, axes = plt.subplots(2, 5, figsize=(16.5, 5.6), dpi=200, sharex=True)
for ax, ind in zip(axes.flat, inds):
    ax.plot(piv_jl.index, piv_jl[ind], marker="o", ms=3.5, lw=1.6, color="#CC7A4C", label="Job Loss")
    ax.plot(piv_rv.index, piv_rv[ind], marker="o", ms=3.5, lw=1.6, color="#3E9651", label="Revenue")
    ax.set_title(ind, fontsize=9.5, weight="bold")
    ax.set_ylim(0, 90)
    ax.tick_params(labelsize=7)
    style_ax(ax)
axes.flat[0].legend(fontsize=8, frameon=False, loc="upper left")
fig.suptitle("Tracking Economic Volatility by Industry, 2020-2025 (%, yearly average)",
             fontsize=12.5, weight="bold", y=1.02)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_volatility.png", facecolor="white", bbox_inches="tight")
plt.close()

# ---------------------------------------------------------------------------
# Figure 6 - AI Tool Usage Frequency
# ---------------------------------------------------------------------------
tc = df["Tool"].value_counts().sort_values(ascending=True)
fig, ax = plt.subplots(figsize=(8.4, 5.0), dpi=200)
ax.barh(tc.index, tc.values, color=PAL10[:len(tc)][::-1], zorder=2)
tot = tc.sum()
for i, v in enumerate(tc.values):
    ax.text(v + 0.6, i, f"{v} ({v / tot * 100:.1f}%)", va="center", fontsize=9.5, color=INK)
style_ax(ax, grid_axis="x")
ax.set_xlabel("Number of Records", fontsize=10)
ax.set_xlim(0, 42)
ax.set_title("Figure 6 - AI Tool Usage Frequency", fontsize=12, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig6_tools.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure 8 - Market Share of AI Companies by Country
# ---------------------------------------------------------------------------
ms = df.groupby("Country")["MktShare"].mean().sort_values(ascending=True)
gmean = df["MktShare"].mean()
fig, ax = plt.subplots(figsize=(8.6, 5.2), dpi=200)
colors = ["#3B6FA8" if v >= gmean else "#8C8C8C" for v in ms.values]
ax.barh(ms.index, ms.values, color=colors, zorder=2)
for i, v in enumerate(ms.values):
    ax.text(v + 0.5, i, f"{v:.1f}%", va="center", fontsize=9.5, color=INK)
ax.axvline(gmean, color="#CC4C4C", ls="--", lw=1.3)
ax.text(gmean + 0.3, -0.9, f"Global mean: {gmean:.1f}%", fontsize=8.5, color="#CC4C4C")
style_ax(ax, grid_axis="x")
ax.set_xlabel("Market Share of AI Companies (%)", fontsize=10)
ax.set_xlim(0, 36)
ax.set_title("Figure 8 - Average Market Share of AI Companies, by Country", fontsize=12, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig8_marketshare.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure 10 - AI-Generated Content Volume Trend (2020-2025)
# ---------------------------------------------------------------------------
cv = df.groupby("Year")["ContentVol"].agg(["mean", "std"])
fig, ax = plt.subplots(figsize=(9.0, 5.0), dpi=200)
ax.fill_between(cv.index, cv["mean"] - cv["std"], cv["mean"] + cv["std"], color="#3B6FA8", alpha=0.18,
                label="+/-1 std dev")
ax.plot(cv.index, cv["mean"], marker="o", color="#3B6FA8", lw=2.2, label="Mean volume")
for xx, yy in zip(cv.index, cv["mean"]):
    ax.text(xx, yy + 3, f"{yy:.1f}", ha="center", fontsize=9, color=INK)
style_ax(ax)
ax.set_ylabel("AI-Generated Content Volume (TBs/year)", fontsize=10)
ax.set_xticks(list(cv.index))
ax.legend(frameon=False, fontsize=9.5)
ax.set_title("Figure 10 - AI-Generated Content Volume Trend (2020-2025)", fontsize=12, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig10_content_volume.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure 5 - Consumer Trust in AI by Regulation Status
# ---------------------------------------------------------------------------
tr = df.groupby("Regulation")["Trust"].agg(["mean", "std", "count"]).reindex(["Lenient", "Moderate", "Strict"])
fig, ax = plt.subplots(figsize=(7.6, 5.0), dpi=200)
cols = ["#3E9651", "#E1812C", "#CC4C4C"]
ax.bar(tr.index, tr["mean"], yerr=tr["std"], capsize=5, color=cols, error_kw=dict(ecolor="#5B6570", lw=1.2))
for i, (idx, row) in enumerate(tr.iterrows()):
    ax.text(i, row["mean"] + row["std"] + 2, f"{row['mean']:.1f}% (n={int(row['count'])})",
            ha="center", fontsize=9.5, color=INK)
style_ax(ax)
ax.set_ylabel("Average Consumer Trust in AI (%)", fontsize=10)
ax.set_ylim(0, 90)
ax.set_title("Figure 5 - Consumer Trust in AI by Regulation Status", fontsize=12, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig5_trust_regulation.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure 7 - Human-AI Collaboration Rate by Country
# ---------------------------------------------------------------------------
cc = df.groupby("Country")["Collab"].mean().sort_values(ascending=False)
overall = df["Collab"].mean()
fig, ax = plt.subplots(figsize=(9.2, 5.0), dpi=200)
ax.bar(cc.index, cc.values, color="#8072B2", zorder=2)
ax.axhline(overall, color="#CC4C4C", ls="--", lw=1.3, label=f"Overall mean: {overall:.1f}%")
for i, v in enumerate(cc.values):
    ax.text(i, v + 1.2, f"{v:.1f}%", ha="center", fontsize=8.8, color=INK)
style_ax(ax)
plt.xticks(rotation=28, ha="right")
ax.set_ylabel("Human-AI Collaboration Rate (%)", fontsize=10)
ax.set_ylim(0, 72)
ax.legend(frameon=False, fontsize=9.5, loc="upper right")
ax.set_title("Figure 7 - Human-AI Collaboration Rate by Country", fontsize=12, weight="bold", loc="left")
plt.tight_layout()
plt.savefig(f"{OUT}/fig7_collab_country.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure 9 - Correlation Heatmap of All Numerical Variables
# ---------------------------------------------------------------------------
num = df[["Adoption", "ContentVol", "JobLoss", "Revenue", "Collab", "Trust", "MktShare"]]
labels = ["AI Adoption", "Content Vol.", "Job Loss", "Revenue", "Collaboration", "Trust", "Mkt Share"]
corr = num.corr().values
fig, ax = plt.subplots(figsize=(7.6, 6.6), dpi=200)
im = ax.imshow(corr, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(labels)))
ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
ax.set_yticks(range(len(labels)))
ax.set_yticklabels(labels, fontsize=9)
for i in range(len(labels)):
    for j in range(len(labels)):
        v = corr[i, j]
        ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.3,
                color="white" if abs(v) > 0.5 else INK)
cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label("Correlation coefficient", fontsize=9)
ax.set_title("Figure 9 - Correlation Heatmap of All Numerical Variables", fontsize=12, weight="bold")
plt.tight_layout()
plt.savefig(f"{OUT}/fig9_corr_matrix.png", facecolor="white")
plt.close()

# ---------------------------------------------------------------------------
# Figure - Consumer Trust heatmap, Country x Industry (border = regulation)
# ---------------------------------------------------------------------------
piv_t = df.pivot_table(index="Country", columns="Industry", values="Trust", aggfunc="mean")
piv_reg = df.pivot_table(index="Country", columns="Industry", values="Regulation",
                          aggfunc=lambda x: x.mode().iat[0] if len(x) > 0 else None)
reg_color = {"Lenient": "#3E9651", "Moderate": "#E1812C", "Strict": "#CC4C4C"}
countries_order = piv_t.mean(axis=1).sort_values(ascending=False).index
piv_t = piv_t.reindex(countries_order)
piv_reg = piv_reg.reindex(countries_order)
inds_order = piv_t.columns
fig, ax = plt.subplots(figsize=(11.5, 6.6), dpi=200)
cmap = plt.cm.BuGn
norm = plt.Normalize(vmin=25, vmax=90)
for i, c in enumerate(piv_t.index):
    for j, ind in enumerate(inds_order):
        v = piv_t.loc[c, ind]
        if pd.isna(v):
            ax.add_patch(plt.Rectangle((j, len(piv_t) - 1 - i), 1, 1, facecolor="#EDEDED", edgecolor="#CCCCCC"))
            ax.text(j + 0.5, len(piv_t) - 1 - i + 0.5, "n/a", ha="center", va="center", fontsize=6.5, color="#999999")
            continue
        reg = piv_reg.loc[c, ind]
        edge = reg_color.get(reg, "#BBBBBB")
        lw = 3.2 if reg == "Strict" else (2.0 if reg == "Moderate" else 1.0)
        ax.add_patch(plt.Rectangle((j, len(piv_t) - 1 - i), 1, 1, facecolor=cmap(norm(v)), edgecolor=edge, linewidth=lw))
        txt_col = "white" if v >= 68 else INK
        ax.text(j + 0.5, len(piv_t) - 1 - i + 0.5, f"{v:.0f}%", ha="center", va="center", fontsize=8, color=txt_col)
ax.set_xlim(0, len(inds_order))
ax.set_ylim(0, len(piv_t))
ax.set_xticks(np.arange(len(inds_order)) + 0.5)
ax.set_xticklabels(inds_order, rotation=30, ha="right", fontsize=9)
ax.set_yticks(np.arange(len(piv_t)) + 0.5)
ax.set_yticklabels(piv_t.index[::-1], fontsize=9)
ax.set_title("Consumer Trust in AI (%) - Country x Industry  |  border color = regulation strictness  |  gray = no data",
             fontsize=10.5, weight="bold")
handles = [Patch(facecolor="white", edgecolor=c, linewidth=2.4, label=k) for k, c in reg_color.items()]
ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False, fontsize=9.5)
for s in ax.spines.values():
    s.set_visible(False)
plt.tight_layout()
plt.savefig(f"{OUT}/fig_trust_heatmap.png", facecolor="white", bbox_inches="tight")
plt.close()

print(f"Done. 17 charts written to: {OUT}")
