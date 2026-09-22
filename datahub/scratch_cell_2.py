# SCRATCH CELL 2
# Feel free to do your rough work here
# Remember to comment out your code for step 5
# =====================================================================================
#  Is NYT coverage of AI more negative when the article is about health & wellness?
#  Needs: news_df_sentiment (from earlier in hw03) + ai_health_scores.csv (from Colab)
# =====================================================================================
AI_COL      = "GPT Model"             # regex-count column for AI coverage
SCORES_FILE = "ai_health_scores.csv"  # upload next to hw03.ipynb
THRESHOLD   = 0.5                     # health_score >= 0.5 -> "about health & wellness"
N_LABELS    = 3                       # headlines to annotate (0 = none)
N_BOOT      = 2000                    # bootstrap resamples for the confidence intervals
rng = np.random.default_rng(100)
import warnings; warnings.filterwarnings("ignore", category=RuntimeWarning)   # empty groups -> NaN, handled below

# ---------- 1 · Inner join the model's scores onto news_df_sentiment by article URL ----------
scores = pl.read_csv(SCORES_FILE, infer_schema_length=10000)
joined = (
    news_df_sentiment
    .select(["web_url", "pub_date", "Quarter", "article_sentiment"])
    .join(scores, on="web_url", how="inner")
    .unique(subset="web_url", keep="first", maintain_order=True)
)
n_ai_total = news_df_sentiment.filter(pl.col(AI_COL) > 0).get_column("web_url").n_unique()
df = pd.DataFrame(joined.to_dict(as_series=False))   # small now (~3k rows), so pandas is fine
# ---------- (polars ends here) ----------

df["date"] = pd.to_datetime(df["pub_date"].str.slice(0, 10))
df["health_score"] = df["health_score"].astype(float)
df["article_sentiment"] = df["article_sentiment"].astype(float)
df["is_health"] = df["health_score"] >= THRESHOLD
ai, base = df[df["group"] == "ai"].copy(), df[df["group"] == "baseline"].copy()

# ---------- 2 · Statistics ----------
def boot_means(x):
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return np.full(N_BOOT, np.nan)
    return x[rng.integers(0, len(x), size=(N_BOOT, len(x)))].mean(axis=1)

def summarize(sub):
    h = sub.loc[sub["is_health"], "article_sentiment"]
    o = sub.loc[~sub["is_health"], "article_sentiment"]
    bh, bo = boot_means(h), boot_means(o)
    return dict(n_h=len(h), n_o=len(o), m_h=h.mean(), m_o=o.mean(),
                ci_h=np.nanpercentile(bh, [2.5, 97.5]), ci_o=np.nanpercentile(bo, [2.5, 97.5]),
                gap=h.mean() - o.mean(), boot_gap=bh - bo)

S_ai, S_base = summarize(ai), summarize(base)
gap_ci  = np.nanpercentile(S_ai["boot_gap"], [2.5, 97.5])
base_ci = np.nanpercentile(S_base["boot_gap"], [2.5, 97.5])
did     = S_ai["gap"] - S_base["gap"]                                   # AI-specific extra health gap
did_ci  = np.nanpercentile(S_ai["boot_gap"] - S_base["boot_gap"], [2.5, 97.5])

fmt = lambda x: "n/a" if np.isnan(x) else f"{x:+.2f}".replace("-", "−")                          # typographic minus for the figure

def rank_corr(a, b):
    return np.corrcoef(pd.Series(a).rank(), pd.Series(b).rank())[0, 1]

ra = ai["health_score"].rank().to_numpy()
rb = ai["article_sentiment"].rank().to_numpy()
rho = np.corrcoef(ra, rb)[0, 1]                                         # Spearman correlation
perm = np.array([np.corrcoef(rng.permutation(ra), rb)[0, 1] for _ in range(N_BOOT)])
p_perm = (np.sum(np.abs(perm) >= abs(rho)) + 1) / (N_BOOT + 1)

def auc(y, s):
    y = np.asarray(y).astype(bool)
    r = pd.Series(s).rank().to_numpy()
    n1, n0 = y.sum(), (~y).sum()
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0) if n1 and n0 else np.nan

val_auc = auc(df["nyt_tag_health"], df["health_score"]) if "nyt_tag_health" in df else np.nan

qt = (ai.groupby("Quarter")
        .agg(n=("web_url", "size"), health_share=("is_health", "mean"),
             mean_health=("health_score", "mean"), mean_sent=("article_sentiment", "mean"))
        .sort_index())
q_ok = qt[qt["n"] >= 5]
rho_q = rank_corr(q_ok["mean_health"], q_ok["mean_sent"]) if len(q_ok) >= 4 else np.nan

# ---------- 3 · Headline that states the result (chosen from the numbers, not hard-coded) ----------
if S_ai["n_h"] == 0:
    verdict = "NYT's GPT coverage almost never discusses health, so health framing can't explain its tone"
elif gap_ci[1] < 0 and did_ci[1] < 0:
    verdict = "Health-related AI coverage is more negative, beyond what health news alone explains"
elif gap_ci[1] < 0:
    verdict = "Health-related AI stories are more negative, but no more than NYT health coverage in general"
elif gap_ci[0] > 0:
    verdict = "Health-related AI stories are, if anything, more positive than other AI coverage"
else:
    verdict = "No clear link between health framing and negative tone in NYT's AI coverage"

# ---------- 4 · Figure ----------
C_H, C_O = "#eb6834", "#2a78d6"                                  # health (orange) / other (blue)
INK, INK2, MUTED, GRID, AXIS, SURF = "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#fcfcfb"
S_MIN, S_MAX = 22, 560                                           # bubble AREA is proportional to health score
size = lambda s: S_MIN + (S_MAX - S_MIN) * np.asarray(s, dtype=float)
X0, X1 = pd.Timestamp("2019-01-01"), pd.Timestamp("2025-01-01")  # same quarters as the original chart
CHATGPT = pd.Timestamp("2022-11-30")

plt.rcParams.update({"font.family": "DejaVu Sans", "axes.titlesize": 11})
fig = plt.figure(figsize=(13.5, 8.2), facecolor=SURF)
gs = fig.add_gridspec(1, 2, width_ratios=[4.4, 1], wspace=0.03, left=0.075, right=0.985, top=0.765, bottom=0.14)
ax = fig.add_subplot(gs[0], facecolor=SURF)
axs = fig.add_subplot(gs[1], sharey=ax, facecolor=SURF)

# quarter grid + two-level x axis (Q1-Q4, then the year)
q_starts = pd.date_range(X0, X1, freq="QS")
for d in q_starts:
    ax.axvline(d, color=GRID, lw=0.7, zorder=0)
for d in pd.date_range(X0, X1, freq="YS"):
    ax.axvline(d, color=AXIS, lw=1.0, zorder=0)
mids = [a + (b - a) / 2 for a, b in zip(q_starts[:-1], q_starts[1:])]
ax.set_xticks(mids)
ax.set_xticklabels([f"Q{m.quarter}" for m in mids], fontsize=8.5, color=MUTED)
ax.tick_params(axis="x", length=0, pad=4)
for y in range(X0.year, X1.year):
    ax.text(pd.Timestamp(f"{y}-07-01"), -0.075, str(y), transform=ax.get_xaxis_transform(),
            ha="center", va="top", fontsize=11, color=INK, fontweight="bold")
ax.set_xlim(X0, X1)

# y axis
ax.set_ylim(-1.18, 1.18)
ax.set_yticks([-1, -0.5, 0, 0.5, 1])
ax.set_yticklabels(["−1.0\nnegative", "−0.5", "0", "+0.5", "+1.0\npositive"], fontsize=9, color=INK2)
for yv in [-1, -0.5, 0.5, 1]:
    ax.axhline(yv, color=GRID, lw=0.7, zorder=0)
ax.axhline(0, color=INK2, lw=1.0, zorder=1)
ax.set_ylabel("Lead-paragraph sentiment", fontsize=10.5, color=INK2, labelpad=8)
for side in ["top", "right"]:
    ax.spines[side].set_visible(False)
for side in ["left", "bottom"]:
    ax.spines[side].set_color(AXIS)
ax.tick_params(axis="y", length=0)

# the bubbles: biggest drawn first so small ones stay visible on top
pts = ai.sort_values("health_score", ascending=False)
ax.scatter(pts["date"], pts["article_sentiment"], s=size(pts["health_score"]),
           c=np.where(pts["is_health"], C_H, C_O), alpha=0.78, edgecolors=SURF, linewidths=0.9, zorder=3)

# ChatGPT launch marker
ax.axvline(CHATGPT, color=INK2, lw=1.0, zorder=2)
ax.text(CHATGPT - pd.Timedelta(days=12), 1.13, "ChatGPT released\nNov 30, 2022", ha="right", va="top",
        fontsize=9, color=INK2, linespacing=1.25)

# per-quarter strip above the plot: how many AI articles, what share were about health
trans = ax.get_xaxis_transform()
ax.text(X0 - pd.Timedelta(days=20), 1.075, "AI articles", transform=trans, ha="right", fontsize=8.5, color=MUTED)
ax.text(X0 - pd.Timedelta(days=20), 1.025, "% about health", transform=trans, ha="right", fontsize=8.5, color=MUTED)
for m in mids:
    key = f"{m.year}Q{m.quarter}"
    if key in qt.index:
        ax.text(m, 1.075, f"{int(qt.loc[key, 'n'])}", transform=trans, ha="center", fontsize=8.5, color=INK2)
        ax.text(m, 1.025, f"{qt.loc[key, 'health_share']:.0%}", transform=trans, ha="center", fontsize=8.5,
                color=INK)

# annotate a few of the most health-related AI articles, labels sit in the quiet middle band
lab = ai[ai["is_health"]].sort_values("health_score", ascending=False)
picks = pd.concat([lab[lab["article_sentiment"] < 0].head(N_LABELS - N_LABELS // 3),
                   lab[lab["article_sentiment"] >= 0].head(N_LABELS // 3)]) if N_LABELS else lab.head(0)
days_per_inch = (X1 - X0).days / (ax.get_position().width * fig.get_figwidth())
W = int(2.45 * days_per_inch)                           # a label is ~2.4 inches wide, in days on the x axis
placed = []                                             # (x, y) of labels already drawn
for _, r in picks.sort_values("date").iterrows():
    sign = 1 if r["article_sentiment"] >= 0 else -1
    for shift, lv in [(0, 0), (-W, 0), (W, 0), (0, 1), (-W, 1), (W, 1)]:   # try spots until one is free
        tx = r["date"] + pd.Timedelta(days=shift)
        tx = min(max(tx, X0 + pd.Timedelta(days=W // 2)), X1 - pd.Timedelta(days=W // 2))
        ty = sign * (0.33 - 0.21 * lv)
        if not any(abs((tx - px).days) < W and abs(ty - py) < 0.2 for px, py in placed):
            break
    placed.append((tx, ty))
    head = textwrap.shorten(str(r["headline"]), width=66, placeholder="…")
    ax.annotate(textwrap.fill(f"“{head}”", 36), xy=(r["date"], r["article_sentiment"]), xytext=(tx, ty),
                ha="center", va="center", fontsize=8.3, color=INK2, fontstyle="italic", zorder=5,
                bbox=dict(boxstyle="round,pad=0.35", fc=SURF, ec=GRID, lw=0.6),
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=0, shrinkB=5))

# legends (in the empty pre-ChatGPT years)
color_handles = [
    Line2D([], [], ls="", marker="o", ms=12, mfc=C_H, mec=SURF, alpha=0.85,
           label=f"About health & wellness (score ≥ {THRESHOLD})"),
    Line2D([], [], ls="", marker="o", ms=6.5, mfc=C_O, mec=SURF, alpha=0.85, label="Other AI coverage"),
]
leg1 = ax.legend(handles=color_handles, loc="upper left", bbox_to_anchor=(0.0, 0.985), frameon=False,
                 fontsize=9.5, labelcolor=INK, handletextpad=0.6, borderaxespad=0.6)
ax.add_artist(leg1)
size_handles = [ax.scatter([], [], s=size(v), color=MUTED, alpha=0.55, edgecolors=SURF, label=f"{v:.1f}")
                for v in (0.1, 0.5, 0.9)]
ax.legend(handles=size_handles, title="Bubble area = health score (0–1)", loc="upper left",
          bbox_to_anchor=(0.0, 0.83), frameon=False, fontsize=9, title_fontsize=9, labelcolor=INK2,
          ncol=3, columnspacing=1.6, handletextpad=0.4, borderaxespad=0.6)

lines = [f"Across {len(ai):,} AI articles",
         f"•  {S_ai['n_h']} scored ≥ {THRESHOLD} on health (highest: {ai['health_score'].max():.2f})",
         f"•  health score vs. sentiment: Spearman ρ = {fmt(rho)}",
         f"    (permutation p = {p_perm:.3f})"]
if len(q_ok) >= 5 and S_ai["n_h"] > 0:               # a quarterly correlation needs enough quarters (and health articles)
    lines += ["•  per quarter, avg. health score vs. avg. sentiment:",
              f"    ρ = {fmt(rho_q)} across {len(q_ok)} quarters (n ≥ 5 each)"]
lines += [f"•  health model vs. NYT editors' own tags: AUC = {val_auc:.2f}"]
key_numbers = "\n".join(lines)
ax.text(0.012, 0.035, key_numbers, transform=ax.transAxes, ha="left", va="bottom", fontsize=9, color=INK2,
        linespacing=1.45, zorder=2, bbox=dict(boxstyle="round,pad=0.6", fc="#f3f2ee", ec="none"))  # bubbles stay on top

# ---------- right panel: average sentiment with 95% bootstrap CIs ----------
for side in ["top", "right", "left"]:
    axs.spines[side].set_visible(False)
axs.spines["bottom"].set_color(AXIS)
axs.tick_params(axis="y", left=False, labelleft=False)
for yv in [-1, -0.5, 0.5, 1]:
    axs.axhline(yv, color=GRID, lw=0.7, zorder=0)
axs.axhline(0, color=INK2, lw=1.0, zorder=1)

for gx, S in [(0, S_ai), (1.4, S_base)]:
    for dx, m, ci, n, col in [(-0.17, S["m_h"], S["ci_h"], S["n_h"], C_H), (0.17, S["m_o"], S["ci_o"], S["n_o"], C_O)]:
        if n == 0:
            axs.text(gx + dx, -0.45, f"no articles\nscored ≥ {THRESHOLD}", ha="center", va="center",
                     fontsize=8, color=MUTED, fontstyle="italic", linespacing=1.3)
            continue
        axs.vlines(gx + dx, ci[0], ci[1], color=col, lw=2.4, zorder=3, capstyle="round")
        axs.scatter([gx + dx], [m], s=80, color=col, edgecolors=SURF, linewidths=1.6, zorder=4)
        tx, ha = (gx + dx - 0.07, "right") if dx < 0 else (gx + dx + 0.07, "left")   # labels face outward
        axs.text(tx, m + 0.012, fmt(m), va="bottom", ha=ha, fontsize=9, color=INK)
        axs.text(tx, m - 0.012, f"n={n:,}", va="top", ha=ha, fontsize=7.5, color=MUTED)
    if np.isnan(S["gap"]):
        continue
    top = np.nanmax([S["ci_h"][1], S["ci_o"][1]]) + 0.1
    axs.plot([gx - 0.17, gx - 0.17, gx + 0.17, gx + 0.17], [top - 0.04, top, top, top - 0.04], color=INK2, lw=0.9)
    axs.text(gx, top + 0.035, f"gap {fmt(S['gap'])}", ha="center", va="bottom", fontsize=9.5, color=INK,
             fontweight="bold")

axs.set_xlim(-0.72, 2.12)
axs.set_xticks([0, 1.4])
axs.set_xticklabels(["AI articles", "Other NYT\narticles*"], fontsize=9.5, color=INK)
axs.tick_params(axis="x", length=0, pad=6)
axs.set_title("Average sentiment, 95% CI", loc="center", fontsize=10.5, color=INK, pad=8)
did_txt = ("AI gap minus NYT-wide gap\n" + (f"{fmt(did)}   [{fmt(did_ci[0])}, {fmt(did_ci[1])}]" if not np.isnan(did)
             else "n/a (no AI health articles)"))
axs.text(0.5, 0.975, did_txt,
         transform=axs.transAxes, ha="center", va="top", fontsize=9, color=INK2, linespacing=1.4)

# ---------- titles and footnote ----------
subtitle = (f"Each bubble is one NYT article that mentions GPT models ({len(ai):,} of {n_ai_total:,} matched to Kaggle "
            f"NYT metadata by an inner join on URL). Bubble area = a zero-shot model's probability that the article is "
            f"about health & wellness (medicine, public health, mental health, wellness). Higher = more positive lead "
            f"paragraph. Right panel: health vs. other articles, within AI coverage and within NYT coverage overall.")
footnote = (f"* {len(base):,} randomly sampled non-AI NYT articles from the same quarters, scored by the same model.  "
            f"Health score: {df['model'].iloc[0] if 'model' in df else 'zero-shot NLI'} on headline + abstract + lead "
            f"paragraph.  Sentiment: lead-paragraph score from news_df_sentiment.  CIs: {N_BOOT:,} bootstrap resamples.")
fig.text(0.075, 0.962, verdict, fontsize=17, fontweight="bold", color=INK, ha="left", va="top")
fig.text(0.075, 0.915, textwrap.fill(subtitle, 158), fontsize=10.5, color=INK2, ha="left", va="top", linespacing=1.4)
fig.text(0.075, 0.012, textwrap.fill(footnote, 185), fontsize=8.5, color=MUTED, ha="left", va="bottom", linespacing=1.3)

# ---------- numbers for the write-up ----------
print(f"""
=================== NUMBERS FOR YOUR WRITE-UP ===================
[A] AI articles in news_df_sentiment ............ {n_ai_total:,}
[B] AI articles kept by the inner join .......... {len(ai):,}  ({len(ai) / n_ai_total:.0%})
[C] AI articles about health (score >= {THRESHOLD}) .. {S_ai['n_h']:,}  ({S_ai['n_h'] / max(len(ai), 1):.0%})
[D] mean sentiment, health AI articles .......... {S_ai['m_h']:+.2f}  (95% CI {S_ai['ci_h'][0]:+.2f} to {S_ai['ci_h'][1]:+.2f})
[E] mean sentiment, other AI articles ........... {S_ai['m_o']:+.2f}  (95% CI {S_ai['ci_o'][0]:+.2f} to {S_ai['ci_o'][1]:+.2f})
[F] health gap within AI coverage (D - E) ....... {S_ai['gap']:+.2f}  (95% CI {gap_ci[0]:+.2f} to {gap_ci[1]:+.2f})
[G] health gap within other NYT coverage ........ {S_base['gap']:+.2f}  (95% CI {base_ci[0]:+.2f} to {base_ci[1]:+.2f})
[H] AI-specific extra gap (F - G) ............... {did:+.2f}  (95% CI {did_ci[0]:+.2f} to {did_ci[1]:+.2f})
[I] Spearman rho, health score vs sentiment ..... {rho:+.2f}  (permutation p = {p_perm:.3f})
[J] quarterly rho, avg health vs avg sentiment .. {rho_q:+.2f}  ({len(q_ok)} quarters with >= 5 AI articles)
[K] validation AUC vs NYT editorial health tags . {val_auc:.2f}
[M] highest health score of any AI article ...... {ai['health_score'].max():.3f}  (mean {ai['health_score'].mean():.3f})
[L] baseline articles kept by the inner join .... {len(base):,}
Headline chosen: {verdict}
""")
print(qt.round(2).to_string())

# Remember to comment out this code for step 5
# Until then, do not edit this code
plt.savefig("final_viz.png", dpi=300, bbox_inches="tight")
