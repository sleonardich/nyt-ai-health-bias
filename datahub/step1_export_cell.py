# ===================== STEP 1 · TEMPORARY EXPORT CELL (DataHub) =====================
# Paste this into SCRATCH CELL 2 *temporarily*, run it once, download the CSV it makes,
# then delete this code (Step 5 replaces it with the plotting code).
# It writes nyt_articles_for_scoring.csv (~1-2 MB) next to hw03.ipynb.
import polars as pl

AI_COL     = "GPT Model"   # your regex-count column for AI / GPT coverage
N_BASELINE = 2500          # random NON-AI articles from the same quarters (the "NYT-wide" comparison group)
SEED       = 100
KEEP       = ["web_url", "pub_date", "Quarter", "lead_paragraph"]   # NOTE: sentiment is deliberately NOT exported,
                                                                    # so the health model never sees it

# 1) every AI article
ai_export = (
    news_df_sentiment
    .filter(pl.col(AI_COL) > 0)
    .select(KEEP)
    .unique(subset="web_url", keep="first", maintain_order=True)
    .with_columns(pl.lit("ai").alias("group"))
)

# 2) a random sample of non-AI articles published in the SAME quarters (controls for time)
ai_quarters = ai_export.get_column("Quarter").unique().to_list()
baseline_pool = (
    news_df_sentiment
    .filter((pl.col(AI_COL) == 0) & pl.col("Quarter").is_in(ai_quarters))
    .select(KEEP)
    .unique(subset="web_url", keep="first", maintain_order=True)
)
baseline_export = (
    baseline_pool
    .sample(n=min(N_BASELINE, baseline_pool.height), seed=SEED)
    .with_columns(pl.lit("baseline").alias("group"))
)
del baseline_pool   # free memory right away (DataHub only has 4 GB)

export = pl.concat([ai_export, baseline_export])
export.write_csv("nyt_articles_for_scoring.csv")

print(f"AI articles exported:        {ai_export.height:,}")
print(f"Baseline articles exported:  {baseline_export.height:,}")
print(f"Quarters covered:            {sorted(ai_quarters)}")
print("Saved -> nyt_articles_for_scoring.csv  (right-click it in the file browser > Download)")
