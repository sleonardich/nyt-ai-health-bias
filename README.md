# Does The New York Times Exhibit Bias? Testing AI Coverage Against Health/Cognitive-Decline Framing

Coursework project (UC Berkeley, Data 100) extended into an independent investigation.

## Question

An earlier assignment showed NYT sentiment toward "GPT Model" coverage falling from strongly
positive in 2020 to negative after ChatGPT's late-2022 release. This project tests one specific
hypothesis for that drop: that the negativity is concentrated in articles that frame AI in terms
of health and cognitive decline (e.g. "AI is rotting our brains"), rather than being a general
property of AI coverage.

**Bias, as defined here:** tone that tracks a topic's framing (health/cognitive-decline) rather
than the underlying facts of the AI news itself. If the hypothesis held, AI articles about health
should read more negatively than other AI articles, by more than health news is already negative
on average, and quarters with more health talk should show lower average AI sentiment.

## Method

1. **Join.** Inner-joined the course's `news_df_sentiment` dataset (articles flagged by a
   `GPT Model` mention count) to the Kaggle
   [NYT Articles: 2.1M+ (2000–Present)](https://www.kaggle.com/datasets/aryansingh0909/nyt-articles-21m-2000-present)
   metadata table (4.6 GB, CC0) on article URL, to recover headline, abstract, section, news desk,
   and NYT's own subject keywords — none of which are in the course dataset.
2. **Score.** Ran a zero-shot NLI classifier
   ([`MoritzLaurer/deberta-v3-large-zeroshot-v2.0`](https://huggingface.co/MoritzLaurer/deberta-v3-large-zeroshot-v2.0),
   base model on CPU) on each article's headline + abstract + lead paragraph, scoring it 0–1 for
   how much it is about health & wellness (medicine, public health, mental health, wellness).
3. **Validate.** Checked the health score against an independent signal the model never saw: NYT's
   own editorial tags (section = Health/Well, or subject keywords such as
   "Mental Health and Disorders"). Agreement measured by AUC.
4. **Test.** Compared mean sentiment for health-scored vs. other AI articles, with bootstrap 95%
   confidence intervals, against the same gap measured in a random sample of non-AI NYT articles
   from the same quarters (the baseline for "health news is negative anyway"). Also tested the
   correlation between health score and sentiment (Spearman ρ, permutation test).
5. **Visualize.** One figure: each AI article as a bubble (x = date, y = lead-paragraph sentiment,
   size = health score), with a side panel showing the health-vs-other sentiment gap for AI
   coverage and for NYT coverage overall.

## Result

The hypothesis was **not supported** — but not because health-framed AI coverage turned out
positive. The `GPT Model` keyword filter turned out to be too narrow: essentially no AI articles
in the matched sample scored as health-related, because articles about AI's effects on cognition,
mental health, or memory typically say "chatbot" or "A.I." rather than naming GPT specifically.
There was no health-framed AI group to compare, and no health trend to track over time.

What the data does show: NYT's `GPT Model` coverage overall is only mildly negative (mean ≈ −0.1,
95% CI includes zero), while negativity in *health news generally* is large and reliable (health
articles average roughly 0.4–0.5 points more negative than non-health articles, across ~2,400
sampled non-AI articles). That gap is a property of health journalism, not of AI coverage.

**Takeaway:** this is a failed test of the hypothesis, not evidence that NYT's AI coverage is
unbiased. A fair retest would widen the AI filter (e.g. `A\.I\.|artificial intelligence|chatbot|
ChatGPT|OpenAI|GPT` against the lead paragraph, not just a `GPT Model` count column) and fix the
join, which currently drops roughly half of matched articles including most of 2024.

## Repo contents

- `NYT_AI_health_scoring_colab.ipynb` — the Google Colab notebook: downloads the Kaggle metadata,
  performs the inner join, runs the zero-shot scoring, and validates against editorial tags.
  Requires a Kaggle account (free) for the dataset download.
- `datahub/step1_export_cell.py` — exports the AI + baseline article sample from the course
  dataset for upload to Colab.
- `datahub/scratch_cell_1.py`, `datahub/scratch_cell_2.py` — rejoin the Colab output back onto
  the course dataset and produce the final figure.
- `final_viz.png` — the resulting visualization (not included in this template; see note below).

**Not included:** the raw Kaggle metadata CSV (4.6 GB, not mine to redistribute — the notebook
downloads it directly) and the course-provided `news_df_sentiment` dataset (Berkeley Data 100
course material). The notebook and scripts expect these as local inputs.

## Tools

Python (Polars, pandas, NumPy), Hugging Face Transformers (zero-shot NLI classification), Google
Colab (GPU), Kaggle API, Matplotlib, bootstrap resampling, Spearman correlation, ROC/AUC.
