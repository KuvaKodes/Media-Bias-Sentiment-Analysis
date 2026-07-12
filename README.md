# Quantifying Media Bias: Sentiment Analysis and Corpus Linguistics in Televised News Discourse

**Author:** Kushaan Vardhan
**Department:** Computational Linguistics, Georgetown University

## Overview

This repository contains the data, analysis code, and write-up for a corpus-linguistics study of sentiment in U.S. televised news discourse. The project examines whether the sentiment expressed around politically charged topics in TV news transcripts correlates with independently rated measures of network bias.

## Data

The underlying transcript data comes from the **COCA Spoken corpus** (Corpus of Contemporary American English), covering TV news broadcasts from **1990 to 2019** across eight networks: **FOX, CNN, PBS, ABC, CBS, MSNBC, NBC, and NPR**. Concordance files were pulled for query terms within three topic domains and are organized into the corresponding directories:

- `DomesticIssue_GunReform/` — gun reform / gun control (e.g., assault ban, Brady Bill, NRA, Second Amendment, mass shootings)
- `ForeignIssue_MiddleEast/` — Middle East and foreign policy (e.g., Iraq/Gulf wars, Afghanistan, Taliban, al-Qaeda, ISIS/ISIL, WMDs)
- `SocialIssue_WomensRights/` — women's rights (e.g., abortion, equal pay, feminism, #MeToo, reproductive rights, Title IX)

Each directory contains the raw per-query concordance `.txt` files exported from COCA, along with a combined `.csv` dataset merging those concordances for analysis.

Network-level bias ratings used for correlation come from **Ad Fontes Media**.

## Methodology

1. **Corpus assembly** — raw COCA concordance exports for each query term are merged into a single combined CSV per topic domain.
2. **Sentiment scoring** — each concordance line (context before + query item + context after) is scored using two independent sentiment tools:
   - **TextBlob** (polarity scoring)
   - **VADER** (valence-aware scoring, tuned for informal/spoken text)
3. **Aggregation** — sentiment scores are aggregated by network/source, with summary statistics (mean, standard deviation, standard error, 95% confidence intervals).
4. **Correlation with bias ratings** — aggregated network-level sentiment is compared against Ad Fontes Media bias ratings using Pearson, Kendall, and Spearman correlation coefficients, along with OLS regression.
5. **Follow-up lexical analysis** — a lexical/collocational pass examines word choice and collocates around key query terms to complement the quantitative sentiment scores.

Results are visualized in the plots included at the repository root (`Figure_1.png`, `Figure_2.png`, `GunControl.png`, `MiddleEast.png`, `WomensRights.png`).

## Running the Scripts

Each topic directory follows the same general pipeline:

1. **Merge/convert raw concordance data** — scripts such as `*_csv_to_txt.py` / `*_txt_to_csv.py` (naming varies slightly by domain, e.g. `gun_control_sentiment_txt_to_csv.py`, `middleeast_csv_to_txt.py`, `womensrights_txt_to_csv.py`) combine the raw per-query concordance files into the domain's combined CSV dataset.
2. **Run sentiment analysis** — the `*_tb.py` scripts run TextBlob-based sentiment scoring, and the `*_vd.py` scripts run VADER-based sentiment scoring, on the combined dataset. Each produces aggregate statistics and generates the corresponding plots.

Scripts are run standalone from within their topic directory (e.g. `python middleeast_tb.py` from inside `ForeignIssue_MiddleEast/`) and depend on `pandas`, `numpy`, `scipy`, `textblob`, `vaderSentiment`, `statsmodels`, and `matplotlib`.

## Full Paper

The complete write-up, including background, full methodology, results, and discussion, is available in **`KRV14FinalPaper.pdf`**.
