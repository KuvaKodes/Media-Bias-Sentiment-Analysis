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

## Optional Political-Leaning Classifier

The independent `political_leaning` module grades an English passage using
[`matous-volf/political-leaning-deberta-large`](https://huggingface.co/matous-volf/political-leaning-deberta-large).
It returns `left`, `center`, or `right`, the winning softmax score (`confidence`),
and all three class probabilities. These are model estimates, not calibrated
certainty, sentiment scores, or a measure of factual accuracy. Labels reflect
U.S. political context. The model does not detect whether text is political;
the authors recommend filtering nonpolitical inputs first. Model weights are
licensed CC BY-NC 4.0; consult the linked model card for terms and attribution.

From the repository root, use Python 3.10+ in a virtual environment:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-political-leaning.txt
python -m political_leaning --text "The government should increase public spending."
python -m political_leaning --file passage.txt
cat passage.txt | python -m political_leaning --stdin
```

The first call downloads the large model and its tokenizer into Hugging Face's
normal cache outside the project; allow several GB of disk space and RAM.
Inference runs locally on CPU by default; use `--device cuda:0` for a compatible
GPU. Input text is not uploaded to an inference service or saved by this module.
Dependencies are optional and do not affect the existing sentiment scripts.

```python
from political_leaning import PoliticalLeaningClassifier

classifier = PoliticalLeaningClassifier()  # Reuse across passages.
result = classifier.grade("The government should increase public spending.")
print(result.to_dict())
```

Inputs exceeding 512 tokens (including special tokens) raise an error. To grade
only the initial portion, opt in with `--truncate` or `grade(text, truncate=True)`.
The result reports `input_tokens` and `truncated`; no automatic chunk averaging
is applied. Empty text is rejected. The CLI emits JSON to stdout and errors to
stderr with exit code 2.

Run lightweight wrapper tests without downloading model weights:

```sh
python -m unittest discover -s tests -v
```

These tests use a simulated model backend; run a CLI example after installing
dependencies to verify actual model inference on your machine.

## Full Paper

The complete write-up, including background, full methodology, results, and discussion, is available in **`KRV14FinalPaper.pdf`**.
