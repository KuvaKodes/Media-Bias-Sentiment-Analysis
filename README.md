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

## Optional GLiNER2 Topic Classifier

`topic_classification` assigns one general topic to short English text using
[`fastino/gliner2-base-v1`](https://huggingface.co/fastino/gliner2-base-v1), the
GLiNER 2.0 checkpoint. It is independent of the sentiment analysis scripts and
political-leaning classifier. No training dataset or API key is required.

From the repository root, with Python 3.10+:

```sh
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements-topics.txt
python -m topic_classification --text "Donald Trump spoke at a rally today."
python -m topic_classification --file tweet.txt
cat tweet.txt | python -m topic_classification --stdin
```

The first classification downloads model weights into Hugging Face's cache;
subsequent calls reuse them. Inference runs locally on CPU. The module does not
save or send input text to a hosted inference service. JSON output contains
`topic`, `confidence`, and `model`. Confidence is the model's score, not a
calibrated guarantee of correctness.

```python
from topic_classification import TopicClassifier

classifier = TopicClassifier()  # Reuse for multiple tweets.
result = classifier.classify("A mass shooting has prompted debate on gun laws.")
print(result.to_dict())
```

The default labels are president, gun issues, gender rights, reproductive rights,
immigration, economy, elections, foreign affairs, healthcare, environment,
crime and justice, education, race and civil rights, technology, and other.
Definitions are in `topic_classification/topics.py`. Intended examples include
Donald Trump → president, mass shootings → gun issues, and transgender rights
→ gender rights. These guide the model rather than imposing keyword rules.
Overlapping subjects can receive different labels depending on context; inspect
representative tweets before relying on labels for research.

To replace the taxonomy, pass a dictionary with at least two label/description
pairs to `TopicClassifier(topics=...)`, or save that dictionary as a JSON object
and use `--topics topics.json`. Descriptions help distinguish related categories.
The classifier selects a single topic; `other` handles subjects outside the
provided default taxonomy. It rejects blank inputs and passages over 1,000
characters, and is intended for short tweets rather than whole articles.

```sh
python -m unittest discover -s tests -v
```

Unit tests use a simulated backend and do not download model weights. The library
version in `requirements-topics.txt` is separate from the model's GLiNER 2.0
version. See the [official classification tutorial](https://github.com/fastino-ai/GLiNER2/blob/main/tutorial/1-classification.md)
for schema and confidence-score details.

An optional real-model smoke test checks the three representative topics above:

```sh
RUN_GLINER_MODEL_TESTS=1 python -m unittest discover -s tests -p test_topics_model.py -v
```

These three examples passed with the actual checkpoint during implementation;
they are smoke checks, not an accuracy benchmark.
