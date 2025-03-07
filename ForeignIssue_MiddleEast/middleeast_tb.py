import pandas as pd
import numpy as np
from textblob import TextBlob
from scipy import stats
from collections import Counter
import re
import matplotlib.pyplot as plt

# ---------------------------------------
# 1. Load Data
# ---------------------------------------
df = pd.read_csv('womens_rights_combined_data.csv', encoding='utf-8')

# Combine the context fields into one text field
df['full_text'] = df['Context before'].astype(str) + " " + df['Query item'].astype(str) + " " + df['Context after'].astype(str)
df['full_text'] = df['full_text'].apply(lambda x: re.sub(r'\s+', ' ', x).strip())

# ---------------------------------------
# 2. Sentiment Analysis (TextBlob)
# ---------------------------------------
def get_textblob_score(text):
    return TextBlob(text).sentiment.polarity

df['sentiment_score'] = df['full_text'].apply(get_textblob_score)

# ---------------------------------------
# 3. Aggregate Sentiment by Source
# ---------------------------------------
source_groups = df.groupby('Subgenre')['sentiment_score']
sentiment_stats = source_groups.agg(['mean', 'count', 'std'])
sentiment_stats['sem'] = sentiment_stats['std'] / np.sqrt(sentiment_stats['count'])
sentiment_stats['ci95_lower'] = sentiment_stats['mean'] - 1.96 * sentiment_stats['sem']
sentiment_stats['ci95_upper'] = sentiment_stats['mean'] + 1.96 * sentiment_stats['sem']

print("Sentiment Statistics by Source (All Queries Combined):")
print(sentiment_stats)

# ---------------------------------------
# 4. Data Imbalance Handling (Optional)
# ---------------------------------------
min_count = sentiment_stats['count'].min()
balanced_df = df.groupby('Subgenre').apply(lambda x: x.sample(min_count, random_state=42) if len(x) > min_count else x).reset_index(drop=True)

balanced_groups = balanced_df.groupby('Subgenre')['sentiment_score']
balanced_stats = balanced_groups.agg(['mean','count','std'])
print("\nBalanced Sample Statistics by Source:")
print(balanced_stats)

# ---------------------------------------
# 5. Statistical Testing (ANOVA)
# ---------------------------------------
source_list = [g.dropna().values for _, g in source_groups]
if len(source_list) > 1:
    f_stat, p_val = stats.f_oneway(*source_list)
    print("\nANOVA test across all sources:")
    print("F-statistic:", f_stat)
    print("p-value:", p_val)

# ---------------------------------------
# 6. Keyword Analysis (Optional)
# ---------------------------------------
q_low = df['sentiment_score'].quantile(0.25)
q_high = df['sentiment_score'].quantile(0.75)
low_sent_df = df[df['sentiment_score'] <= q_low]
high_sent_df = df[df['sentiment_score'] >= q_high]

def tokenize(text):
    tokens = re.findall(r'\w+', text.lower())
    return tokens

low_tokens = [tok for text in low_sent_df['full_text'] for tok in tokenize(text)]
high_tokens = [tok for text in high_sent_df['full_text'] for tok in tokenize(text)]

low_freq = Counter(low_tokens)
high_freq = Counter(high_tokens)

print("\nTop 20 words in low sentiment lines:")
print(low_freq.most_common(20))

print("\nTop 20 words in high sentiment lines:")
print(high_freq.most_common(20))

# Example chi-square test on a single word:
test_word = "crime"
low_word_count = low_freq[test_word]
high_word_count = high_freq[test_word]
low_total = sum(low_freq.values())
high_total = sum(high_freq.values())

cont_table = np.array([
    [low_word_count, low_total - low_word_count],
    [high_word_count, high_total - high_word_count]
])

chi2, chi_p, dof, expected = stats.chi2_contingency(cont_table)
print(f"\nChi-square test for word '{test_word}' between low and high sentiment:")
print("Chi2:", chi2)
print("p-value:", chi_p)
print("Degrees of freedom:", dof)
print("Expected counts:", expected)

# ---------------------------------------
# 7. Optional Visualizations (Distribution)
# ---------------------------------------
# We won't do the boxplot here, as we're going to create a media-bias-chart-like plot below.

# ---------------------------------------
# 8. Media Bias Chart-Style Visualization
# ---------------------------------------
# We'll plot each source as a single point on the x-axis (sentiment score),
# and place them at different y positions just for visualization.
# We'll then annotate each point with the source name.

sources = sentiment_stats.index.tolist()
means = sentiment_stats['mean'].values

# Determine a suitable y-position for each source (just to separate them visually)
y_positions = np.arange(len(sources))

fig, ax = plt.subplots(figsize=(10, 6))

# Scatter plot: sentiment means on x-axis, sources spaced out on y
ax.scatter(means, y_positions, color='blue')

# Add annotations: source names next to each point
for i, source in enumerate(sources):
    ax.text(means[i] + 0.001, y_positions[i], source, va='center', fontsize=9)

# Dynamically set x-limits based on data to only show relevant range
x_min = min(means)
x_max = max(means)
padding = (x_max - x_min) * 0.1 if x_max != x_min else 0.1
ax.set_xlim(x_min - padding, x_max + padding)

ax.set_xlabel("Average Sentiment Score (Polarity)")
ax.set_ylabel("Sources")
ax.set_title("Media Bias Chart - Sentiment by Source")

# Hide y-axis ticks and labels since we annotated the points directly
ax.yaxis.set_visible(False)

plt.tight_layout()
plt.show()

# ---------------------------------------
# 9. Selected Sources vs. Bias Scores (Extended)
# ---------------------------------------
# Bias scores for FOX, CNN, PBS, ABC, CBS, MSNBC, NBC, NPR:
bias_map = {
    'SPOKFOX': 11.04,
    'SPOKCNN': -6.22,
    'SPOKPBS': -4.05,
    'SPOKABC': -2.94,
    'SPOKCBS': -2.77,
    'SPOKMSNBC': -13.99,
    'SPOKNBC': -5.70,
    'SPOKNPR': -4.18
}

selected_sources = ['SPOKFOX', 'SPOKCNN', 'SPOKPBS', 'SPOKABC', 'SPOKCBS', 'SPOKMSNBC', 'SPOKNBC', 'SPOKNPR']
filtered_stats = sentiment_stats.loc[sentiment_stats.index.isin(selected_sources)]
filtered_stats['bias_score'] = filtered_stats.index.map(bias_map)

x = filtered_stats['mean']
y = filtered_stats['bias_score']

# Pearson correlation
corr = x.corr(y)

# Kendall's Tau correlation
from scipy.stats import kendalltau, spearmanr
tau_corr, tau_p = kendalltau(x, y)

# Spearman Rank correlation
spearman_corr, spearman_p = spearmanr(x, y)

print("Pearson Correlation between sentiment and bias:", corr)
print("Kendall Tau correlation:", tau_corr, "p-value:", tau_p)
print("Spearman Rank correlation:", spearman_corr, "p-value:", spearman_p)

# Scatter plot with line of best fit
plt.figure(figsize=(8, 6))
plt.scatter(x, y, color='blue')

for source in selected_sources:
    plt.text(filtered_stats.loc[source, 'mean'] + 0.001,
             filtered_stats.loc[source, 'bias_score'],
             source, va='center', fontsize=9)

# Line of best fit via polyfit (for visualization)
m, b = np.polyfit(x, y, 1)
plt.plot(x, m * x + b, color='red', linestyle='--', 
         label=(f'Line of Best Fit\nPearson r={corr:.2f}\n'
                f'Kendall τ={tau_corr:.2f}\n'
                f'Spearman ρ={spearman_corr:.2f}'))

plt.xlabel("Average Sentiment Score (Mean)")
plt.ylabel("Bias Score")
plt.title("Sentiment vs. Bias for Selected Sources")
plt.legend()
plt.tight_layout()
plt.show()

# Residual Analysis with a Linear Regression Model
import statsmodels.api as sm
X = sm.add_constant(x)
model = sm.OLS(y, X).fit()
residuals = model.resid
pred_y = model.predict(X)

print(model.summary())

# Plot residuals
plt.figure(figsize=(8, 6))
plt.scatter(x, residuals, color='green')
plt.axhline(0, color='red', linestyle='--')
plt.xlabel("Average Sentiment Score (Mean)")
plt.ylabel("Residuals (Actual Bias - Predicted Bias)")
plt.title("Residual Analysis (Sentiment Predicting Bias)")
plt.tight_layout()
plt.show()
