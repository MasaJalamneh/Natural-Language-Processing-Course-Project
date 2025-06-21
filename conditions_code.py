import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import re
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
import numpy as np

"""
NOTE: 
for pandas : pip install pandas
for seaborn: pip install seaborn
for scipy  : pip install scipy
for numpy  : pip install numpy
"""

# Load the CSV file
df = pd.read_csv("C:\\Users\\hp\\Desktop\\NLP_Proj\\Amazon_Reviews.csv", encoding='utf-8-sig')


# Display the shape and the first few rows
print("Shape of the dataset:", df.shape)
print(df.head())



""" Ranking Conditions """

"""1. rating_bias_weight"""
def compute_rating_bias_weight_per_product(group):
    """
    For a group of reviews of one product, return:
    1 → favor high-rated reviews, give high_rated higher score
    -1 → favor low-rated reviews, give low_rated higher score
    0 → neutral, give all same score
    """
    rating_counts = group['star_rating'].value_counts(normalize=True)
    high_ratio = rating_counts.get(3, 0) + rating_counts.get(4, 0) + rating_counts.get(5, 0)
    low_ratio = rating_counts.get(1, 0) + rating_counts.get(2, 0)

    if high_ratio > 0.7:
        return 1
    elif low_ratio > 0.7:
        return -1
    else:
        return 0


def rating_score(row, bias_map):
    rating = row['star_rating']
    product_id = row['product_id']
    bias = bias_map.get(product_id, 0)  # default to neutral

    if bias == 0:
        return rating / 5.0
    elif bias == 1:
        return rating / 5.0
    else:  # bias == -1
        return (6 - rating) / 5.0


print("Available columns:", df.columns.tolist())


# 1: compute bias per product
bias_map = df.groupby('product_id').apply(compute_rating_bias_weight_per_product).to_dict()


# 2: apply rating score for each review
df['score_rating'] = df.apply(lambda row: rating_score(row, bias_map), axis=1)

print(df[['product_id', 'star_rating', 'score_rating']].head(10))


""""--------"""
"""Visualize"""

# convert bias_map to a DataFrame
bias_df = pd.DataFrame(list(bias_map.items()), columns=['product_id', 'bias'])

# count each type of bias
bias_counts = bias_df['bias'].value_counts().sort_index()

# map bias labels for clarity
bias_labels = {-1: 'Negative Bias', 0: 'Neutral Bias', 1: 'Positive Bias'}

# plot
plt.figure(figsize=(8, 5))
sns.barplot(x=bias_counts.index.map(bias_labels), y=bias_counts.values, palette='viridis')

plt.title('Product Rating Bias Distribution')
plt.xlabel('Bias Type')
plt.ylabel('Number of Products')
plt.tight_layout()
plt.show()



""" 2. Votes """


def wilson_score(helpful, total, confidence=0.95):
    if total == 0:
        return 0
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = helpful / total
    return ((phat + z**2 / (2*total) - z * np.sqrt((phat*(1 - phat) + z**2 / (4*total)) / total)) / (1 + z**2 / total)) + 0.4



# 1: compute bias per product
df['wilson_score'] = df.apply(lambda row: wilson_score(row['helpful_votes'], row['total_votes']), axis=1)

# after computing wilson_score for each review
vote_score = df.groupby('product_id')['wilson_score'].mean().to_dict()

print(df[['product_id', 'review_body', 'helpful_votes', 'total_votes', 'wilson_score']].head(30))