import pandas as pd
import spacy
from textblob import TextBlob
import yake
from sentence_transformers import SentenceTransformer, util
import torch
import html
from scipy.stats import norm
import numpy as np

# load spaCy
nlp = spacy.load("en_core_web_sm")

# initialize YAKE
kw_extractor = yake.KeywordExtractor(lan="en", top=10)

# load data
df = pd.read_csv("Amazon_Reviews.csv")
df_sample = df.head(1000).reset_index(drop=True)

# clean text
def clean_text(text):
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

df_sample['clean_review'] = df_sample['review_body'].fillna("").apply(clean_text)

# sentiment analysis
def analyze_sentiment(text):
    blob = TextBlob(text)
    polarity = abs(blob.sentiment.polarity)
    subjectivity = blob.sentiment.subjectivity
    return polarity, subjectivity

df_sample[['polarity', 'subjectivity']] = df_sample['clean_review'].apply(lambda x: pd.Series(analyze_sentiment(x)))

# keyword extraction
def extract_keywords(text):
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

df_sample['review_length'] = df_sample['clean_review'].apply(lambda x: len(x.split()))
df_sample['user_profile'] = df_sample['customer_id']
df_sample['rating'] = df_sample['star_rating']
df_sample['vote'] = df_sample['helpful_votes']

# usage classification
def classify_usage(text):
    text = text.lower()
    if "return" in text or "returned" in text:
        return "returned"
    elif "use" in text or "used" in text or "wear" in text:
        return "used"
    else:
        return "unknown"

df_sample['used_returned'] = df_sample['clean_review'].apply(classify_usage)
df_sample['user_rank'] = df_sample['vote'].apply(lambda x: 1 if x >= 10 else 0)

# load BERT
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

# fill missing
df_sample['product_title'] = df_sample['product_title'].fillna("")
df_sample['review_headline'] = df_sample['review_headline'].fillna("")
df_sample['review_body'] = df_sample['review_body'].fillna("")

# combined fields
df_sample['combined_text'] = df_sample['product_title'] + " " + df_sample['review_headline'] + " " + df_sample['review_body']
df_sample['clean_combined_text'] = df_sample['combined_text'].apply(clean_text)

# encode with BERT
combined_embeddings = bert_model.encode(df_sample['clean_combined_text'].tolist(), convert_to_tensor=True)

# extract title keywords
df_sample['title_keywords'] = df_sample['product_title'].apply(lambda x: extract_keywords(x.lower()))

# user query
user_query = input("Enter your query (product name or description): ").strip()
clean_query = clean_text(user_query)

# BERT query embedding
query_embedding = bert_model.encode(clean_query, convert_to_tensor=True)
product_scores = util.cos_sim(query_embedding, combined_embeddings)[0]
df_sample['product_similarity'] = product_scores.cpu().numpy()

# keyword match
query_keywords = extract_keywords(user_query.lower())

def keyword_overlap_score(title_kws, query_kws):
    if not title_kws or not query_kws:
        return 0
    return len(set(title_kws) & set(query_kws)) / len(set(query_kws))

df_sample['keyword_match'] = df_sample['title_keywords'].apply(lambda x: keyword_overlap_score(x, query_keywords))

# combined score
df_sample['combined_score'] = 0.7 * df_sample['product_similarity'] + 0.3 * df_sample['keyword_match']

# Top 3 products
top_products = df_sample.sort_values(by='combined_score', ascending=False) \
    .drop_duplicates(subset='product_title') \
    .head(3)

if top_products.empty:
    print("\nNo products found matching your description. Try a different query.")
    exit()

print("\nTop matching products based on your query:")
for idx, (_, row) in enumerate(top_products.iterrows(), start=1):
    print(f"{idx}. {row['product_title']} (score: {row['combined_score']:.3f})")

# let user select product
selected_index = int(input("\nSelect a product number to see top reviews: ")) - 1
selected_product = top_products.iloc[selected_index]['product_title']
product_reviews = df_sample[df_sample['product_title'] == selected_product].copy()

# normalize
def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)

product_reviews['norm_review_length'] = normalize(product_reviews['review_length'])
product_reviews['norm_rating'] = normalize(product_reviews['rating'])
product_reviews['used_score'] = product_reviews['used_returned'].apply(lambda x: 1 if x == "used" else 0)

# keywords
product_reviews['review_keywords'] = product_reviews['clean_review'].apply(extract_keywords)

important_keywords = {'quality', 'material', 'size', 'fit', 'worth', 'value', 'recommend', 'again'}

def keyword_relevance_score(keywords):
    if not keywords:
        return 0
    return len(set(keywords) & important_keywords) / len(important_keywords)

product_reviews['keyword_relevance'] = product_reviews['review_keywords'].apply(keyword_relevance_score)

# wilson score
def wilson_lower_bound(pos, total, confidence=0.95):
    if total == 0:
        return 0
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = pos / total
    return (phat + z**2 / (2 * total) - z * ((phat * (1 - phat) + z**2 / (4 * total)) / total)**0.5) / (1 + z**2 / total)

product_reviews['pos_votes'] = product_reviews['vote']
product_reviews['total_votes'] = product_reviews['vote'] + 1
product_reviews['wilson_score'] = product_reviews.apply(lambda x: wilson_lower_bound(x['pos_votes'], x['total_votes']), axis=1)

# final score with updated weights (calc)
def calc_final_score(row):
    return (
        row['user_rank'] * 0.1 +
        row['norm_review_length'] * 0.1 +
        row['wilson_score'] * 0.1 +
        row['used_score'] * 0.05 +
        row['polarity'] * 0.15 +
        row['subjectivity'] * 0.3 +
        row['keyword_relevance'] * 0.15 +
        row['norm_rating'] * 0.05
    )

product_reviews['final_score'] = product_reviews.apply(calc_final_score, axis=1)

# Top reviews
top_reviews = product_reviews.sort_values(by='final_score', ascending=False).head(5)

print(f"\nTop 5 reviews for selected product: {selected_product}\n")

for i, (_, row) in enumerate(top_reviews.iterrows(), start=1):
    clean_text = html.unescape(row['review_body']).replace('<br />', '\n').replace('<br>', '\n')
    print(f"{i}. Review (score: {row['final_score']:.3f}):\n{clean_text}\n{'-'*80}\n")