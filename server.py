from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd
import spacy
from textblob import TextBlob
from scipy.stats import norm
import yake


app = Flask(__name__)
CORS(app)

# load models and data once
nlp = spacy.load("en_core_web_sm")
kw_extractor = yake.KeywordExtractor(lan="en", top=10)
df = pd.read_csv("Amazon_Reviews.csv")

important_keywords = {'quality', 'material', 'size', 'fit', 'worth', 'value', 'recommend', 'again'}

def clean_text(text):
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

def analyze_sentiment(text):
    blob = TextBlob(text)
    return abs(blob.sentiment.polarity), blob.sentiment.subjectivity

def extract_keywords(text):
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def classify_usage(text):
    text = text.lower()
    if "return" in text or "returned" in text:
        return "returned"
    elif "use" in text or "used" in text or "wear" in text:
        return "used"
    else:
        return "unknown"

def normalize(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-8)

def wilson_lower_bound(pos, total, confidence=0.95):
    if total == 0:
        return 0
    z = norm.ppf(1 - (1 - confidence) / 2)
    phat = pos / total
    return (phat + z*2 / (2 * total) - z * ((phat * (1 - phat) + z*2 / (4 * total)) / total)*0.5) / (1 + z*2 / total)

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

@app.route('/api/top_reviews')
def top_reviews():
    product_id = request.args.get('productID')
    if not product_id:
        return jsonify({'error': 'Missing productID'}), 400

    filtered_df = df[df['product_id'] == product_id]

    if filtered_df.empty:
        return jsonify([])

    filtered_df['clean_review'] = filtered_df['review_body'].fillna("").apply(clean_text)
    filtered_df['review_length'] = filtered_df['clean_review'].apply(lambda x: len(x.split()))
    filtered_df['rating'] = filtered_df['star_rating']
    filtered_df['vote'] = filtered_df['helpful_votes']
    filtered_df['used_returned'] = filtered_df['clean_review'].apply(classify_usage)
    filtered_df['user_rank'] = filtered_df['vote'].apply(lambda x: 1 if x >= 10 else 0)
    filtered_df[['polarity', 'subjectivity']] = filtered_df['clean_review'].apply(lambda x: pd.Series(analyze_sentiment(x)))
    filtered_df['review_keywords'] = filtered_df['clean_review'].apply(extract_keywords)

    filtered_df['keyword_relevance'] = filtered_df['review_keywords'].apply(
        lambda kws: len(set(kws) & important_keywords) / len(important_keywords) if kws else 0
    )

    filtered_df['norm_review_length'] = normalize(filtered_df['review_length'])
    filtered_df['norm_rating'] = normalize(filtered_df['rating'])
    filtered_df['used_score'] = filtered_df['used_returned'].apply(lambda x: 1 if x == "used" else 0)
    filtered_df['pos_votes'] = filtered_df['vote']
    filtered_df['total_votes'] = filtered_df['vote'] + 1
    filtered_df['wilson_score'] = filtered_df.apply(lambda x: wilson_lower_bound(x['pos_votes'], x['total_votes']), axis=1)
    filtered_df['final_score'] = filtered_df.apply(calc_final_score, axis=1)

    top_reviews = filtered_df.sort_values(by='final_score', ascending=False).head(5)

    result = [
        {
            'review': row['review_body'],
            'score': round(row['final_score'], 3),
            'rating': int(row['rating']),
            'votes': int(row['vote'])
        }
        for _, row in top_reviews.iterrows()
    ]

    return jsonify(result)

if __name__ == '__main__':
    app.run(port=5000)
