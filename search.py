# search.py
import pandas as pd
import spacy
import yake
from sentence_transformers import SentenceTransformer, util

# load models
nlp = spacy.load("en_core_web_sm")
kw_extractor = yake.KeywordExtractor(lan="en", top=10)
bert_model = SentenceTransformer('all-MiniLM-L6-v2')

def clean_text(text):
    doc = nlp(str(text).lower())
    tokens = [token.lemma_ for token in doc if not token.is_stop and not token.is_punct and not token.is_space]
    return " ".join(tokens)

def extract_keywords(text):
    keywords = kw_extractor.extract_keywords(text)
    return [kw[0] for kw in keywords]

def keyword_overlap_score(title_kws, query_kws):
    if not title_kws or not query_kws:
        return 0
    return len(set(title_kws) & set(query_kws)) / len(set(query_kws))

def search_products(user_query, df):
    df['product_title'] = df['product_title'].fillna("")
    df['review_headline'] = df['review_headline'].fillna("")
    df['review_body'] = df['review_body'].fillna("")

    df['combined_text'] = df['product_title'] + " " + df['review_headline'] + " " + df['review_body']
    df['clean_combined_text'] = df['combined_text'].apply(clean_text)
    combined_embeddings = bert_model.encode(df['clean_combined_text'].tolist(), convert_to_tensor=True)

    df['title_keywords'] = df['product_title'].apply(lambda x: extract_keywords(x.lower()))
    clean_query = clean_text(user_query)
    query_keywords = extract_keywords(user_query.lower())
    query_embedding = bert_model.encode(clean_query, convert_to_tensor=True)

    product_scores = util.cos_sim(query_embedding, combined_embeddings)[0]
    df['product_similarity'] = product_scores.cpu().numpy()
    df['keyword_match'] = df['title_keywords'].apply(lambda x: keyword_overlap_score(x, query_keywords))

    df['combined_score'] = 0.7 * df['product_similarity'] + 0.3 * df['keyword_match']

    top_products = df.sort_values(by='combined_score', ascending=False).drop_duplicates(subset='product_title').head(3)
    return top_products[['product_title', 'combined_score']]

if __name__ == "__main__":
    df = pd.read_csv("Amazon_Reviews.csv").head(1000).copy()
    query = input("Enter your query (product name or description): ")
    results = search_products(query, df)

    if results.empty:
        print("No products found.")
    else:
        print("\nTop matching products:")
        for i, row in results.iterrows():
            print(f"{i+1}. {row['product_title']} (score: {row['combined_score']:.3f})")
