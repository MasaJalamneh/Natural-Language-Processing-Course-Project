import pandas as pd
from textblob import TextBlob
from collections import Counter
import os

print("جارٍ قراءة الملف...")
print("الموقع الحالي:", os.getcwd())
print("محتويات المجلد:", os.listdir())

df = pd.read_csv("Amazon_reviews.csv")
print("تمت قراءة الملف، عدد الصفوف:", len(df))
print("أسماء الأعمدة:", df.columns.tolist())
print("أول 3 صفوف:")
print(df.head(3))

df = df.dropna(subset=['customer_id', 'review_body', 'star_rating', 'helpful_votes'])

df['review_length'] = df['review_body'].apply(lambda x: len(str(x).split()))
df['sentiment'] = df['review_body'].apply(lambda x: TextBlob(str(x)).sentiment.polarity)

local_stopwords = set("""a about above after again against all am an and any are aren't ...""".split())

df['keywords'] = df['review_body'].apply(
    lambda text: [word.lower() for word in str(text).split() if word.lower() not in local_stopwords]
)

df['reviewed_after_purchase'] = df['verified_purchase'].apply(lambda x: 1 if str(x).strip().upper() == 'Y' else 0)

def profile_user(group):
    all_keywords = [word for kw_list in group['keywords'] for word in kw_list]
    top_keywords = [word for word, _ in Counter(all_keywords).most_common(10)]
    
    return pd.Series({
        'total_ratings': group.shape[0],
        'avg_star_rating': group['star_rating'].mean(),
        'avg_helpful_votes': group['helpful_votes'].mean(),
        'avg_sentiment': group['sentiment'].mean(),
        'avg_review_length': group['review_length'].mean(),
        'products_reviewed': group['product_id'].nunique(),
        'percent_reviewed_after_purchase': group['reviewed_after_purchase'].mean(),
        'top_keywords': ", ".join(top_keywords)
    })

user_profiles = df.groupby('customer_id').apply(profile_user).reset_index()

print("نماذج من نتائج التحليل:")
print(user_profiles.head())

user_profiles['password'] = [f"u_{i+1}" for i in range(len(user_profiles))]

user_profiles.to_csv("user_profiles.csv", index=False)
print("تم إنشاء ملف user_profiles.csv بنجاح!")
