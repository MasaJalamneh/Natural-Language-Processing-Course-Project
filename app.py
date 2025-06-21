from flask import Flask, render_template, request, jsonify
import pandas as pd
from search import search_products

app = Flask(__name__)

# load data once at startup
df = pd.read_csv("Amazon_Reviews.csv").head(1000).copy()

@app.route('/')
def home():
    return render_template("home.html")

@app.route('/search', methods=['POST'])
def search():
    data = request.get_json()
    query = data.get("query", "")
    if not query.strip():
        return jsonify([])

    results = search_products(query, df)
    result_list = results.to_dict(orient='records')
    return jsonify(result_list)

if __name__ == '__main__':
    app.run(debug=True)
