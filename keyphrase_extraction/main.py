from keyphrase_vectorizers import KeyphraseCountVectorizer
from keybert import KeyBERT
from flask import Flask, request
from flask import jsonify
from flask import send_file
# from flask import render_template
from flask_cors import CORS, cross_origin
import os

app = Flask(__name__)
CORS(app, support_credentials=True)
cors = CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/keywords', methods=['POST'])
@cross_origin(supports_credentials=True)
def index():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'text/plain'):
        data = str(request.data)
        output = process(data)
        return jsonify(Keywords=output)
    else:
        return 'Content-Type not supported!'

def process(doc):
    vectorizer = KeyphraseCountVectorizer()
    kw_model = KeyBERT(model='all-MiniLM-L6-v2')
    doc = [doc]
    keyphrases = kw_model.extract_keywords(docs=doc, vectorizer=KeyphraseCountVectorizer(),stop_words='english',
    use_mmr=True, diversity=0.2, top_n=10)
    keys = [item for sublist in keyphrases for item in sublist]
    filtered_keys = [x for x in keys if not isinstance(x, float)]
    return filtered_keys

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# gcloud builds submit --tag gcr.io/insight7-353714/keyphrase
# gcloud run deploy --image gcr.io/insight7-353714/keyphrase --platform managed
# https://keyphrase-azi5xvdx6a-ue.a.run.app/keywords