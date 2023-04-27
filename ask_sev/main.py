import openai
import re
import os
import json
import nltk
import nltk.corpus
nltk.download('stopwords')
nltk.download('punkt')
from nltk import tokenize
from nltk.corpus import stopwords
from gensim.summarization.summarizer import summarize
stop = stopwords.words('english')
from nltk import tokenize
from flask import Flask, request
from flask import jsonify
from flask_cors import CORS, cross_origin

api_key = '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'

app = Flask(__name__)
cors = CORS(app, support_credentials=True, resources={r"/*": {"origins": "*"}})

@app.route('/query', methods=['POST'])
def index():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'application/json'):
        data = json.loads(request.data)
        array = data.get("array",None)
        custom_prompt = data.get("custom_prompt",None)
        text_input = ' '.join(array)
        text_sum = summarize_text(text_input)
        openai.api_key = api_key
        userPrompt=custom_prompt + "without numbering" + ":"
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=userPrompt + "\n\n" + text_sum,
        temperature=0.7,
        max_tokens=500,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )
        # for r in response['choices']:
        #     print(r['text'])
        # result = asksev(text_sum).get("choices")[0]['text']
        output = response.get("choices")[0]['text']
        text_sentences = tokenize.sent_tokenize(output)
        results = []
        for sentence in text_sentences:
            # sentence = clean_text(sentence)
            results.append(sentence)
        # result = [clean_text(sentence) for sentence in text_sentences]
            # response = jsonify(summary=results)
            # response.headers.add('Access-Control-Allow-Origin', '*')
        return jsonify(feedback=results)
    else:
        return 'Content-Type not supported!'

def summarize_text(doc):
    summary = summarize(doc,ratio=0.8,word_count=1200)
    return summary

# def clean_text(doc):
#     doc = doc.lower()
#     cleaned = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", doc)
#     # cleaned = " ".join([word for word in doc.split() if word not in (stop)])
#     return cleaned

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# gcloud builds submit --tag gcr.io/text-tagging-api/insights
# gcloud run deploy --image gcr.io/text-tagging-api/insights --platform managed

# gcloud builds submit --tag gcr.io/insight7-353714/asksev
# gcloud run deploy --image gcr.io/insight7-353714/asksev --platform managed
# gcloud run deploy asksev --image gcr.io/insight7-353714/asksev --region us-east1 --allow-unauthenticated 
# https://asksev-azi5xvdx6a-ue.a.run.app/query