import openai
import re
import os
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
# cors = CORS(app, support_credentials=True, resources={r"/*": {"origins": "*"}})

@app.route('/summarize', methods=['POST'])
def index():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'text/plain'):
        data = str(request.data)
        data = summarize_text(data)
        data = clean_text(data)
        output = ideas(data).get("choices")[0]['text']
        text_sentences = tokenize.sent_tokenize(output)
        results = [sentence for sentence in text_sentences]
        # for sentence in text_sentences:
        #     results.append(sentence)
        # response = jsonify(insights=results)
        # response.headers.add('Access-Control-Allow-Origin', '*')
        return jsonify(insights=results)
    else:
        return 'Content-Type not supported!'

def summarize_text(doc):
    summary = summarize(doc,ratio=0.8,word_count=1200)
    return summary

def clean_text(doc):
    doc = doc.lower()
    doc = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", doc)
    cleaned = " ".join([word for word in doc.split() if word not in (stop)])
    return cleaned

def ideas(text,userPrompt="List the key insights from this customer interview and indicate the sentiment for each one:"):
    openai.api_key = api_key
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=userPrompt + "\n\n" + text,
    temperature=0.6,
    max_tokens=1200,
    top_p=1.0,
    frequency_penalty=1,
    presence_penalty=1
    )
    for r in response['choices']:
        print(r['text'])
    return response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# gcloud builds submit --tag gcr.io/text-tagging-api/gpt3demo
# gcloud run deploy --image gcr.io/text-tagging-api/gpt3demo --platform managed

# gcloud builds submit --tag gcr.io/insight7-353714/singleinsights
# gcloud run deploy --image gcr.io/insight7-353714/singleinsights --platform managed

# https://gpt3processor-azi5xvdx6a-ew.a.run.app/summarize

# curl -H \
# "Authorization: Bearer $(gcloud auth print-identity-token)" \
# https://singlenote-azi5xvdx6a-ue.a.run.app