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

@app.route('/summarize', methods=['POST'])
def index():
    try:
        content_type = request.headers.get('Content-Type')
        if (content_type == 'text/plain'):
            data = str(request.data)
            # data = summarize_text(data)
            # data = clean_text(data)
            results = []
            if len(data.split()) > 1200:
                    summarized = summarize_text(data)
                    cleaned = clean_text(summarized)
                    output = ideas(cleaned).get("choices")[0]['text']
                    text_sentences = tokenize.sent_tokenize(output)
                    for sentence in text_sentences:
                        sentence = sentence.replace('\n\n', '')
                        sent = sentiment(sentence)
                        results.append({"insight":sentence.strip(),"sentiment":sent.strip()})
            else:
                cleaned = clean_text(data)
                output = ideas(cleaned).get("choices")[0]['text']
                text_sentences = tokenize.sent_tokenize(output)
                for sentence in text_sentences:
                    sent = sentiment(sentence)
                    results.append({"insight":sentence.strip(),"sentiment":sent.strip()})
            # output = ideas(data).get("choices")[0]['text']
            # output = ideas(data)
            # results = []
            # text_sentences = tokenize.sent_tokenize(output)
            # for sentence in text_sentences:
            #     sent = sentiment(sentence)
            #     results.append({"insight":sentence.strip(),"sentiment":sent.strip()})
            return jsonify(insights=results)
        else:
            return 'Content-Type not supported!'
    except Exception as e:
            return {
                'status': 'Failed',
                'body': json.dumps({"error": repr(e)})
            }
def summarize_text(doc):
    try:
        summary = summarize(doc,ratio=0.8,word_count=1200)
        return summary
    except Exception as e:
            return {
                'status': 'Failed',
                'body': json.dumps({"error": repr(e)})
            }

def clean_text(doc):
    try:
        doc = doc.lower()
        cleaned = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", doc)
        # cleaned = " ".join([word for word in doc.split() if word not in (stop)])
        return cleaned
    except Exception as e:
            return {
                'status': 'Failed',
                'body': json.dumps({"error": repr(e)})
            }    

# Extract the key insights from this customer interview and indicate the sentiment for each one:

def ideas(text,userPrompt="Present important insights without numbering and heading:"):
    try:
        openai.api_key = api_key
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=userPrompt + text,
        temperature=0.1,
        max_tokens=600,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )
        for r in response['choices']:
            print(r['text'])
        return response
    except Exception as e:
            return {
                'status': 'Failed',
                'body': json.dumps({"error": repr(e)})
            }  

def sentiment(text,userPrompt="classify the sentiment of this sentence as positive, negative, or neutral:"):
    try:
        openai.api_key = api_key
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=userPrompt + "without numbering and heading" + text,
        temperature=0.3,
        max_tokens=60,
        top_p=1.0,
        frequency_penalty=0.5,
        presence_penalty=0.0
        )
        # sentiments = []
        for r in response['choices']:
            print(r['text'])
        return response.get("choices")[0]['text']
    except Exception as e:
            return {
                'status': 'Failed',
                'body': json.dumps({"error": repr(e)})
            }

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# gcloud builds submit --tag gcr.io/text-tagging-api/gpt3demo
# gcloud run deploy --image gcr.io/text-tagging-api/gpt3demo --platform managed

# gcloud builds submit --tag gcr.io/insight7-353714/gpt3processor
# gcloud run deploy --image gcr.io/insight7-353714/gpt3processor --platform managed

# gcloud beta run deploy demo-app --image gcr.io/<PROJECT_ID>/demo-image --region us-central1 --platform managed --allow-unauthenticated --quiet

# https://gpt3processor-azi5xvdx6a-ew.a.run.app/summarize
# gcloud beta code dev - for testing before deployment
