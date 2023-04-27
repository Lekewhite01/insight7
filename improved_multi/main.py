import openai
import json
import re
import os
import nltk
import nltk.corpus
# nltk.download('stopwords')
nltk.download('punkt')
from nltk import tokenize
from textwrap import dedent
from nltk.corpus import stopwords
from gensim.summarization.summarizer import summarize
# stop = stopwords.words('english')
from nltk import tokenize
from flask import Flask, request
from flask import jsonify
from flask_cors import CORS, cross_origin
api_key = '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'

app = Flask(__name__)
# cors = CORS(app, support_credentials=True, resources={r"/*": {"origins": "*"}})

@app.route('/extract', methods=['POST'])
def index():
    try:
        content_type = request.headers.get('Content-Type')
        if (content_type == 'application/json'):
            data = json.loads(request.data)
            array = data.get("array",None)
            # results = []
            insights = {}
            for i in range(len(array)):
                insights[f"Doc {i+1} insights"] = []
                if len(array[i].split()) > 1200:
                    summary = summarize_text(array[i])
                    cleaned = clean_text(summary)
                    output = ideas(cleaned).get("choices")[0]['text']
                    text_sentences = tokenize.sent_tokenize(output)
                    for sentence in text_sentences:
                        insights[f"Doc {i+1} insights"].append({
                            "insight": sentence.strip(),
                            "source": openai.Completion.create(
                                    model="text-davinci-003",
                                    prompt=f'for {sentence}, display the relevant portion of {array[i]} that {sentence} was extracted from',
                                    temperature=0.1,
                                    max_tokens=600,
                                    top_p=1.0,
                                    frequency_penalty=0.0,
                                    presence_penalty=0.0
                                    ).get("choices")[0]['text']
                        })
                    #         "insight": sentence.strip()
                    #     }
                    # Look through {summarize_text(array[i])} and 
                else:
                    cleaned = clean_text(array[i])
                    output = ideas(cleaned).get("choices")[0]['text']
                    text_sentences = tokenize.sent_tokenize(output)
                    for sentence in text_sentences:
                        insights[f"Doc {i+1} insights"].append({
                            "insight": sentence.strip(),
                            "source": openai.Completion.create(
                                    model="text-davinci-003",
                                    prompt=f'for {sentence}, display the relevant portion of {array[i]} that {sentence} was extracted from',
                                    temperature=0.1,
                                    max_tokens=600,
                                    top_p=1.0,
                                    frequency_penalty=0.0,
                                    presence_penalty=0.0
                                    ).get("choices")[0]['text']
                        })       
                         # full_text = ' '.join(array)
                        #  Look through {array[i]} and 
            return {"Insights": insights}

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

def ideas(text,userPrompt="extract insights from the following text:"):
    try:
        openai.api_key = api_key
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=userPrompt + '\n\n' + text,
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

def related(text,userPrompt="list insights related with one another from this text:"):
    try:
        openai.api_key = api_key
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=userPrompt + '\n\n' + text,
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

def aspect(text,userPrompt="extract aspects from the following text:"):
    try:
        openai.api_key = api_key
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=userPrompt + '\n\n' + text,
        temperature=0.1,
        max_tokens=80,
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

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# gcloud builds submit --tag gcr.io/text-tagging-api/insights
# gcloud run deploy --image gcr.io/text-tagging-api/insights --platform managed

# gcloud builds submit --tag gcr.io/insight7-353714/multinote
# gcloud run deploy --image gcr.io/insight7-353714/multinote --platform managed
# https://multinote-azi5xvdx6a-ue.a.run.app/extract
# gcloud beta code dev - for testing before deployment
