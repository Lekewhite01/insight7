import openai
import re
import os
import json
import nltk
import nltk.corpus
# nltk.download('stopwords')
nltk.download('punkt')
from nltk import tokenize
from nltk.corpus import stopwords
# from textwrap import dedent
from gensim.summarization.summarizer import summarize
# stop = stopwords.words('english')
from nltk import tokenize
from flask import Flask, request
from flask import jsonify
from flask_cors import CORS, cross_origin

api_key = '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'

app = Flask(__name__)
cors = CORS(app, support_credentials=True, resources={r"/*": {"origins": "*"}})

@app.route('/aspect', methods=['POST'])
def index():
    try:
        content_type = request.headers.get('Content-Type')
        if (content_type == 'text/plain'):
            data = str(request.data)
            results = []
            # results['insights'] = []
    
            if len(data.split()) > 1200:
                binned = get_chunks(data)
                # cleaned = clean_text(summarized)
                for chunk in binned:
                    output = ideas(chunk).get("choices")[0]['text']
                    text_sentences = tokenize.sent_tokenize(output) 
                    for sentence in text_sentences:
                        results.append({
                            "insight": sentence.strip(),
                            "source": openai.Completion.create(
                                        model="text-davinci-003",
                                        prompt=f'for {sentence}, display the relevant portion of {chunk} that {sentence} was extracted from',
                                        temperature=0.1,
                                        max_tokens=600,
                                        top_p=1.0,
                                        frequency_penalty=0.0,
                                        presence_penalty=0.0
                                        ).get("choices")[0]['text']
                                        })
            else:
                # cleaned = clean_text(data)
                output = ideas(data).get("choices")[0]['text']
                text_sentences = tokenize.sent_tokenize(output) 
                for sentence in text_sentences:
                    results.append({
                        "insight": sentence.strip(),
                        "source": openai.Completion.create(
                                    model="text-davinci-003",
                                    prompt=f'for {sentence}, display the relevant portion of {data} that {sentence} was extracted from',
                                    temperature=0.1,
                                    max_tokens=600,
                                    top_p=1.0,
                                    frequency_penalty=0.0,
                                    presence_penalty=0.0
                                    ).get("choices")[0]['text']
                                    })
                
            return jsonify(Result=results)
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

def ideas(text,userPrompt="Extract the key insights from this customer interview:"):
    try:
        openai.api_key = api_key
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=userPrompt + '\n\n' + text,
        temperature=0.1,
        max_tokens=400,
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

def get_chunks(string, maxlength=1000):
    try:
        start = 0
        end = 0
        while start + maxlength  < len(string) and end != -1:
            end = string.rfind(" ", start, start + maxlength + 1)
            yield string[start:end]
            start = end +1
        yield string[start:]
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
