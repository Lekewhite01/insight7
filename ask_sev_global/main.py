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
# cors = CORS(app, support_credentials=True, resources={r"/*": {"origins": "*"}})

@app.route('/query', methods=['POST'])
def index():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'text/plain'):
        data = str(request.data)
        userPrompt="I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ:"+f'{data}'+"\nA:"
        openai.api_key = api_key
        response = openai.Completion.create(
        model="text-davinci-003",
        prompt=userPrompt,
        temperature=0.7,
        max_tokens=1024,
        top_p=1.0,
        frequency_penalty=0.0,
        presence_penalty=0.0
        )
        output = response.get("choices")[0]['text']
        return jsonify(feedback=output)
    else:
        return 'Content-Type not supported!'


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# gcloud builds submit --tag gcr.io/insight7-353714/asksev_global
# gcloud run deploy asksevglobal --image gcr.io/insight7-353714/asksev_global --platform managed
# gcloud run deploy asksevglobal --image gcr.io/insight7-353714/asksev_global --region us-east1 --allow-unauthenticated 
# https://asksev-azi5xvdx6a-ue.a.run.app/query