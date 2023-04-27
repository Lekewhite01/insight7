import openai
import os
import re
import nltk
from nltk import tokenize
from flask import Flask, request
from flask import jsonify
from flask import render_template
from pysummarization.nlpbase.auto_abstractor import AutoAbstractor
from pysummarization.tokenizabledoc.simple_tokenizer import SimpleTokenizer
from pysummarization.abstractabledoc.top_n_rank_abstractor import TopNRankAbstractor
from flask import Flask, request
from flask import jsonify
from flask_cors import CORS, cross_origin
nltk.download('punkt')
api_key = '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'

app = Flask(__name__)
cors = CORS(app, support_credentials=True, resources={r"/*": {"origins": "*"}})

@app.route('/extract', methods=['POST'])
def index():
    content_type = request.headers.get('Content-Type')
    if (content_type == 'text/plain'):
        data = str(request.data)
        # data = clean_text(data)
        # Object of automatic summarization.
        auto_abstractor = AutoAbstractor()
        # Set tokenizer.
        auto_abstractor.tokenizable_doc = SimpleTokenizer()
        # Set delimiter for making a list of sentence.
        auto_abstractor.delimiter_list = [".", "\n"]
        # Object of abstracting and filtering document.
        abstractable_doc = TopNRankAbstractor()
        # Summarize document.
        result_dict = auto_abstractor.summarize(data, abstractable_doc)
        summary = result_dict["summarize_result"]
        summary = ''.join(summary)
        insights = []
        insights.append(summary)
        if len(insights)>1:
            results = []
            for i in range(len(insights)):
                output = tldr(insights[i]).get("choices")[0]['text']
                results.append(output)
        else:
            results = []
            output = tldr(insights[0]).get("choices")[0]['text']
            results.append(output)
            results = results[0]
            # response = jsonify(summary=results)
            # response.headers.add('Access-Control-Allow-Origin', '*')
        return jsonify(summary=results)
    else:
        return 'Content-Type not supported!'


def tldr(text,userPrompt="\n\nTl;dr"):
    openai.api_key = api_key
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt= text + userPrompt,
      temperature=0.7,
      max_tokens=2000,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    return response

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# gcloud builds submit --tag gcr.io/text-tagging-api/insights
# gcloud run deploy --image gcr.io/text-tagging-api/insights --platform managed

# gcloud builds submit --tag gcr.io/insight7-353714/tldr_summary
# gcloud run deploy --image gcr.io/insight7-353714/tldr_summary --platform managed

# https://tldrsummary-azi5xvdx6a-ue.a.run.app/extract