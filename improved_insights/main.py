import json
import os
from flask import Flask, request
from flask import jsonify
from flask_cors import CORS, cross_origin
from langchain import OpenAI
from llama_index import  SimpleDirectoryReader, GPTSimpleVectorIndex, LLMPredictor
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

# OPENAI_API_KEY = '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'

# api_key = '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'

app = Flask(__name__)
cors = CORS(app, support_credentials=True, resources={r"/*": {"origins": "*"}})

@app.route('/insights', methods=['POST'])
def index():
    try:
        content_type = request.headers.get('Content-Type')
        if (content_type == 'text/plain'):
            data = str(request.data)
            # text = data.get("text",None)
            # directory_name = data.get("directory_name",None)
            directory_name = 'test_directory'
            # check if directory exists
            if not os.path.exists(directory_name):
                # create directory if it doesn't exist
                os.mkdir(directory_name)
            with open("test_directory/text.txt", "w") as file:
                file.write(str(data))
    
            documents = SimpleDirectoryReader('./test_directory').load_data()
            
            # documents = StringIterableReader().load_data(
            #      texts=[data])
            llm_predictor = LLMPredictor(llm=OpenAI(temperature=0.7, model_name="text-davinci-003",stop=None,seed=42,max_tokens=512))
            index = GPTSimpleVectorIndex(documents,llm_predictor=llm_predictor)
            # index.save_to_disk('insight7_list') 
            # index = GPTSimpleVectorIndex([])
            # index = index.load_from_disk('insight7_list')

            prompt =  "extract the key insights from the following transcript(s) and group them by topics.:\n\n\
           For each insight under each topic, match the tokens with their exact sources in the original transcript(s)"

            response = index.query(prompt)

            input_str = str(response)

            topics = {}
            current_topic = ""
            for line in input_str.split('\n'):
                line = line.strip()
                if line.startswith("Topic "):
                    current_topic = line.split(':')[1].strip()
                    topics[current_topic] = {}
                    topics[current_topic]['insights'] = []
                    topics[current_topic]['highlights'] = []
                elif line.startswith("Insight "):
                    insight_text = line.split(":")[1].strip()
                    topics[current_topic]['insights'].append(insight_text)
                elif line.startswith("Source:"):
                    source = line.split(":")[1].strip()[1:-1]
                    topics[current_topic]['highlights'].append(source)
                    topics[current_topic]['insights_count'] = len(topics[current_topic]['insights'])   

            return topics
        else:
            return 'Content-Type not supported!'
    except Exception as e:
            return {
                'status': 'Failed',
                'body': json.dumps({"error": repr(e)})
            }


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))

# gcloud builds submit --tag gcr.io/insight7-353714/themed_insights
# gcloud run deploy --image gcr.io/insight7-353714/themed_insights --platform managed

# gcloud beta run deploy demo-app --image gcr.io/<PROJECT_ID>/demo-image --region us-central1 --platform managed --allow-unauthenticated --quiet

# https://gpt3processor-azi5xvdx6a-ew.a.run.app/summarize
# gcloud beta code dev - for testing before deployment
