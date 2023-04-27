import streamlit as st
import openai
import re
import nltk
import nltk.corpus
nltk.download('stopwords')
import json
nltk.download('punkt')
from nltk import tokenize
# from textwrap import dedent
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from gensim.summarization.summarizer import summarize
stop = stopwords.words('english')

st.write("""
    # Insight7 Features Demo
    """)

api_key = st.text_input(label='Enter your API Key',)

def clean_text(doc):
    doc = doc.lower()
    cleaned = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", doc)
    # cleaned = " ".join([word for word in doc.split() if word not in (stop)])
    return cleaned

def summarize_text(doc):
    summary = summarize(doc,ratio=0.9,word_count=700)
    return summary

def ideas(text,userPrompt="extract and display the key insights in this transcript without numbering:"):
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

def main():
    # st.write("""
    # # GPT-3 Text Processing Demo
    # """)
    input_help_text = """
    Enter Text
    """
    final_message = """
    The data was successfully analyzed
    """
    text1 = st.text_area(label='INPUT TEXT1',placeholder="Enter Sample Text")
    text2 = st.text_area(label='INPUT TEXT2',placeholder="Enter Sample Text")
    text3 = st.text_area(label='INPUT TEXT3',placeholder="Enter Sample Text")
    text4 = st.text_area(label='INPUT TEXT4',placeholder="Enter Sample Text")
    text5 = st.text_area(label='INPUT TEXT',placeholder="Enter Sample Text")

    input = []
    input.append(text1)
    input.append(text2)
    input.append(text3)
    input.append(text4)
    input.append(text5)
    # while("" in input):
    #     input.remove("")
    input = list(filter(None, input))
    # full_text= ''.join(input)
 
    with st.sidebar:
        st.markdown("**Processing**")
        insight = st.button(
            label="Process Insights",
            help=""
        )
    if insight:
        st.markdown("#### Results")
        
        with st.spinner('Wait for it...'):
            insights = {}
            for i in range(len(input)):
                insights[f"Doc {i+1} insights"] = []
                
                if len(input[i].split()) > 1200:
                    binned = get_chunks(input[i])
                    # st.write(binned)
                # cleaned = clean_text(summarized)
                    for chunk in binned:
                        # st.write(chunk)
                        output = ideas(chunk).get("choices")[0]['text']
                        # st.write(output)
                        text_sentences = tokenize.sent_tokenize(output) 
                        for sentence in text_sentences:
                            insights[f"Doc {i+1} insights"].append({
                            "insight": sentence})
            #                 "source": openai.Completion.create(
            #                         model="text-davinci-003",
            #                         prompt=f'Please display the related segment of {chunk} that {sentence} was extracted from',
            #                         temperature=0.1,
            #                         max_tokens=800,
            #                         top_p=1.0,
            #                         frequency_penalty=0.0,
            #                         presence_penalty=0.0
            #                         ).get("choices")[0]['text']
            #                         })
                else:
            #         # cleaned = clean_text(input[i])
                    output = ideas(input[i]).get("choices")[0]['text']
                    text_sentences = tokenize.sent_tokenize(output) 
                    for sentence in text_sentences:
                        insights[f"Doc {i+1} insights"].append({
                            "insight": sentence})
            #                 "source": openai.Completion.create(
            #                         model="text-davinci-003",
            #                         prompt=f'Please display the related segment of {input[i]} that {sentence} was extracted from',
            #                         temperature=0.1,
            #                         max_tokens=800,
            #                         top_p=1.0,
            #                         frequency_penalty=0.0,
            #                         presence_penalty=0.0
            #                         ).get("choices")[0]['text']
            #             })
            # st.write(insights)

if __name__ == '__main__':
    main()

# gcloud builds submit --tag gcr.io/text-tagging-api/featuresdemo
# gcloud run deploy --image gcr.io/text-tagging-api/featuresdemo --platform managed

# gcloud builds submit --tag gcr.io/insight7-353714/featuresdemo
# gcloud run deploy --image gcr.io/insight7-353714/featuresdemo --platform managed

# return {
#         'statusCode': 200,
#         'body': json.dumps('Hello from Lambda!')
#     }
