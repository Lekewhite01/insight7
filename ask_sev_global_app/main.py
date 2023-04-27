import openai
import re
import streamlit as st
import os
import json
from gensim.summarization.summarizer import summarize
from nltk import tokenize


# api_key = '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'
st.write("""
    # Global Ask Demo
    """)

api_key = st.text_input(label='Enter your API Key',)

def main():
    data = st.text_area(label='INPUT',placeholder="Ask me Anything")
    userPrompt="I am a highly intelligent question answering bot. If you ask me a question that is rooted in truth, I will give you the answer. If you ask me a question that is nonsense, trickery, or has no clear answer, I will respond with \"Unknown\".\n\nQ: What is human life expectancy in the United States?\nA: Human life expectancy in the United States is 78 years.\n\nQ: Who was president of the United States in 1955?\nA: Dwight D. Eisenhower was president of the United States in 1955.\n\nQ: Which party did he belong to?\nA: He belonged to the Republican Party.\n\nQ: What is the square root of banana?\nA: Unknown\n\nQ: How does a telescope work?\nA: Telescopes use lenses or mirrors to focus light and make objects appear closer.\n\nQ: Where were the 1992 Olympics held?\nA: The 1992 Olympics were held in Barcelona, Spain.\n\nQ: How many squigs are in a bonk?\nA: Unknown\n\nQ:"+f'{data}'+"\nA:"
    if data:
        st.markdown("#### Results")
        with st.spinner('Wait for it...'):
            openai.api_key = api_key
            response = openai.Completion.create(
            model="text-davinci-003",
            prompt=userPrompt,
            temperature=0.7,
            max_tokens=1000,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
            )
            output = response.get("choices")[0]['text']
            st.write(output)
    # return output


if __name__ == '__main__':
    main()


# gcloud builds submit --tag gcr.io/insight7-353714/featuresdemo
# gcloud run deploy --image gcr.io/insight7-353714/featuresdemo --platform managed