import streamlit as st
import openai
import re
import nltk
import nltk.corpus
nltk.download('stopwords')
nltk.download('punkt')
from nltk import tokenize
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
    doc = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", " ", doc)
    cleaned = " ".join([word for word in doc.split() if word not in (stop)])
    return cleaned

def summarize_text(doc):
    summary = summarize(doc,ratio=0.8,word_count=1200)
    return summary

# Extract the key insights from this customer interview and indicate the sentiment for each one:

def ideas(text,userPrompt="Extract the key insights from this customer interview and indicate the sentiment for each one:"):
    openai.api_key = api_key
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=userPrompt + "\n\n" + text,
    temperature=0.6,
    max_tokens=500,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    for r in response['choices']:
        print(r['text'])
    return response

# userPrompt="\n\nTl;dr"
# prompt= text + userPrompt,

def tldr(text,userPrompt="Give a concise summary of this interview:"):
    openai.api_key = api_key
    response = openai.Completion.create(
      model="text-davinci-003",
      prompt= userPrompt + "\n\n" + text,
      temperature=0.6,
      max_tokens=250,
      top_p=1.0,
      frequency_penalty=0.0,
      presence_penalty=0.0
    )
    return response

    # topics covered as single words
    # extract single-word topics covered in this text
    # extract the top 10 single-word topic headers covered in this text

def topics(text,userPrompt="What are the key topics from this customer interview ?:"):
    openai.api_key = api_key
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=userPrompt + "\n\n" + text,
    temperature=0.6,
    max_tokens=50,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    for r in response['choices']:
        print(r['text'])
    return response

custom_prompt = st.text_input(label='Ask your document')
# ask_doc = st.button(
#             label="Query Doc",
#             help=""
#         )


def ask_sev(text,userPrompt=custom_prompt + ":"):
    openai.api_key = api_key
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt=userPrompt + "\n\n" + text,
    temperature=0.6,
    max_tokens=150,
    top_p=1.0,
    frequency_penalty=1,
    presence_penalty=1
    )
    for r in response['choices']:
        print(r['text'])
    return response

def display_app_header(main_txt, sub_txt, is_sidebar=False):
    """
    Code Credit: https://github.com/soft-nougat/dqw-ivves
    function to display major headers at user interface
    :param main_txt: the major text to be displayed
    :param sub_txt: the minor text to be displayed
    :param is_sidebar: check if its side panel or major panel
    :return:
    """
    html_temp = f"""
    <h2 style = "text_align:center; font-weight: bold;"> {main_txt} </h2>
    <p style = "text_align:center;"> {sub_txt} </p>
    </div>
    """
    if is_sidebar:
        st.sidebar.markdown(html_temp, unsafe_allow_html=True)
    else:
        st.markdown(html_temp, unsafe_allow_html=True)

def divider():
    """
    Sub-routine to create a divider for webpage contents
    """
    st.markdown("""---""")

@st.cache
def clean(doc):
    return clean_text(doc)

@st.cache
def tldr_summary(doc):
    return tldr(doc)


@st.cache
def subjects(doc):
    return topics(doc)

@st.cache
def summary(doc):
    return summarize_text(doc)

@st.cache
def sent(doc):
    return sentiment(doc)


@st.cache
def generate_insights(text,userPrompt="Extract the key insights from this customer interview and indicate the sentiment for each one:"):
    return ideas(text,userPrompt="Extract the key insights from this customer interview and indicate the sentiment for each one:")

@st.cache
def ask(doc):
    return ask_sev(doc)


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

    text1 = summary(text1)
    text2 = summary(text2)
    text3 = summary(text3)
    text4 = summary(text4)
    text4 = summary(text5)

    input = []
    input.append(text1)
    input.append(text2)
    input.append(text3)
    input.append(text4)
    input.append(text5)

    # text_sum = summary(text)
    # text_clean = clean(text_sum)
    # st.write(text_clean)
    # if ask_doc:
    if custom_prompt:
        st.markdown("#### Query Results")
        with st.spinner('Wait for it...'):
            # if len(input)>1:
            #     for text in input:
            #         output = subjects(text).get("choices")[0]['text']
            #     # st.write(output)
            #         text_sentences = tokenize.sent_tokenize(output)
            #         for sentence in text_sentences:
            #             st.write('•',sentence)
            # else:
            output = ask(text_sum).get("choices")[0]['text']
            # st.write(output)
            # text_sentences = tokenize.sent_tokenize(output)
            # for sentence in text_sentences:
            st.write(output)

    # text1 = st.text_area(label='INPUT TEXT1',placeholder="Enter Sample Text")
    # text2 = st.text_area(label='INPUT TEXT2',placeholder="Enter Sample Text")
    # text3 = st.text_area(label='INPUT TEXT3',placeholder="Enter Sample Text")
    # text4 = st.text_area(label='INPUT TEXT4',placeholder="Enter Sample Text")
    # text = clean_txt(text)
    # text1 = summary(text1)
    # text2 = summary(text2)
    # text3 = summary(text3)
    # text4 = summary(text4)

    # input = []
    # input.append(text1)
    # input.append(text2)
    # input.append(text3)
    # input.append(text4)
    # st.write(text)
    
    with st.sidebar:
        st.markdown("**Processing**")
        insights = st.button(
            label="Extract Insights",
            help=""
        )
        tldr = st.button(
            label="TL;DR",
            help=""
        )
        subject = st.button(
            label="Topics",
            help=""
        )
        # disposition = st.button(
        #     label="Sentiment",
        #     help=""
        # )

    if insights:
        st.markdown("#### Key Insights")
        with st.spinner('Wait for it...'):
            # if len(input)>1:
            #     for text in input:
            #         output = generate_insights(text).get("choices")[0]['text']
            #     # st.write(output)
            #         text_sentences = tokenize.sent_tokenize(output)
            #         for sentence in text_sentences:
            #             st.write('•',sentence)
            # else:
            output = generate_insights(text_sum).get("choices")[0]['text']
            # st.write(output)
            text_sentences = tokenize.sent_tokenize(output)
            # response = [sentence for sentence in text_sentences]
            for sentence in text_sentences:
                st.write(sentence)
                # overall_sent = sent(sentence).get("choices")[0]['text'] 
                # st.write(overall_sent)
   
    if tldr:
        st.markdown("#### TLDR")
        with st.spinner('Wait for it...'):
            # if len(input)>1:
            #     for text in input:
            #         output = tldr_summary(text).get("choices")[0]['text']
            #     # st.write(output)
            #         text_sentences = tokenize.sent_tokenize(output)
            #         for sentence in text_sentences:
            #             st.write('•',sentence) 
            # else:
            output = tldr_summary(text_sum).get("choices")[0]['text']
            # st.write(output)
            # text_sentences = tokenize.sent_tokenize(output)
            # for sentence in text_sentences:
            st.write(output) 

    if subject:
        st.markdown("#### Topics")
        with st.spinner('Wait for it...'):
            # if len(input)>1:
            #     for text in input:
            #         output = subjects(text).get("choices")[0]['text']
            #     # st.write(output)
            #         text_sentences = tokenize.sent_tokenize(output)
            #         for sentence in text_sentences:
            #             st.write('•',sentence)
            # else:
            output = subjects(text_sum).get("choices")[0]['text']
            st.write(output)
            # text_sentences = tokenize.sent_tokenize(output)
            # for sentence in text_sentences:
            #     st.write('-',sentence)
        
if __name__ == '__main__':
    main()

# gcloud builds submit --tag gcr.io/text-tagging-api/featuresdemo
# gcloud run deploy --image gcr.io/text-tagging-api/featuresdemo --platform managed

# gcloud builds submit --tag gcr.io/insight7-353714/featuresdemo
# gcloud run deploy --image gcr.io/insight7-353714/featuresdemo --platform managed

