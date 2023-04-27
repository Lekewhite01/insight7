import os
os.environ['OPENAI_API_KEY'] = "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"

import openai
import re
import pprint
# import nltk.corpus

"""Wrapper around OpenAI embedding models."""
from typing import Any, Dict, List, Optional
import os
from langchain.embeddings.base import Embeddings
from langchain.utils import get_from_dict_or_env
from openai.error import APIConnectionError, APIError, RateLimitError, Timeout
from pydantic import BaseModel, Extra, root_validator
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

import re
from io import BytesIO
from typing import Any, Dict, List

import docx2txt
import streamlit as st
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.llms import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import VectorStore
from langchain.vectorstores.faiss import FAISS
from openai.error import AuthenticationError


class OpenAIEmbeddings(BaseModel, Embeddings):
    """Wrapper around OpenAI embedding models.

    To use, you should have the ``openai`` python package installed, and the
    environment variable ``OPENAI_API_KEY`` set with your API key or pass it
    as a named parameter to the constructor.

    Example:
        .. code-block:: python

            from langchain.embeddings import OpenAIEmbeddings
            openai = OpenAIEmbeddings(openai_api_key="my-api-key")
    """

    client: Any  #: :meta private:
    document_model_name: str = "text-embedding-ada-002"
    query_model_name: str = "text-embedding-ada-002"
    openai_api_key: Optional[str] = None

    class Config:
        """Configuration for this pydantic object."""

        extra = Extra.forbid

    # TODO: deprecate this
    @root_validator(pre=True, allow_reuse=True)
    def get_model_names(cls, values: Dict) -> Dict:
        """Get model names from just old model name."""
        if "model_name" in values:
            if "document_model_name" in values:
                raise ValueError(
                    "Both `model_name` and `document_model_name` were provided, "
                    "but only one should be."
                )
            if "query_model_name" in values:
                raise ValueError(
                    "Both `model_name` and `query_model_name` were provided, "
                    "but only one should be."
                )
            model_name = values.pop("model_name")
            values["document_model_name"] = f"text-search-{model_name}-doc-001"
            values["query_model_name"] = f"text-search-{model_name}-query-001"
        return values

    @root_validator(allow_reuse=True)
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        openai_api_key = get_from_dict_or_env(
            values, "openai_api_key", "OPENAI_API_KEY"
        )
        try:
            import openai

            openai.api_key = openai_api_key
            values["client"] = openai.Embedding
        except ImportError:
            raise ValueError(
                "Could not import openai python package. "
                "Please it install it with `pip install openai`."
            )
        return values

    @retry(
        reraise=True,
        stop=stop_after_attempt(100),
        wait=wait_exponential(multiplier=1, min=10, max=60),
        retry=(
            retry_if_exception_type(Timeout)
            | retry_if_exception_type(APIError)
            | retry_if_exception_type(APIConnectionError)
            | retry_if_exception_type(RateLimitError)
        ),
    )
    def _embedding_func(self, text: str, *, engine: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint with exponential backoff."""
        # replace newlines, which can negatively affect performance.
        text = text.replace("\n", " ")
        return self.client.create(input=[text], engine=engine)["data"][0]["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Call out to OpenAI's embedding endpoint for embedding search docs.

        Args:
            texts: The list of texts to embed.

        Returns:
            List of embeddings, one for each text.
        """
        responses = [
            self._embedding_func(text, engine=self.document_model_name)
            for text in texts
        ]
        return responses

    def embed_query(self, text: str) -> List[float]:
        """Call out to OpenAI's embedding endpoint for embedding query text.

        Args:
            text: The text to embed.

        Returns:
            Embeddings for the text.
        """
        embedding = self._embedding_func(text, engine=self.query_model_name)
        return embedding
    


def parse_docx(file: BytesIO) -> str:
    text = docx2txt.process(file)
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def parse_txt(file: BytesIO) -> str:
    text = file.read().decode("utf-8")
    # Remove multiple newlines
    text = re.sub(r"\n\s*\n", "\n\n", text)
    return text


def text_to_docs(text) -> List[Document]:
    """Converts a string or list of strings to a list of Documents
    with metadata."""
    if isinstance(text, str):
        # Take a single string as one page
        text = [text]
    page_docs = [Document(page_content=page) for page in text]

    # Add page numbers as metadata
    for i, doc in enumerate(page_docs):
        doc.metadata["page"] = i + 1

    # Split pages into chunks
    doc_chunks = []

    for doc in page_docs:
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=800,
            separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""],
            chunk_overlap=0,
        )
        chunks = text_splitter.split_text(doc.page_content)
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk, metadata={"page": doc.metadata["page"], "chunk": i}
            )
            # Add sources a metadata
            doc.metadata["source"] = f"{doc.metadata['page']}-{doc.metadata['chunk']}"
            doc_chunks.append(doc)
    return doc_chunks



def embed_docs(docs: List[Document]) -> VectorStore:
    """Embeds a list of Documents and returns a FAISS index"""

    embeddings = OpenAIEmbeddings(
        # openai_api_key=st.session_state.get("OPENAI_API_KEY")
    )  # type: ignore
    index = FAISS.from_documents(docs, embeddings)

    return index



def search_docs(index: VectorStore, query: str) -> List[Document]:
    """Searches a FAISS index for similar chunks to the query
    and returns a list of Documents."""

    # Search for similar chunks
    docs = index.similarity_search(query, k=5)
    return docs



def get_answer(docs: List[Document], query: str) -> Dict[str, Any]:
    """Gets an answer to a question from a list of Documents."""

    # Get the answer

    chain = load_qa_with_sources_chain(
        OpenAI(
            temperature=0.7,model_name="text-davinci-003",max_tokens=1024),  # type: ignore,# openai_api_key=st.session_state.get("OPENAI_API_KEY")
        chain_type="stuff",
        prompt=STUFF_PROMPT,
    )

    answer = chain(
        {"input_documents": docs, "question": query}, return_only_outputs=True
    )
    return answer



def get_sources(answer: Dict[str, Any], docs: List[Document]) -> List[Document]:
    """Gets the source documents for an answer."""

    # Get sources for the answer
    source_keys = [s for s in answer["output_text"].split("SOURCES: ")[-1].split(", ")]

    source_docs = []
    for doc in docs:
        if doc.metadata["source"] in source_keys:
            source_docs.append(doc)

    return source_docs


def wrap_text_in_html(text) -> str:
    """Wraps each text block separated by newlines in <p> tags"""
    if isinstance(text, list):
        # Add horizontal rules between pages
        text = "\n<hr/>\n".join(text)
    return "".join([f"<p>{line}</p>" for line in text.split("\n")])



from langchain.prompts import PromptTemplate

## Use a shorter template to reduce the number of tokens in the prompt
template = """Create a final answer to the given questions using the provided document excerpts(in no particular order) as references. ALWAYS include a "SOURCES" section in your answer including only the minimal set of sources needed to answer the question. If you are unable to answer the question, simply state that you do not know. Do not attempt to fabricate an answer and leave the SOURCES section empty.
---------
QUESTION: What  is the purpose of ARPA-H?
=========
Content: More support for patients and families. \n\nTo get there, I call on Congress to fund ARPA-H, the Advanced Research Projects Agency for Health. \n\nIt’s based on DARPA—the Defense Department project that led to the Internet, GPS, and so much more.  \n\nARPA-H will have a singular purpose—to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.\
While we’re at it, let’s make sure every American can get the health care they need. \n\nWe’ve already made historic investments in health care. \n\nWe’ve made it easier for Americans to get the care they need, when they need it. \n\nWe’ve made it easier for Americans to get the treatments they need, when they need them. \n\nWe’ve made it easier for Americans to get the medications they need, when they need them.\
The V.A. is pioneering new ways of linking toxic exposures to disease, already helping  veterans get the care they deserve. \n\nWe need to extend that same care to all Americans. \n\nThat’s why I’m calling on Congress to pass legislation that would establish a national registry of toxic exposures, and provide health care and financial assistance to those affected.
=========
FINAL ANSWER: The purpose of ARPA-H is to drive breakthroughs in cancer, Alzheimer’s, diabetes, and more.(SOURCES: 1-32)
---------
QUESTION: {question}
=========
{summaries}
=========
FINAL ANSWER:"""

STUFF_PROMPT = PromptTemplate(
    template=template, input_variables=["summaries", "question"])


query = 'Given the following transcript, list all the pain points, desire and behaviour \
    discussed by the team and group them by topic. ALWAYS include a "SOURCES" section in your answer. \
    Instruction: \
    Follow these steps, Identify pain points, desire and behaviour with the corresponding sources.\
    For each pain points, desires and behviour group \
    them into topics based on related insights'
    


def extract_insights(text,userPrompt="Group Insights from pain points, desires and behaviour into topics based on related themes,\
return the related pain points, desires and behaviour for each topic and the corresponding sources"):
    openai.api_key = "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"
    response = openai.Completion.create(
    model="text-davinci-003",
    prompt= userPrompt + '\n:' + text,
    temperature=0.8,
    max_tokens=1024,
    top_p=1.0,
    frequency_penalty=0.0,
    presence_penalty=0.0
    )
    insight = response.choices[0].text.strip()
    return insight


def get_info(group, sources):
    source_ref = {
        source.metadata['source']: source.page_content 
        for source in sources
    }
    
    value_point, insights = group.split(': ')
    insights = insights.split('; ')
    
    points = []
    
    for insight in insights:
        _insight, ref = insight.split('(')
        ref = ref.replace(')', '').replace('.', '')
        ref = source_ref[ref]
        
        points.append({_insight: ref})
    
    return {value_point: points}

def prepare_json_format(response):
    
    result = {'data': {}}
    
    topics = response.split('\n\nTopic')
    topics[0] = topics[0].replace('Topic', ' ')
    
    for topic in topics:
        groups = topic.split('\n\n')
        
        topic_count = 0
        for i, group in enumerate(groups):
            if i == 0:
                topic, group = group.split('\n')
                topic = re.sub(r'^[\W_0-9]+', '', topic)
                
                result['data'][topic] = {'agg': [], 'count': 0}
            
            if not group.startswith('SOURCE'):
                data = get_info(group)
                result['data'][topic]['agg'].append(data)
                topic_count += sum(len(v) for v in data.values())
            
        result['data'][topic]['count'] = topic_count
    
    return result



def main():
    doc = open('data/untitled.txt','r')
    text = text_to_docs(doc)
    index = embed_docs(text)
    sources = search_docs(index, query)
    answer = get_answer(sources, query)
    answer_text = answer['output_text']
    response = extract_insights(answer_text)
    
    return prepare_json_format(response)

    



