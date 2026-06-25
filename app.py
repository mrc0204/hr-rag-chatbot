import os
import streamlit as st

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# =========================
# CONFIGURATION
# =========================

LLM_PROVIDER = "gemini"
LLM_MODEL = "gemini-2.5-flash"

CORPUS_PATH = "./zyro-dynamics-hr-corpus"

# =========================
# PAGE CONFIG
# =========================

st.set_page_config(
    page_title="HR RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 HR Policy RAG Chatbot")
st.markdown("Ask questions related to company HR policies.")

# =========================
# LOAD VECTORSTORE
# =========================

@st.cache_resource
def initialize_rag():

    # Load documents
    loader = PyPDFDirectoryLoader(CORPUS_PATH)
    documents = loader.load()

    # Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=150
    )

    chunks = splitter.split_documents(documents)

    # Embeddings
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-mpnet-base-v2"
    )

    # Vector DB
    vectorstore = FAISS.from_documents(
        documents=chunks,
        embedding=embeddings
    )

    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )

    # LLM Selection
    if LLM_PROVIDER == "groq":
        from langchain_groq import ChatGroq

        llm = ChatGroq(
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=512
        )

    elif LLM_PROVIDER == "gemini":
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=LLM_MODEL,
            temperature=0.1,
            max_output_tokens=512
        )

    elif LLM_PROVIDER == "openai":
        from langchain_openai import ChatOpenAI

        llm = ChatOpenAI(
            model=LLM_MODEL,
            temperature=0.1,
            max_tokens=512
        )

    else:
        raise ValueError("Unsupported provider")

    return retriever, llm


retriever, llm = initialize_rag()

# =========================
# PROMPTS
# =========================

RAG_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        '''
Use ONLY the provided context.

Give a detailed answer.

If the answer is not available in the context, reply:

"I can only answer the queries related to the company HR policies."

Mention sources at the end.
'''
    ),
    (
        "human",
        "Context:\n{context}\n\nQuestion: {question}"
    )
])

OOS_PROMPT = ChatPromptTemplate.from_template(
    '''
You are a classifier.

Determine whether the user's question can be answered using the document knowledge base.

Respond with only one word:

YES - if the question is related to the document/domain.
NO - if it is unrelated.

Question:
{question}
'''
)

REFUSAL_MESSAGE = (
    "Sorry, I can only answer questions that are related to the uploaded documents."
)

# =========================
# HELPERS
# =========================

def format_docs(docs):
    return "\n\n".join(
        doc.page_content
        for doc in docs
    )

def rag_chain(question):

    docs = retriever.invoke(question)

    context = format_docs(docs)

    response = (
        RAG_PROMPT
        | llm
        | StrOutputParser()
    ).invoke({
        "context": context,
        "question": question
    })

    return response

def ask_bot(question):

    classifier = (
        OOS_PROMPT
        | llm
        | StrOutputParser()
    )

    decision = classifier.invoke(
        {"question": question}
    ).strip().upper()

    if decision == "NO":
        return REFUSAL_MESSAGE

    return rag_chain(question)

# =========================
# CHAT UI
# =========================

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

prompt = st.chat_input(
    "Ask an HR policy question..."
)

if prompt:

    st.session_state.messages.append(
        {
            "role": "user",
            "content": prompt
        }
    )

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):

        with st.spinner("Searching documents..."):
            response = ask_bot(prompt)

        st.markdown(response)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": response
        }
    )