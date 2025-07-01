import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFacePipeline
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.embeddings.base import Embeddings

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from sentence_transformers import SentenceTransformer

from htmltemplate import css, bot_template, user_template

# Load environment variables
load_dotenv()


# Custom embedding class
class CustomInstructEmbeddings(Embeddings):
    def __init__(self, model_name: str):
        self.model = SentenceTransformer(model_name)

    def embed_documents(self, texts):
        embeddings = self.model.encode(texts, show_progress_bar=True)
        return [embedding.tolist() for embedding in embeddings]

    def embed_query(self, text):
        embedding = self.model.encode([text])[0]
        return embedding.tolist()


# Extract text from uploaded PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


# Split text into chunks
def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)


# Create a FAISS vectorstore from chunks
def get_vectorstore(text_chunks):
    embeddings = CustomInstructEmbeddings(model_name="all-MiniLM-L6-v2")
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)


# Load LLM and connect to vectorstore for retrieval chain
def get_conversation_chain(vectorstore):
    model_name = "google/flan-t5-base"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation", model=model, tokenizer=tokenizer, max_length=512
    )
    llm = HuggingFacePipeline(pipeline=pipe)

    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    return ConversationalRetrievalChain.from_llm(
        llm=llm, retriever=vectorstore.as_retriever(), memory=memory
    )


# Handle a single user question
def handle_user_input(user_question):
    if st.session_state.conversation is None:
        st.warning("Please upload and process your PDFs first.")
        return

    response = st.session_state.conversation.run(user_question)

    st.session_state.chat_history.append({"role": "user", "content": user_question})
    st.session_state.chat_history.append({"role": "bot", "content": response})

    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.write(
                user_template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", msg["content"]), unsafe_allow_html=True
            )


def main():
    st.set_page_config(page_title="Chat with Multiple PDFs", page_icon="📚")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    st.header("📚 Chat with Multiple PDFs")

    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_user_input(user_question)

    with st.sidebar:
        st.subheader("Upload Your PDFs")
        pdf_docs = st.file_uploader("Upload multiple PDFs", accept_multiple_files=True)
        if st.button("Process") and pdf_docs:
            with st.spinner("Processing..."):
                try:
                    raw_text = get_pdf_text(pdf_docs)
                    text_chunks = get_text_chunks(raw_text)
                    vectorstore = get_vectorstore(text_chunks)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Documents processed! You can now ask questions.")
                except Exception as e:
                    st.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
