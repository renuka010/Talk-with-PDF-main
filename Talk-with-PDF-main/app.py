import streamlit as st
from PyPDF2 import PdfReader

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import uuid
import chromadb
from langchain_community.chat_models import ChatOllama
from langchain.text_splitter import CharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory

from htmlTemplates import css, bot_template, user_template, hide_st_style, footer
from matplotlib import style

def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# id generator for metadata
def generate_id():
    return str(uuid.uuid4().int)

def get_vectorstore(pdf_docs):
    new_collection = False

    embeddings = SentenceTransformerEmbeddings(model_name="all-mpnet-base-v2")
    persistent_client = chromadb.PersistentClient()
    try:
        collection = persistent_client.get_collection("nsdc")
        print("collection is retrived")
    except:
        collection = persistent_client.create_collection("nsdc")
        print("collection is created")
        new_collection = True

    vectorstore =Chroma(
        collection_name="nsdc",
        embedding_function=embeddings,
        client=persistent_client,  # Where to save data locally
    )

    if new_collection:
        raw_text = get_pdf_text(pdf_docs)
        text_chunks = get_text_chunks(raw_text)
        ids = [generate_id() for _ in range(len(text_chunks))]
        vectorstore.add_texts(texts = text_chunks,
                              ids=ids)
    return vectorstore

### Statefully manage chat history ###
if "store" not in st.session_state:
    st.session_state.store = {}


def get_session_history(session_id: str) -> BaseChatMessageHistory:
    store = st.session_state.store
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def get_conversation_chain(vector_store):
    # Initializing LLM Model
    llm = ChatOllama(base_url='http://host.docker.internal:11434', model="phi3", format="text", temperature=0.7, keep_alive=-1)

    retriever = vector_store.as_retriever(k=5)

    ### Contextualize question ###
    contextualize_q_system_prompt = """Given a chat history and the latest user question \
    which might reference context in the chat history, formulate a standalone question \
    which can be understood without the chat history. Do NOT answer the question, \
    just reformulate it if needed and otherwise return it as is."""
    contextualize_q_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", contextualize_q_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    history_aware_retriever = create_history_aware_retriever(
        llm, retriever, contextualize_q_prompt
    )

    ### Answer question ###
    qa_system_prompt = """You are an assistant for question answer tasks. 
        Use the following pieces of retrieved context to answer the question. \
        If the context doesn't have relavant information to answer the question, say, \n\n
        Sorry, I am not having relavant information to answer the question." \n. 
        Keep the answer concise. \n\n
        Example of an 'question' and 'answer':\
        -----------------------\n \
            Question: What is MSDE's vision for 2025? \n
            Answer: MSDE’s Vision 2025 aims to transition India to a high-skills equilibrium, fostering... \n
                    source: Gen_FAQ_Annex \n\n
        context:\
        -------- \n   {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

    conversational_rag_chain = RunnableWithMessageHistory(
        rag_chain,
        get_session_history,
        input_messages_key="input",
        history_messages_key="chat_history",
        output_messages_key="answer",
    )
    return conversational_rag_chain


def handle_userinput(query):
    if st.session_state.conversation is None:
        st.error("Please upload PDF data before starting the chat.")
        return

    response = st.session_state.conversation.invoke(
        {"input": query},
        config={"configurable": {"session_id": "abc123"}},
    )
    print(response)
    
    st.session_state.chat_history = response['chat_history']
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
            
    st.write(user_template.replace(
                "{{MSG}}", query), unsafe_allow_html=True)
    st.write(bot_template.replace(
                "{{MSG}}", response["answer"]), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Talk with PDF",
                       page_icon="icon.png")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with AI with Custom Data 🚀")
    user_question = st.chat_input("Ask a question about your Data:")

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your Data here  in PDF format and click on 'Process'", accept_multiple_files=True, type=['pdf'])

        if st.button("Process"):
            if pdf_docs is None:
                st.error("Please upload at least one PDF file.")
            else:
                with st.spinner("Processing"):
                    vectorstore = get_vectorstore(pdf_docs)
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Your Data has been processed successfully")

    if user_question:
        handle_userinput(user_question)

    st.markdown(hide_st_style, unsafe_allow_html=True)
    st.markdown(footer, unsafe_allow_html=True)


if __name__ == '__main__':
    main()
