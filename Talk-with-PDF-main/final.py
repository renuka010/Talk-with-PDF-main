import streamlit as st
import uuid
import chromadb
import pymupdf4llm
import fitz
import re
import os
from dotenv import load_dotenv
from typing import (
    List,
    Tuple
)

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.retrievers.document_compressors import EmbeddingsFilter
from langchain.retrievers import ContextualCompressionRetriever
from langchain_community.chat_models import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

from htmlTemplates import css, bot_template, user_template, hide_st_style, footer

load_dotenv()

def get_chunks(pdf_docs):
    text_chunks: List[str] = []
    for pdf in pdf_docs:

        # Fetch Title and Page Numbers
        pattern = re.compile(r'(.+?)\.{10,}\s*(\d+)')
        contents: List[Tuple] = []
        doc = fitz.open(stream=pdf.read(), filetype="pdf")
        md_text = pymupdf4llm.to_markdown(doc)
        for line in md_text.splitlines():
            line = line.strip()
            match =  pattern.match(line)
            if match:
                title = match.group(1).strip()
                page_num = match.group(2).strip()
                contents.append((title,page_num))

        documents: List[Tuple] = []
        if contents:
            for i in range(len(contents)):
                title = contents[i][0]
                start_page = contents[i][1]
                
                if i == len(contents)-1:
                    next_title = ""
                    end_page = doc.page_count-1
                else:
                    next_title = contents[i+1][0]
                    end_page = contents[i+1][1]

                pages = [j for j in range(int(start_page), int(end_page)+1)]
                md_text = pymupdf4llm.to_markdown(doc, pages= pages)

                cleaned_content = '\n'.join(md_text.split("\n\n"))
                pattern = re.compile(r'\d+\n\n-{5}\n')
                cleaned_content = re.sub(pattern, "", cleaned_content)

                pattern_with_next_title = re.compile(rf'({re.escape(title)})(.*?)(?={re.escape(next_title)})', re.IGNORECASE | re.DOTALL)
                pattern_without_next_title = re.compile(rf'({re.escape(title)})(.*)', re.IGNORECASE | re.DOTALL)

                match = pattern_with_next_title.search(cleaned_content)
                if not match:
                    match = pattern_without_next_title.search(cleaned_content)
                
                if match:
                    final_content = match.group(1) + match.group(2)
                    documents.append((title, final_content))
                else:
                    print(f"No Match for title {title}")
        else:
            cleaned_content = '\n'.join(md_text.split("\n\n"))
            pattern = re.compile(r'\d+\n\n-{5}\n')
            cleaned_content = re.sub(pattern, "", cleaned_content)
            documents.append(("", cleaned_content))
        # Split Documents
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1024, 
            chunk_overlap=200, 
            separators=["\n", "."], 
            length_function=len
        )

        for doc in documents:
            chunks = text_splitter.split_text(doc[1])
            for chunk in chunks:
                text_chunks.append(doc[0]+'\n'+chunk)

    return text_chunks

def generate_id():
    return str(uuid.uuid4().int)

def get_vectorstore(pdf_docs):
    new_collection = False

    embeddings = SentenceTransformerEmbeddings(model_name=os.environ["EMBEDDING_MODEL"])
    collection_name = os.environ["CHROMA_COLLECTION_NAME"]
    persistent_client = chromadb.PersistentClient()
    try:
        collection = persistent_client.get_collection(collection_name)
        print("collection is retrived")
    except:
        collection = persistent_client.create_collection(collection_name)
        print("collection is created")
        new_collection = True

    vectorstore =Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=persistent_client,  # Where to save data locally
    )

    if new_collection:
        text_chunks = get_chunks(pdf_docs)

        ids = [generate_id() for _ in range(len(text_chunks))]
        vectorstore.add_texts(texts = text_chunks,
                              ids=ids)
    return vectorstore

if "store" not in st.session_state:
    st.session_state.store = {}

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    store = st.session_state.store
    if session_id not in store:
        store[session_id] = ChatMessageHistory()

    if len(store[session_id].messages) > 4:
            store[session_id].messages = store[session_id].messages[-4:]

    print("Store history", store[session_id])

    return store[session_id]

def update_chat_history(session_id: str, query, response):
    st.session_state.store[session_id].add_user_message(query)
    st.session_state.store[session_id].add_ai_message(response)

def get_conversation_chain(vector_store):
    # base_url='http://host.docker.internal:11434'
    llm = ChatOllama(model=os.environ["MODEL_NAME"], 
                     format="text", 
                     temperature=0, 
                     keep_alive=-1)
    base_retriever = vector_store.as_retriever(search_kwargs={"k": 5})
    embeddings = SentenceTransformerEmbeddings(model_name=os.environ["EMBEDDING_MODEL"])
    embeddings_filter = EmbeddingsFilter(embeddings=embeddings, 
                                         similarity_threshold=float(os.environ["RELEVANCE_THRESHOLD"]))
    retriever = ContextualCompressionRetriever(base_compressor=embeddings_filter, 
                                               base_retriever=base_retriever)

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

    qa_system_prompt = """You are an AI assistant that can help the user with question answer task.\
    Answer the USER query based on the CONTEXT provided.\n\
    
    Reminder:\n\
    - If the provided context does'nt have relavant information to answer the question, say 'Sorry, I'm not having relavant information currently'.\
    - IMPORTANT - DO NOT ANSWER THE QUESTION IF THE CONTEXT DON'T HAVE RELAVANT INFORMATION.\
    - Keep the answer very concise and easy to undersytand.\
     

    context:\n\
    -------- \n\
    {context}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    )
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    def retrieve_and_check_context(inputs):
        print(inputs)
        question = inputs["input"]
        chat_history = inputs["chat_history"]
        
        # Use the history_aware_retriever to get context
        context = history_aware_retriever.invoke({"input": question, "chat_history": chat_history})
        
        if not context:  # If no relevant context is retrieved
            return {
                "context": "",
                "input": question,
                "chat_history": chat_history,
                "no_context": True
            }
        else:
            return {
                "context": context,
                "input": question,
                "chat_history": chat_history,
                "no_context": False
            }

    def answer_or_apologize(inputs):
        print(inputs,"\n\n")
        if inputs["no_context"]:
            return "Sorry, I don't have relevant information to answer that question."
        else:
            return question_answer_chain.invoke(inputs)

    rag_chain = (
        RunnablePassthrough()
        | retrieve_and_check_context
        | answer_or_apologize
    )
    
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
    session_id = os.environ["SESSION_ID"]

    response = st.session_state.conversation.invoke(
        {"input": query},
        config={"configurable": {"session_id": session_id}},
    )
    print(f'Response ---> {response}')
    ai_response = response

    update_chat_history(session_id, query, ai_response)
    
    st.session_state.chat_history.append(query)
    st.session_state.chat_history.append(ai_response)
    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{MSG}", message), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{MSG}", message), unsafe_allow_html=True)


def main():
    st.set_page_config(page_title="Talk with PDF",
                       page_icon="icon.png")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

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
