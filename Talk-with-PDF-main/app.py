import streamlit as st
from PyPDF2 import PdfReader

from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)
import uuid
import chromadb
import json
import csv
import ast
import pandas as pd
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
        chunk_size=1024,
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
    # try:
    #     collection = persistent_client.get_collection("nsdc")
    #     print("collection is retrived")
    # except:
    #     collection = persistent_client.create_collection("nsdc")
    #     print("collection is created")
    #     new_collection = True

    try:
        persistent_client.delete_collection("nsdc")
    except Exception as e:
        pass
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
    # base_url='http://host.docker.internal:11434'
    llm = ChatOllama(model="llama3.1", format="text", temperature=0.3, keep_alive=-1)

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
    qa_system_prompt = """You are an AI assistant that can help the user with a variety of tasks.\
    You have access to the following functions: \
    Use the function "create_csv" to "Get csv link from the given python lists":\
    {tools}\
    
    When the user asks you a question, if you need to use tool, provide ONLY the function call in the format: \
    {function_call_format} \
    
    For example to create a csv as below,\n\
    Eligibility, PMKK scheme \n\
    Nationality, Indian \n\
    ID Proof, voter's ID or Aadhaar card or bank account \n\
    Age, 15 to 45 \n\
    Employment, Unemployed/school or college dropout \n\

    Your function call should be as below. \n\
    {{"function": "create_csv", "columns": [["Eligibility", "Nationality", "ID Proof", "Age", "Employment"],
    ["PMKK scheme", "Indian", "voter's ID or Aadhaar card or bank account", "15 to 45", "Unemployed/school or college dropout"]]}}
    
    Example of question and answers: \n\
        Question: What is eligibility of PMKK scheme? give me in csv format \n\
        Answer: {{"function": "create_csv", "columns": [["Eligibility", "Nationality", "ID Proof", "Age", "Employment"],
    ["PMKK Scheme", "Indian", "voter's ID or Aadhaar card or bank account", "15 to 45", "Unemployed/school or college dropout"]]}}\n\n\

        Question: What is eligibility of PMKK scheme? \n\
        Answer: To be eligible for the Pradhan Mantri Kaushal Vikas Yojana (PMKVY), or Pradhan Mantri Kaushal Kendra (PMKK), you must meet the following criteria: \
            - Be an Indian national \
            - Have a valid identity proof, such as a voter's ID, Aadhaar card, or bank account \
            - Be unemployed or have dropped out of school or college \
            - Be between the ages of 15 and 45 \
            - Fulfill the eligibility criteria for the job role you're applying for \n\n\
        
    
    Reminder:\n\
    - Function calls MUST follow the specified format, start with <function= and end with </function>\
    - Required parameters MUST be specified\
    - Function call parameters should be from the context only.
    - Only call one function at a time\
    - Put the entire function call reply on one line\
    - Use double quotes for string in function call.
    - If there is no function call available, answer the question like normal from the context provided without function call.\
    - DO NOT CALL THE FUNCTION IF THE USER DID NOT ASK CSV FILE.\
    - DO NOT TELL THE USER ABOUT FUNCTION CALLS.\
    - DO NOT TELL USER YOU CAN CREATE CSV IF USER DID NOT ASK.\
    - Keep the answer concise.\
    - If the provided context does'nt have relavant information to answer the question, say 'I DON'T KNOW'.\
    - DO NOT ANSWER THE QUESTION IF THE CONTEXT DON'T HAVE RELAVANT INFORMATION.\
        
    - IMPORTANT - CALL THE FUNCTION ONLY IF USER ASKS FOR A CSV. \n\n\

    context:\n\
    -------- \n\
    {context}"""
    tools = """{
            "name": "create_csv",
            "description": "Get csv link from the given python lists",
            "parameters": {
                "type": "object",
                "properties": {
                    "columns":{
                        "type": "List[List[string]]",
                        "description": "A python list of lists where each inner list contains column name and values to fill for each column that are used to create a csv file"
                    }
                }
                "required": ["columns"],
            },
        }"""
    function_call_format = """{"function": "create_csv", "example_name": "example_value"}"""
    qa_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", qa_system_prompt),
            MessagesPlaceholder("chat_history"),
            ("human", "{input}"),
        ]
    ).partial(tools=tools, function_call_format=function_call_format)
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

def is_csv(response: str):
    return True if response.startswith('{"function":') else False

def create_csv(response: str):
    try:
        json_data = response.strip('{}')
        print(f'Json data:>>>>>>>{json_data}')
        parsed_data = json.loads(f'{{{json_data}}}')
        print(f'Pasered data:>>>>>>>{parsed_data}')
        columns = parsed_data['columns']
        print(type(columns))
        csv_file_name = 'output.csv'
        df = pd.DataFrame(columns).transpose()
        df.to_csv(csv_file_name, header=False, index=False)
        print("csv file created successfully")
        return f"I have created the requested details as csv. To download use the link {csv_file_name}"
    except Exception as e:
        print(e)
        return "Sorry, I am unable to generate csv currently."

def handle_userinput(query):
    if st.session_state.conversation is None:
        st.error("Please upload PDF data before starting the chat.")
        return

    response = st.session_state.conversation.invoke(
        {"input": query},
        config={"configurable": {"session_id": "abc123"}},
    )
    print(response)
    ai_response = response["answer"]
    if (is_csv(ai_response)):
        ai_response = create_csv(ai_response)
    
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
