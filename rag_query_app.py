from io import BytesIO
import streamlit as st
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
import cassio
from langchain_groq import ChatGroq



import os
from dotenv import load_dotenv
load_dotenv()

##

def truncate_db(table_name):
    s = cassio.config.resolve_session()
    keyspace_name = 'default_keyspace'
    table_name = astra_vector_store.table_name
    truncate_table_query = f"TRUNCATE {keyspace_name}.{table_name}"
    s.execute(truncate_table_query)

# Function to process PDF and extract text
@st.cache_data
def process_pdf(uploaded_file):
    bytes_data = uploaded_file.getvalue()
    pdf_stream = BytesIO(bytes_data)
    pdf_reader = PyPDF2.PdfReader(pdf_stream)
    pdf_text = ""
    
    # Extract text
    for page in pdf_reader.pages:
        pdf_text += page.extract_text()

    ## Split Text
    split_pdf_text = text_spliter.split_text(pdf_text)

    ## add embedding into db
    astra_vector_store.add_texts(split_pdf_text)
    
    

@st.cache_resource
def init():
    st.title("PDF RAG QnA APP")
    os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
    os.environ["ASTRA_DB_APPLICATION_TOKEN"] = st.secrets["ASTRA_DB_APPLICATION_TOKEN"]
    os.environ["ASTRA_DB_ID"] = st.secrets["ASTRA_DB_ID"]
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
    os.environ["LANGCHAIN_PROJECT"] = "RAG PDF QnA"
    os.environ['LANGCHAIN_API_KEY'] = st.secrets["LANGCHAIN_API_KEY"] 
    os.environ['LANGCHAIN_TRACING_V2'] = "true"

    ## init db connection
    cassio.init(token=os.getenv('ASTRA_DB_APPLICATION_TOKEN'), database_id= os.getenv('ASTRA_DB_ID'))

    text_spliter = RecursiveCharacterTextSplitter(chunk_size = 500, chunk_overlap= 100)
    ## Embeddings
    embedding = OpenAIEmbeddings()

    ## vector store

    astra_vector_store = Cassandra(embedding=embedding, table_name='rag_test')

    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

    ## llm
    llm = ChatGroq(model="llama3-70b-8192")

    return astra_vector_store, astra_vector_index, llm ,text_spliter

astra_vector_store, astra_vector_index, llm ,text_spliter = init()


i = 0
uploaded_file = st.file_uploader("Choose a file")

# Initialize session state for uploaded file
if "uploaded_file" not in st.session_state:
    st.session_state.uploaded_file = None

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

if uploaded_file is not None:
    # If a new file is uploaded, process it
    if uploaded_file != st.session_state.uploaded_file:
        st.session_state.uploaded_file = uploaded_file
        pdf_text = process_pdf(uploaded_file)
        # Show success message
        success = st.success("Pdf processed successfully!")
    
    
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    # Accept user input
    if prompt := st.chat_input("What is up?"):
        
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        with st.spinner("Thinking..."):
            assitant_response= astra_vector_index.query(prompt, llm=llm)
            # Display assistant response in chat message container
            with st.chat_message("assistant"):
                st.session_state.messages.append({"role": "assistant", "content": assitant_response})
                st.markdown(assitant_response)

else:
    # Clear the cache if the file is removed
    truncate_db("rag_test")
    if st.session_state.uploaded_file is not None:
        st.session_state.uploaded_file = None
        process_pdf.clear()
    if "messages"  in st.session_state:
        st.session_state.messages = []
