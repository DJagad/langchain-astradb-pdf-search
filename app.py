import streamlit as st
from langchain.vectorstores.cassandra import Cassandra
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.llms import OpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import cassio
import os

load_dotenv()

# Setup title and introduction
st.title("PDF Querying with AstraDB and Langchain")
st.write("A question-answering demo using Astra DB and LangChain, powered by Vector Search")

# Prerequisites
st.header("Prerequisites")
st.markdown("You need a Serverless Cassandra with vector Search database on [AstraDB](https://astra.datastax.com) to run this demo. As outlined in more detail [here](https://docs.datastax.com/en/astra/astra-db-vector/index.html), you should get a DB Token with role Database Administrator and copy your Database ID: these connection parameters are needed momentarily.")

# Provide input for AstraDB connection details and OpenAI API Key
ASTRA_DB_APPLICATION_TOKEN = os.getenv("ASTRA_DB_APPLICATION_TOKEN")
ASTRA_DB_ID = os.getenv("ASTRA_DB_ID")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
print(ASTRA_DB_APPLICATION_TOKEN)
print(ASTRA_DB_ID)
print(OPENAI_API_KEY)

# Initialize LangChain components
llm = OpenAI(openai_api_key=OPENAI_API_KEY)

embedding = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

text_splitter = CharacterTextSplitter(separator='\n', chunk_size=800, chunk_overlap=200, length_function=len)


# Provide the path of the PDF file
pdf_file = st.file_uploader("Upload PDF file")

# Main functionality
if pdf_file and ASTRA_DB_APPLICATION_TOKEN and ASTRA_DB_ID and OPENAI_API_KEY:
    #Adding and creating the text file
    st.header("Processing PDF File")

    # Read text from the PDF
    raw_text = ''
    pdf_reader = PdfReader(pdf_file)
    for page in pdf_reader.pages:
        content = page.extract_text()
        if content:
            raw_text += content

    #st.subheader(raw_text)

    print("Inside the Main Functionality")
    st.header("Initializing Database Connection")

    # Initialize the connection of the Database
    cassio.init(token=ASTRA_DB_APPLICATION_TOKEN, database_id=ASTRA_DB_ID)

    st.header("Creating LangChain Embedding and Vector Store")

    # Split text and add to vector store
    texts = text_splitter.split_text(raw_text)
    astra_vector_store = Cassandra(embedding=embedding, table_name="qa_mini_demo_streamlit", session=None, keyspace=None)
    astra_vector_store.add_texts(texts)
    astra_vector_index = VectorStoreIndexWrapper(vectorstore=astra_vector_store)

    st.success("Initialization complete. You can now ask questions.")

    while True:
        query_text = st.text_input("Enter your question:")
        if st.button("Submit"):
            if query_text:
                st.subheader("QUESTION: \"%s\"" % query_text)
                answer = astra_vector_index.query(query_text, llm=llm).strip()
                st.success("ANSWER: \"%s\"" % answer)

                st.subheader("FIRST DOCUMENTS BY RELEVANCE:")
                for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
                    st.write("[%0.4f] \"%s.....\"" % (score, doc.page_content[:84]))

        # Ask the user if they want to continue or exit
        continue_question = st.radio("Do you want to ask another question or exit?", ("Ask another question", "Exit"))
        if continue_question == "Exit":
            break
        st.subheader("End")

    """# Question-Answering Loop
    query_text = st.text_input("Enter your question:")
    if st.button("Submit"):
        if query_text:
            st.subheader("QUESTION: \"%s\"" % query_text)
            answer = astra_vector_index.query(query_text, llm=llm).strip()
            st.success("ANSWER: \"%s\"" % answer)

            st.subheader("FIRST DOCUMENTS BY RELEVANCE:")
            for doc, score in astra_vector_store.similarity_search_with_score(query_text, k=4):
                st.write("[%0.4f] \"%s.....\"" % (score, doc.page_content[:84]))"""

#st.subheader("End")