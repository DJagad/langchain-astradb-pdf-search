# PDF Querying with AstraDB and Langchain

This repository contains a demo application for question-answering using AstraDB and LangChain, powered by Vector Search.

## Prerequisites
Before running this demo, you need:
- A Serverless Cassandra with Vector Search database on [AstraDB](https://astra.datastax.com).
- Obtain a DB Token with role Database Administrator and copy your Database ID. These connection parameters are needed for setup.

## Seting up the variables
Ensure you have the following environment variables configured:
- `ASTRA_DB_APPLICATION_TOKEN`: AstraDB Application Token
- `ASTRA_DB_ID`: AstraDB Database ID
- `OPENAI_API_KEY`: OpenAI API Key

## Setup
To run this chatbot, you'll need to have Python installed on your machine. Follow these steps to set up the environment:

1. Clone this repository to your local machine:

    ```bash
    git clone https://github.com/DJagad/langchain-astradb-pdf-search.git
    ```

2. Navigate to the project directory:

    ```bash
    cd langchain-astradb-pdf-search
    ```

3. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

4. Set up your OpenAI API key by creating a `.env` file in the root directory of the project and adding your API key:

    ```plaintext
    - `ASTRA_DB_APPLICATION_TOKEN`: AstraDB Application Token
    - `ASTRA_DB_ID`: AstraDB Database ID
    - `OPENAI_API_KEY`: OpenAI API Key
    ```

5. Run the following command to start the Streamlit app:

```bash
streamlit run app.py
```

## Functionality
1. **Processing PDF File**: Upload a PDF file to extract text from.
2. **Initializing Database Connection**: Connect to AstraDB to store text data.
3. **Creating LangChain Embedding and Vector Store**: Split text and add to vector store for question-answering.
4. **Question-Answering Loop**: Enter a question and get the answer from the stored text data.
5. **End**: Conclusion of the demo.

## Usage
1. Upload a PDF file.
2. Input your question.
3. Click on the "Submit" button to get the answer.
4. View the top relevant documents by relevance.

## Technologies Used
- Streamlit: For building interactive web applications.
- PyPDF2: For extracting text from PDF files.
- AstraDB: For storing text data and performing vector searches.
- LangChain: For natural language processing tasks such as question-answering and text embeddings.
- OpenAI: For accessing language models and embeddings.

## Acknowledgements
- This demo application is built using Streamlit, PyPDF2, AstraDB, LangChain, and OpenAI.