### Installation Guide

To install the necessary dependencies for this project, you will need to have Python installed on your machine. Once Python is installed, you can use pip to install the required packages. Here's how you can do it:

```bash
# Clone the repository
git clone https://github.com/Arunprakaash/rag-chatbot.git

# Navigate to the project directory
cd rag-chatbot
```
```bash
# Install the required packages
pip install -r requirements.txt
```
or
```bash
# Install the required packages using poetry
pip install poetry

# Install the required packages
poetry install
```

### Usage of CLI
The CLI can be used to interact with the chatbot. Here's how you can use it:

```bash
# Run the CLI
python cli.py "Your query here"
```
Replace "Your query here" with your actual query.

### Chainlit App
The Chainlit app can be used to interact with the chatbot. Here's how you can use it:

```bash
chainlit run app.py
```

### Demo
![demo gif of user interacting with the chatbot](imgs/demo.gif)

### Project Explanation
This project is a Python-based application that leverages the OpenAI and Pinecone APIs to create a chatbot with context-aware capabilities. 
The project is structured into three main Python files:  openai_utils.py, pinecone_utils.py, and cli.py.  
The openai_utils.py file contains the OpenAIUtils class, which is responsible for interacting with the OpenAI API. 
It has methods for generating embeddings from a given query and for chatting with the OpenAI model.  
The pinecone_utils.py file contains the PineconeUtils class, which is responsible for interacting with the Pinecone API. 
It has methods for initializing a Pinecone index, adding embeddings to the index, and retrieving relevant history based on a query embedding.  '
The cli.py file is the main entry point of the application. It sets up the Pinecone and OpenAI clients, and defines the main function which parses the user's query, retrieves the relevant context from the Pinecone index, and sends the query and context to the OpenAI model for a response. 
The chainlit app (app.py) provides a user-friendly interface to interact with the chatbot. It takes user input, sends it to the chatbot, and displays the response.