# Guide to the RAG Tutor

## Dependencies
  Installation of Dependecies:
  
        pip3 install streamlit
        pip3 install rake-nltk
        pip3 install langchain
        pip3 install langchain-community
        pip3 install chromadb
        pip3 install rank_bm25
        pip3 install langchain-experimental
        pip3 install langchain_ollama
        pip3 install langchain_chroma
  Installation on nltk packets:
  
        python
        import nltk
        nltk.download('stopwords')
        nltk.download('punkt_tab')
  
  Install Ollama: \n
        install olama from the Ollama doc
        download all the models present in the 2 arrays chat_models & embedding_models
  
## Run
  To run the code from the folder write streamlit run streamilt_unito

## Usage
  there are 2 pages the DB one and the Chat one, the Chat is a normal Chat AI interface with lots of configurable tools
  th DB page is usefull to create a new data base, using the Embedding model you want and then in the Chat inteface use it (you have to select the correct Database)

  
