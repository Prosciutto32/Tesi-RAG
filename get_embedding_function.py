from langchain_ollama import OllamaEmbeddings


def get_embedding_function(fine_tuned_model_path):
    embeddings = OllamaEmbeddings(model=fine_tuned_model_path)
    #embeddings = HuggingFaceEmbeddings(model_name=fine_tuned_model_path)
    return embeddings


    