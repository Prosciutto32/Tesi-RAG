import os
import queue
import threading
import streamlit as st
from pathlib import Path
from populate_database import clear_database
from populate_database import main as populate
from query_data import query_rag 

def reset_preprompt_callback():
    st.session_state.current_preprompt = st.session_state.default_preprompt

if "default_preprompt" not in st.session_state:
    st.session_state.default_preprompt = "You are a tutor for the course on Computer Networks at Pisa University." \
        " Your goal is to support the students during the lecture and to answer questions about the lecture by having a " \
        "conversation with them. You can generate exercises for the students and correct their answers. You can only answer questions about the course. You should refuse " \
        "to answer any content not part of the course. Always be friendly, and if you cannot answer a question, admit it." \
        " In summary, the tutor is a powerful system that can help with various tasks and provide valuable insight and information on various topics." \
        " Whether you need help with a specific question or just want to have a conversation about a particular topic, Tutor is here to help."
    

if "embedding_function" not in st.session_state:
    st.session_state.embedding_function = "mxbai-embed-large"

# Streamlit App Title
st.title("AI Scholar: Your Intelligent Academic Assistant")

# Sidebar for navigation
app_mode = st.sidebar.radio("Choose section", ["RAG Interface", "Database Management"])
@st.dialog("Create Database")
def database_form():
    dir_name = st.text_input("Database Name")
    st.selectbox(
        "selezionare la funzione di embedding che si desidera utilizzare",
        ("mxbai-embed-large","ryanshillington/Qwen3-Embedding-0.6B","embeddinggemma"),
        key="embedding_function"
    )
    if st.button("Submit"):
        if not os.path.exists("data/"+dir_name):
            os.makedirs("data/"+dir_name)
            st.success(f"Directory '{dir_name}' created successfully!")
            # qualcosa che crei una pagina streamlit che carica una cartella di materiale, magari implementiamola prima
        else:
            st.info(f"Directory '{dir_name}' already exists.")
        st.rerun()

if app_mode == "RAG Interface":
    # Initialize the chat history if it's not already in session state
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Pre-prompt input in the sidebar
    preprompt = st.sidebar.text_area(
        "Write the prompt you want here",
        value=st.session_state.default_preprompt,
        key="current_preprompt"
    )
    st.sidebar.button("Reset Prompt", on_click=reset_preprompt_callback)

    chat_models = ["llama3.2","cogito:3b","alibayram/smollm3", "all"]
    # Model selection in the sidebar
    model_type = st.sidebar.selectbox(
        "Pick the model you want to use",
        embedding_models
    )

    history_length = st.sidebar.slider("Number of history messages to use", min_value=2, max_value=7, value=4)

    directories = [
        d for d in os.listdir("data")
        if os.path.isdir("data/"+d) and (d.endswith("_chroma") )
    ]

    QUERY_FOLDER = st.sidebar.selectbox("choose the database you prefer", directories)

    PREPOCESSING = st.sidebar.selectbox("choose the preprocessing method you like",
                                        ["keyword", "semantic", "keyword + semantic", "all"] )

    # User input for prompt
    if prompt := st.chat_input("Ask your question"):
        # Aggiungi la domanda dell'utente alla cronologia
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Visualizza la domanda
        with st.chat_message("user"):
            st.markdown(prompt)

        if len(st.session_state.messages) > history_length*2:
            history_for_rag = st.session_state.messages[-history_length*2:]
        else:
            if len(st.session_state.messages) < 2:
                history_for_rag =[]
            else:
                history_for_rag = st.session_state.messages
        if (model_type == "all" and PREPOCESSING == "all"):
            with st.chat_message("user"):
                    st.markdown("Ciao User")
            with st.chat_message("assistant"):
                st.markdown("Ciao Assistant")
            with st.spinner("Processing your request, please wait about 3h 30m..."):
                with open("Risposte_ottenute", "a") as f:
                    f.write("Domanda: "+ prompt)
                    f.write("\n")

                risposta_qwen_semantic_llama = query_rag("data/"+"Transport_Mixed_Qwen_chroma", prompt, "llama3.2", preprompt, history_for_rag, "semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Llama 3.2 - Qwen - Semantic\n")
                    f.write(risposta_qwen_semantic_llama)
                risposta_qwen_keyword_llama = query_rag("data/"+"Transport_Mixed_Qwen_chroma", prompt, "llama3.2", preprompt, history_for_rag, "keyword")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Llama 3.2 - Qwen - Keyword\n")
                    f.write(risposta_qwen_keyword_llama)
                risposta_qwen_semantic_keyword_llama = query_rag("data/"+"Transport_Mixed_Qwen_chroma", prompt, "llama3.2", preprompt, history_for_rag, "keyword + semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Llama 3.2 - Qwen - Semantic_Keyword\n")
                    f.write(risposta_qwen_semantic_keyword_llama)


                risposta_mxbai_semantic_llama = query_rag("data/"+"Transport_Mixed_Mxbai_chroma", prompt, "llama3.2", preprompt, history_for_rag, "semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Llama 3.2 - Mxbai - Semantic\n")
                    f.write(risposta_mxbai_semantic_llama)
                risposta_mxbai_keyword_llama = query_rag("data/"+"Transport_Mixed_Mxbai_chroma", prompt, "llama3.2", preprompt, history_for_rag, "keyword")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Llama 3.2 - Mxbai - Keyword\n")
                    f.write(risposta_mxbai_keyword_llama)
                risposta_mxbai_semantic_keyword_llama = query_rag("data/"+"Transport_Mixed_Mxbai_chroma", prompt, "llama3.2", preprompt, history_for_rag, "keyword + semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Llama 3.2 - Mxbai - Semantic_Keyword\n")
                    f.write(risposta_mxbai_semantic_keyword_llama)


                risposta_gemma_semantic_llama = query_rag("data/"+"Transport_Mixed_Gemma_chroma", prompt, "llama3.2", preprompt, history_for_rag, "semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Llama 3.2 - Gemma - Semantic\n")
                    f.write(risposta_gemma_semantic_llama)
                risposta_gemma_keyword_llama = query_rag("data/"+"Transport_Mixed_Gemma_chroma", prompt, "llama3.2", preprompt, history_for_rag, "keyword")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Llama 3.2 - Gemma - Keyword\n")
                    f.write(risposta_gemma_keyword_llama)
                risposta_gemma_semantic_keyword_llama = query_rag("data/"+"Transport_Mixed_Gemma_chroma", prompt, "llama3.2", preprompt, history_for_rag, "keyword + semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Llama 3.2 - Gemma - Semantic_Keyword\n")
                    f.write(risposta_gemma_semantic_keyword_llama)




                risposta_qwen_semantic_cogito = query_rag("data/"+"Transport_Mixed_Qwen_chroma", prompt, "cogito:3b", preprompt, history_for_rag, "semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Cogito - Qwen - Semantic\n")
                    f.write(risposta_qwen_semantic_cogito)
                risposta_qwen_keyword_cogito = query_rag("data/"+"Transport_Mixed_Qwen_chroma", prompt, "cogito:3b", preprompt, history_for_rag, "keyword")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Cogito - Qwen - Keyword\n")
                    f.write(risposta_qwen_keyword_cogito)
                risposta_qwen_semantic_keyword_cogito = query_rag("data/"+"Transport_Mixed_Qwen_chroma", prompt, "cogito:3b", preprompt, history_for_rag, "keyword + semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Cogito - Qwen - Semantic_Keyword\n")
                    f.write(risposta_qwen_semantic_keyword_cogito)


                risposta_mxbai_semantic_cogito = query_rag("data/"+"Transport_Mixed_Mxbai_chroma", prompt, "cogito:3b", preprompt, history_for_rag, "semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Cogito - Mxbai - Semantic\n")
                    f.write(risposta_mxbai_semantic_cogito)
                risposta_mxbai_keyword_cogito  = query_rag("data/"+"Transport_Mixed_Mxbai_chroma", prompt, "cogito:3b", preprompt, history_for_rag, "keyword")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Cogito - Mxbai - Keyword\n")
                    f.write(risposta_mxbai_keyword_cogito)
                risposta_mxbai_semantic_keyword_cogito = query_rag("data/"+"Transport_Mixed_Mxbai_chroma", prompt, "cogito:3b", preprompt, history_for_rag, "keyword + semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Cogito - Mxbai - Semantic_Keyword\n")
                    f.write(risposta_mxbai_semantic_keyword_cogito)


                risposta_gemma_semantic_cogito = query_rag("data/"+"Transport_Mixed_Gemma_chroma", prompt, "cogito:3b", preprompt, history_for_rag, "semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Cogito - Gemma - Semantic\n")
                    f.write(risposta_gemma_semantic_cogito)
                risposta_gemma_keyword_cogito  = query_rag("data/"+"Transport_Mixed_Gemma_chroma", prompt, "cogito:3b", preprompt, history_for_rag, "keyword")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Cogito - Gemma - Keyword\n")
                    f.write(risposta_gemma_keyword_cogito)
                risposta_gemma_semantic_keyword_cogito = query_rag("data/"+"Transport_Mixed_Gemma_chroma", prompt, "cogito:3b", preprompt, history_for_rag, "keyword + semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Cogito - Gemma - Semantic_Keyword\n")
                    f.write(risposta_gemma_semantic_keyword_cogito)


                

                risposta_qwen_semantic_smollm = query_rag("data/"+"Transport_Mixed_Qwen_chroma", prompt, "alibayram/smollm3", preprompt, history_for_rag, "semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Smollm - Qwen - Semantic \n")
                    f.write(risposta_qwen_semantic_smollm)
                risposta_qwen_keyword_smollm = query_rag("data/"+"Transport_Mixed_Qwen_chroma", prompt, "alibayram/smollm3", preprompt, history_for_rag, "keyword")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Smollm - Qwen - Keyword \n")
                    f.write(risposta_qwen_keyword_smollm)
                risposta_qwen_semantic_keyword_smollm = query_rag("data/"+"Transport_Mixed_Qwen_chroma", prompt, "alibayram/smollm3", preprompt, history_for_rag, "keyword + semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Smollm - Qwen - Semantic_Keyword\n")
                    f.write(risposta_qwen_semantic_keyword_smollm)


                risposta_mxbai_semantic_smollm = query_rag("data/"+"Transport_Mixed_Mxbai_chroma", prompt, "alibayram/smollm3", preprompt, history_for_rag, "semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Smollm - Mxbai - Semantic\n")
                    f.write(risposta_mxbai_semantic_smollm)
                risposta_mxbai_keyword_smollm  = query_rag("data/"+"Transport_Mixed_Mxbai_chroma", prompt, "alibayram/smollm3", preprompt, history_for_rag, "keyword")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Smollm - Mxbai - Keyword\n")
                    f.write(risposta_mxbai_keyword_smollm)
                risposta_mxbai_semantic_keyword_smollm = query_rag("data/"+"Transport_Mixed_Mxbai_chroma", prompt, "alibayram/smollm3", preprompt, history_for_rag, "keyword + semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Smollm - Mxbai - Semantic_Keyword\n")
                    f.write(risposta_mxbai_semantic_keyword_smollm)


                risposta_gemma_semantic_smollm  = query_rag("data/"+"Transport_Mixed_Gemma_chroma", prompt, "alibayram/smollm3", preprompt, history_for_rag, "semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Smollm - Gemma - Semantic\n")
                    f.write(risposta_gemma_semantic_smollm)
                risposta_gemma_keyword_smollm  = query_rag("data/"+"Transport_Mixed_Gemma_chroma", prompt, "alibayram/smollm3", preprompt, history_for_rag, "keyword")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Smollm - Gemma - Keyword\n")
                    f.write(risposta_gemma_keyword_smollm)
                risposta_gemma_semantic_keyword_smollm = query_rag("data/"+"Transport_Mixed_Gemma_chroma", prompt, "alibayram/smollm3", preprompt, history_for_rag, "keyword + semantic")
                with open("Risposte_ottenute", "a") as f:
                    f.write("Smollm - Gemma - Semantic_Keyword\n")
                    f.write(risposta_gemma_semantic_keyword_cogito)


            with st.chat_message("user"):
                st.markdown("Llama 3.2 - Qwen - Semantic")
            with st.chat_message("assistant"):
                st.markdown(risposta_qwen_semantic_llama)

            with st.chat_message("user"):
                st.markdown("Llama 3.2 - Qwen - Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_qwen_keyword_llama)

            with st.chat_message("user"):
                st.markdown("Llama 3.2 - Qwen - Semantic_Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_qwen_semantic_keyword_llama)

            with st.chat_message("user"):
                st.markdown("Llama 3.2 - Mxbai - Semantic")
            with st.chat_message("assistant"):
                st.markdown(risposta_mxbai_semantic_llama)

            with st.chat_message("user"):
                st.markdown("Llama 3.2 - Mxbai - Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_mxbai_keyword_llama)

            with st.chat_message("user"):
                st.markdown("Llama 3.2 - Mxbai - Semantic_Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_mxbai_semantic_keyword_llama)
            
            with st.chat_message("user"):
                st.markdown("Llama 3.2 - Gemma - Semantic")
            with st.chat_message("assistant"):
                st.markdown(risposta_gemma_semantic_llama)

            with st.chat_message("user"):
                st.markdown("Llama 3.2 - Gemma - Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_gemma_keyword_llama)

            with st.chat_message("user"):
                st.markdown("Llama 3.2 - Gemma - Semantic_Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_gemma_semantic_keyword_llama)
            
            with st.chat_message("user"):
                st.markdown("Cogito - Qwen - Semantic")
            with st.chat_message("assistant"):
                st.markdown(risposta_qwen_semantic_cogito)

            with st.chat_message("user"):
                st.markdown("Cogito - Qwen - Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_qwen_keyword_cogito)

            with st.chat_message("user"):
                st.markdown("Cogito - Qwen - Semantic_Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_qwen_semantic_keyword_cogito)

            with st.chat_message("user"):
                st.markdown("Cogito - Mxbai - Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_mxbai_semantic_cogito)

            with st.chat_message("user"):
                st.markdown("Cogito - Mxbai - Semantic")
            with st.chat_message("assistant"):
                st.markdown(risposta_mxbai_keyword_cogito)

            with st.chat_message("user"):
                st.markdown("Cogito - Mxbai - Semantic_Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_mxbai_semantic_keyword_cogito)
            
            with st.chat_message("user"):
                st.markdown("Cogito - Gemma - Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_gemma_semantic_cogito)

            with st.chat_message("user"):
                st.markdown("Cogito - Gemma - Semantic")
            with st.chat_message("assistant"):
                st.markdown(risposta_gemma_keyword_cogito)

            with st.chat_message("user"):
                st.markdown("Cogito - Gemma - Semantic_Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_gemma_semantic_keyword_cogito)
            
            
            
            with st.chat_message("user"):
                st.markdown("Smollm - Qwen - Semantic")
            with st.chat_message("assistant"):
                st.markdown(risposta_qwen_semantic_smollm)

            with st.chat_message("user"):
                st.markdown("Smollm - Qwen - Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_qwen_keyword_smollm)

            with st.chat_message("user"):
                st.markdown("Smollm - Qwen - Semantic_Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_qwen_semantic_keyword_smollm)




            with st.chat_message("user"):
                st.markdown("Smollm - Mxbai - Semantic")
            with st.chat_message("assistant"):
                st.markdown(risposta_mxbai_semantic_smollm)

            with st.chat_message("user"):
                st.markdown("Smollm - Mxbai - Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_mxbai_keyword_smollm)

            with st.chat_message("user"):
                st.markdown("Smollm - Mxbai - Semantic_Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_mxbai_semantic_keyword_smollm)
            


            with st.chat_message("user"):
                st.markdown("Smollm - Gemma - Semantic")
            with st.chat_message("assistant"):
                st.markdown(risposta_gemma_semantic_smollm)

            with st.chat_message("user"):
                st.markdown("Smollm - Gemma - Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_gemma_keyword_smollm)

            with st.chat_message("user"):
                st.markdown("Smollm - Gemma - Semantic_Keyword")
            with st.chat_message("assistant"):
                st.markdown(risposta_gemma_semantic_keyword_smollm)

            
            

        else:
            with st.spinner("Processing your request, please wait..."):
                

                risposta = query_rag("data/"+QUERY_FOLDER, prompt, model_type, preprompt, history_for_rag, PREPOCESSING)


            # Visualizza la risposta del modello
            with st.chat_message("assistant"):
                st.markdown(risposta)

            # Aggiungi la risposta del modello alla cronologia
            st.session_state.messages.append({"role": "assistant", "content": risposta})

elif app_mode == "Database Management":
    
    if "db_cleared" not in st.session_state:
        st.session_state.db_cleared = False
    if "db_updated" not in st.session_state:
        st.session_state.db_updated = False

    directories = [
        d for d in os.listdir("data")
        if os.path.isdir("data/"+d) and not (d.endswith("_chroma") or d.startswith("__"))
    ]

    UPLOAD_FOLDER = st.sidebar.selectbox("Select the database", directories)
    if UPLOAD_FOLDER:
        directory = Path("data/" + UPLOAD_FOLDER)
        directory.mkdir(parents=True, exist_ok=True)

    # File uploader for documents
    uploaded_files = st.file_uploader(
        "Upload document",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed",
    )

    if st.button("Upload files to the database"):
        if not uploaded_files:
            st.warning("No files selected. Please upload at least one file.")
        else:
            with st.spinner("Uploading files..."):
                for uploaded_file in uploaded_files:
                    file_path = directory / uploaded_file.name
                    with open(file_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())

            # Progress bar setup
            progress_bar = st.progress(0)
            progress_text = st.empty()
            progress_queue = queue.Queue()

            def update_progress_main_thread():
                text=""
                processed = 0
                total = None
                while True:
                    if total == None:
                        progress_bar.progress(0)
                        progress_text.text(f"processing the input, this may take a feaw seconds")

                    try:
                        msg = progress_queue.get(timeout=0.2)
                        if msg == (-1, -1, ""):
                            break
                        processed, total, state = msg
                        if (state == "split"):
                            text = (f"{processed} of {total} documents splitted...")
                        if (state == "insert"):
                            text = (f"{processed} of {total} documents inserted...")
                        percent = int(processed / total * 100)
                        progress_bar.progress(percent)
                        progress_text.text(text)
                    except queue.Empty:
                        continue

            # Callback for progress
            def progress_callback(processed, total, state):
                progress_queue.put((processed, total, state))


            # Run populate in separate thread
            populate_thread = threading.Thread(
                target=populate,
                kwargs={"model":st.session_state.embedding_function, "folder": "data/" + str(UPLOAD_FOLDER), "progress_callback": progress_callback}
            )
            populate_thread.start()

            # Update progress bar
            update_progress_main_thread()

            # Set flag + rerun
            st.session_state.db_updated = True
            st.rerun()

    if st.button("Delete database"):
        clear_database("data/" + UPLOAD_FOLDER)
        st.session_state.db_cleared = True
        st.rerun()

    if st.session_state.db_updated:
        st.success("🎉 Files uploaded and database updated!")
        st.session_state.db_updated = False

    if st.session_state.db_cleared:
        st.success("🎉 Database emptied, files deleted!")
        st.session_state.db_cleared = False

    if st.sidebar.button("Create new database"):
        with st.form("new_database"):
            database_form()
