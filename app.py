import os
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import platform
import time
import glob
import base64
import platform


# Configuración visual
st.set_page_config(page_title="ESCLAVO ROBOT 📚💔", page_icon="🤖", layout="centered")
st.markdown(
    '<style>body {background-color: #1e1e2f; color: #fce4ec; font-family: "Courier New", monospace;} '
    'h1, h2, h3 {color: #ff80ab;} .stButton>button {background-color: #ffb6b9; color: #ffffff; border-radius: 10px; padding: 8px 20px;} '
    '.stButton>button:hover {background-color: #f48fb1;}</style>',
    unsafe_allow_html=True
)

st.title('ESCLAVO ROBOT 📚💔')
st.write("Versión de Python:", platform.python_version())

# Imagen decorativa

# Sidebar
with st.sidebar:
    st.subheader("Sube un PDF y pregúntale lo que quieras. Luego escucha la respuesta.")

# Crear carpeta temporal si no existe
if not os.path.exists("temp"):
    os.makedirs("temp")

# Clave API
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Subir PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Procesamiento si se carga PDF y clave API
if pdf and ke:
    try:
        pdf_reader = PdfReader(pdf)
        text = "".join(page.extract_text() for page in pdf_reader.pages if page.extract_text())
        
        # Dividir texto
        splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20)
        chunks = splitter.split_text(text)
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)

        st.subheader("Escribe tu pregunta sobre el PDF")
        user_question = st.text_area(" ", placeholder="¿Qué quieres saber?")
        
        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            
            st.markdown("### Respuesta:")
            st.markdown(response)

            # Convertir respuesta a audio
            def text_to_speech(text, lang='es'):
                tts = gTTS(text, lang=lang)
                filename = "temp/response.mp3"
                tts.save(filename)
                return filename

            audio_file = text_to_speech(response)
            with open(audio_file, "rb") as f:
                audio_bytes = f.read()
                st.audio(audio_bytes, format="audio/mp3")

            def get_binary_file_downloader_html(bin_file, file_label='Audio'):
                with open(bin_file, 'rb') as f:
                    data = f.read()
                bin_str = base64.b64encode(data).decode()
                href = f'<a href="data:application/octet-stream;base64,{bin_str}" download="{os.path.basename(bin_file)}">⬇️ Descargar {file_label}</a>'
                return href

            st.markdown(get_binary_file_downloader_html(audio_file), unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")

# Limpieza automática de archivos antiguos
def remove_old_files(days=7):
    now = time.time()
    for f in glob.glob("temp/*.mp3"):
        if os.stat(f).st_mtime < now - days * 86400:
            os.remove(f)

remove_old_files()
