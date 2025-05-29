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
from gtts import gTTS
import glob
import time
import base64

# Configuración inicial
st.set_page_config(page_title="ESCLAVO ROBOT 💬", page_icon="🤖", layout="centered")
st.title('Generación Aumentada por Recuperación (ESCLAVO ROBOT) 💬')
st.write("Versión de Python:", platform.python_version())

# Carga de imagen
try:
    image = Image.open('Chat_pdf.png')
    st.image(image, width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar
with st.sidebar:
    st.subheader("Este Robot te ayudará a estudiar tu PDF, ¡hazle todas las preguntas que quieras!")
    st.write("Sube el PDF en la parte derecha de la página para poner a trabajar a tu nuevo esclavo.")

# Clave API
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Carga PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Funciones auxiliares
os.makedirs("temp", exist_ok=True)

def text_to_speech(text):
    filename = text[:20].strip().replace(" ", "_") or "audio"
    path = f"temp/{filename}.mp3"
    if not os.path.exists(path):
        tts = gTTS(text, lang='es')
        tts.save(path)
    return path

def remove_old_files(days_old=7):
    now = time.time()
    for f in glob.glob("temp/*.mp3"):
        if os.path.isfile(f) and os.stat(f).st_mtime < now - days_old * 86400:
            os.remove(f)

# Procesamiento del PDF
if pdf and ke:
    try:
        text = "".join(page.extract_text() for page in PdfReader(pdf).pages)
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        splitter = CharacterTextSplitter(separator="\n", chunk_size=500, chunk_overlap=20, length_function=len)
        chunks = splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")
        
        embeddings = OpenAIEmbeddings()
        kb = FAISS.from_texts(chunks, embeddings)
        
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

        if user_question:
            docs = kb.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)

            st.markdown("### Respuesta:")
            st.markdown(response)

            # Generar y reproducir audio automáticamente
            audio_path = text_to_speech(response)
            with open(audio_path, "rb") as audio_file:
                st.audio(audio_file.read(), format="audio/mp3")

            # Descargar audio
            with open(audio_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            st.markdown(f'<a href="data:audio/mp3;base64,{b64}" download="{os.path.basename(audio_path)}">Descargar audio</a>',
                        unsafe_allow_html=True)

            remove_old_files()

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
