import os, glob, time, base64, platform
import streamlit as st
from PIL import Image
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from gtts import gTTS

# Configuración de la página
st.set_page_config(page_title="ESCLAVO ROBOT 💬", page_icon="🤖", layout="centered")
st.title('Generación Aumentada por Recuperación (ESCLAVO ROBOT) 💬')
st.write("Versión de Python:", platform.python_version())

# Imagen
try:
    st.image(Image.open('Chat_pdf.png'), width=350)
except Exception as e:
    st.warning(f"No se pudo cargar la imagen: {e}")

# Sidebar
with st.sidebar:
    st.subheader("Este Robot te ayudará a estudiar tu PDF, ¡hazle todas las preguntas que quieras!")
    st.write("Sube el PDF y pregunta; el audio saldrá solo.")

# Clave OpenAI
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke: os.environ['OPENAI_API_KEY'] = ke
else:  st.warning("Ingresa tu clave para continuar")

# Cargar PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Utilidades
os.makedirs("temp", exist_ok=True)
def text_to_speech(text:str)->str:
    name = text[:20].strip().replace(" ", "_") or "audio"
    path = f"temp/{name}.mp3"
    if not os.path.exists(path):  # evitar regenerar si ya existe
        gTTS(text, lang='es').save(path)
    return path

def clean_temp(days:int=7):
    limit = time.time() - days*86400
    for f in glob.glob("temp/*.mp3"):
        if os.path.getmtime(f) < limit: os.remove(f)

# Proceso principal
if pdf and ke:
    try:
        text = "".join(p.extract_text() for p in PdfReader(pdf).pages)
        st.info(f"Texto extraído: {len(text)} caracteres")
        splitter = CharacterTextSplitter("\n", 500, 20, len)
        chunks = splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")
        kb = FAISS.from_texts(chunks, OpenAIEmbeddings())

        user_q = st.text_area("Pregunta sobre el documento")

        if user_q:
            docs = kb.similarity_search(user_q)
            respuesta = load_qa_chain(OpenAI(temperature=0, model_name="gpt-4o"), chain_type="stuff")\
                        .run(input_documents=docs, question=user_q)

            st.markdown("### Respuesta:")
            st.markdown(respuesta)

            # Audio automático en español
            audio_file = text_to_speech(respuesta)
            with open(audio_file, "rb") as f:
                st.audio(f.read(), format="audio/mp3")

            # Enlace de descarga
            with open(audio_file, "rb") as f:
                b64 = base64.b64encode(f.read()).decode()
            st.markdown(f'<a href="data:audio/mp3;base64,{b64}" download="{os.path.basename(audio_file)}">Descargar audio</a>',
                        unsafe_allow_html=True)

            clean_temp()

    except Exception as e:
        import traceback
        st.error(f"Error: {e}")
        st.error(traceback.format_exc())
