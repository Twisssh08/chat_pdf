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
    st.write("sube el pdf en la parte derecha de la página para poner a trabajar a tu nuevo esclavo!")

# Clave API
ke = st.text_input('Ingresa tu Clave de OpenAI', type="password")
if ke:
    os.environ['OPENAI_API_KEY'] = ke
else:
    st.warning("Por favor ingresa tu clave de API de OpenAI para continuar")

# Carga PDF
pdf = st.file_uploader("Carga el archivo PDF", type="pdf")

# Procesamiento del PDF
if pdf is not None and ke:
    try:
        pdf_reader = PdfReader(pdf)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
        
        st.info(f"Texto extraído: {len(text)} caracteres")
        
        text_splitter = CharacterTextSplitter(
            separator="\n", chunk_size=500, chunk_overlap=20, length_function=len
        )
        chunks = text_splitter.split_text(text)
        st.success(f"Documento dividido en {len(chunks)} fragmentos")
        
        embeddings = OpenAIEmbeddings()
        knowledge_base = FAISS.from_texts(chunks, embeddings)
        
        st.subheader("Escribe qué quieres saber sobre el documento")
        user_question = st.text_area(" ", placeholder="Escribe tu pregunta aquí...")

        if user_question:
            docs = knowledge_base.similarity_search(user_question)
            llm = OpenAI(temperature=0, model_name="gpt-4o")
            chain = load_qa_chain(llm, chain_type="stuff")
            response = chain.run(input_documents=docs, question=user_question)
            
            st.markdown("### Respuesta:")
            st.markdown(response)

            # Sección de texto a voz
            st.subheader("Texto para convertir a audio")
            texto_audio = st.text_area("Texto que se convertirá en audio:", value=response)

            option_lang = st.selectbox("Selecciona el idioma del audio", ("Español", "English"))
            lg = 'es' if option_lang == "Español" else 'en'

            try:
                os.mkdir("temp")
            except:
                pass

            def text_to_speech(text, tld, lg):
                tts = gTTS(text, lang=lg)
                file_name = text[:20].strip().replace(" ", "_")
                tts.save(f"temp/{file_name}.mp3")
                return file_name

            if st.button("Convertir a Audio"):
                filename = text_to_speech(texto_audio, 'com', lg)
                audio_path = f"temp/{filename}.mp3"
                with open(audio_path, "rb") as audio_file:
                    st.audio(audio_file.read(), format="audio/mp3")

                def get_download_link(file_path):
                    with open(file_path, "rb") as f:
                        data = f.read()
                    b64 = base64.b64encode(data).decode()
                    href = f'<a href="data:application/octet-stream;base64,{b64}" download="{os.path.basename(file_path)}">Descargar Audio</a>'
                    return href

                st.markdown(get_download_link(audio_path), unsafe_allow_html=True)

            def remove_old_files(days_old=7):
                now = time.time()
                limit = days_old * 86400
                for f in glob.glob("temp/*.mp3"):
                    if os.path.isfile(f) and os.stat(f).st_mtime < now - limit:
                        os.remove(f)

            remove_old_files()

    except Exception as e:
        st.error(f"Error al procesar el PDF: {str(e)}")
        import traceback
        st.error(traceback.format_exc())
