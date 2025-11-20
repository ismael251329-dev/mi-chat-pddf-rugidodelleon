import streamlit as st
import os
import io
import json
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate

# --- CONFIGURACIÓN VISUAL ---
st.set_page_config(page_title="Mi Chat PDF", page_icon="🦁")

# Estilos para que se vea limpio
st.markdown("""
    <style>
    .main {
        background-color: #F0F2F6;
    }
    h1 {
        color: #1E88E5;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🦁 Chat PDF - Rugido del León")
st.write("Bienvenido. Pregúntame lo que necesites sobre tus documentos.")

# --- CONEXIÓN CON LOS SECRETOS (LA CAJA FUERTE) ---
def get_keys():
    # Verifica si las llaves existen en la caja fuerte de Streamlit
    if "GOOGLE_API_KEY" not in st.secrets or "GCP_SERVICE_ACCOUNT" not in st.secrets or "DRIVE_FOLDER_ID" not in st.secrets:
        st.error("⚠️ Faltan las llaves en la configuración (Secrets).")
        st.stop()
    return st.secrets["GOOGLE_API_KEY"], st.secrets["DRIVE_FOLDER_ID"]

# Intentamos obtener las llaves de forma segura
try:
    api_key, folder_id = get_keys()
except FileNotFoundError:
    st.stop()

# --- FUNCIONES DEL ROBOT ---
def leer_drive():
    # Convertir la llave del robot (que es texto) a un objeto que Google entienda
    info_robot = json.loads(st.secrets["GCP_SERVICE_ACCOUNT"])
    creds = service_account.Credentials.from_service_account_info(
        info_robot, scopes=['https://www.googleapis.com/auth/drive.readonly']
    )
    service = build('drive', 'v3', credentials=creds)
    
    # Buscar PDFs
    results = service.files().list(
        q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
        fields="files(id, name)"
    ).execute()
    archivos = results.get('files', [])
    
    texto_total = ""
    barra = st.progress(0)
    status = st.empty()
    
    if not archivos:
        st.warning("No encontré PDFs en esa carpeta. ¿Compartiste la carpeta con el email del robot?")
        return None

    for i, archivo in enumerate(archivos):
        status.text(f"Leyendo: {archivo['name']}...")
        request = service.files().get_media(fileId=archivo['id'])
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while done is False: status_d, done = downloader.next_chunk()
        fh.seek(0)
        try:
            lector = PdfReader(fh)
            for pagina in lector.pages: texto_total += pagina.extract_text() or ""
        except: pass
        barra.progress((i + 1) / len(archivos))
    
    status.empty()
    barra.empty()
    return texto_total

def preparar_cerebro(texto):
    # Cortar el texto en pedacitos y guardarlo en la memoria
    cortador = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    pedacitos = cortador.split_text(texto)
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    base_datos = FAISS.from_texts(pedacitos, embedding=embeddings)
    base_datos.save_local("memoria_pdfs")
    return True

def responder(pregunta):
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    base_datos = FAISS.load_local("memoria_pdfs", embeddings, allow_dangerous_deserialization=True)
    docs = base_datos.similarity_search(pregunta)
    
    plantilla = """
    Eres un asistente inteligente y útil. Responde usando solo la información siguiente.
    Si no sabes la respuesta, di "No encontré eso en los documentos".
    
    Contexto: {context}
    Pregunta: {question}
    
    Respuesta:
    """
    modelo = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=plantilla, input_variables=["context", "question"])
    cadena = load_qa_chain(modelo, chain_type="stuff", prompt=prompt)
    
    respuesta = cadena({"input_documents": docs, "question": pregunta}, return_only_outputs=True)
    return respuesta["output_text"]

# --- LA PANTALLA QUE VES (SIDEBAR) ---
with st.sidebar:
    st.header("⚙️ Panel de Control")
    if st.button("🔄 Actualizar Memoria"):
        with st.spinner("El robot está leyendo tus archivos en Drive..."):
            try:
                texto = leer_drive()
                if texto:
                    preparar_cerebro(texto)
                    st.success("¡Listo! Ya me aprendí tus documentos.")
            except Exception as e:
                st.error(f"Error: {e}")
    st.info("Pulsa el botón cada vez que subas un nuevo PDF a Drive.")

# --- CHAT ---
if "mensajes" not in st.session_state: st.session_state.mensajes = []

for m in st.session_state.mensajes:
    with st.chat_message(m["role"]): st.markdown(m["content"])

pregunta = st.chat_input("Escribe tu pregunta aquí...")

if pregunta:
    with st.chat_message("user"): st.markdown(pregunta)
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    
    if os.path.exists("memoria_pdfs"):
        with st.chat_message("assistant"):
            with st.spinner("Pensando..."):
                try:
                    resp = responder(pregunta)
                    st.markdown(resp)
                    st.session_state.mensajes.append({"role": "assistant", "content": resp})
                except Exception as e:
                    st.error(f"Ocurrió un error al responder: {e}")
    else:
        st.warning("⚠️ Primero pulsa 'Actualizar Memoria' en el menú de la izquierda.")
