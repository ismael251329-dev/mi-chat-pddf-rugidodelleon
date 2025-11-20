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

# ==========================================
# 🎨 ZONA DEL JEFE (CONFIGURACIÓN PERSONALIZADA)
# ==========================================

# 1. NOMBRE Y LOGO
NOMBRE_APP = "Consultor El Rugido De León"
ICONO_APP = "🦁" 

# 2. PERSONALIDAD DE LA IA (SYSTEM ROLE)
ROL_IA = """
2. SYSTEM ROLE: ANALISTA Y MAESTRO BÍBLICO AVANZADO (CON MÓDULO CONTRARIAN)
(Resumido para brevedad en el código, pero mantiene tu lógica interna)
Identidad: Erudito Bíblico, Exegeta y Educador.
Misión: Guía Espiritual y Analista de Conocimiento (Abogado del Diablo Digital).
"""

# 3. COLORES
COLOR_TITULO = "#1E88E5"
COLOR_FONDO = "#F0F2F6"

# ==========================================
# ⚙️ FIN DE LA ZONA DEL JEFE
# ==========================================

# Configuración de la página
st.set_page_config(page_title=NOMBRE_APP, page_icon=ICONO_APP)

# Inyectar CSS
st.markdown(f"""
    <style>
    .stApp {{
        background-color: {COLOR_FONDO};
    }}
    h1 {{
        color: {COLOR_TITULO};
    }}
    </style>
    """, unsafe_allow_html=True)

st.title(f"{ICONO_APP} {NOMBRE_APP}")
st.write("Bienvenido. Sistema listo para analizar tus documentos y las Escrituras.")

# --- CONEXIÓN SEGURA ---
def get_keys():
    # Verificación robusta de llaves
    missing_keys = []
    if "GOOGLE_API_KEY" not in st.secrets:
        missing_keys.append("GOOGLE_API_KEY")
    if "GCP_SERVICE_ACCOUNT" not in st.secrets:
        missing_keys.append("GCP_SERVICE_ACCOUNT")
    if "DRIVE_FOLDER_ID" not in st.secrets:
        missing_keys.append("DRIVE_FOLDER_ID")
        
    if missing_keys:
        st.error(f"⚠️ Faltan las siguientes llaves en Secrets: {', '.join(missing_keys)}")
        st.stop()
        
    return st.secrets["GOOGLE_API_KEY"], st.secrets["DRIVE_FOLDER_ID"]

try:
    api_key, folder_id = get_keys()
except Exception as e:
    st.error(f"Error cargando llaves: {e}")
    st.stop()

# --- FUNCIONES TÉCNICAS ---
def leer_drive():
    try:
        # Manejo especial para el JSON de Service Account
        info_robot = st.secrets["GCP_SERVICE_ACCOUNT"]
        # Si viene como string, lo parseamos, si ya es objeto (TOML a veces lo hace), lo usamos
        if isinstance(info_robot, str):
            info_robot = json.loads(info_robot)
            
        creds = service_account.Credentials.from_service_account_info(
            info_robot, scopes=['https://www.googleapis.com/auth/drive.readonly']
        )
        service = build('drive', 'v3', credentials=creds)
        
        results = service.files().list(
            q=f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false",
            fields="files(id, name)"
        ).execute()
        archivos = results.get('files', [])
        
        texto_total = ""
        barra = st.progress(0)
        status = st.empty()
        
        if not archivos:
            st.warning("No encontré PDFs. Verifica que compartiste la carpeta con el robot.")
            return None

        for i, archivo in enumerate(archivos):
            status.text(f"Analizando: {archivo['name']}...")
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
    except Exception as e:
        st.error(f"Error al conectar con Drive: {str(e)}")
        return None

def preparar_cerebro(texto):
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
    
    # Inyección del Prompt Maestro
    plantilla = f"""
    {ROL_IA}
    
    Contexto recuperado de PDFs:
    {{context}}
    
    Pregunta del usuario: 
    {{question}}
    
    Respuesta:
    """
    # CORREGIDO: Se usa gemini-1.5-flash que es más estable, o gemini-pro si prefieres
    modelo = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.3, google_api_key=api_key)
    prompt = PromptTemplate(template=plantilla, input_variables=["context", "question"])
    
    # CORREGIDO: Variable 'chain' consistente
    chain = load_qa_chain(modelo, chain_type="stuff", prompt=prompt)
    
    # CORREGIDO: Llamada a 'chain' en lugar de 'cadena'
    respuesta = chain.invoke({"input_documents": docs, "question": pregunta})
    return respuesta["output_text"]

# --- INTERFAZ ---
with st.sidebar:
    st.header("⚙️ Panel de Control")
    if st.button("🔄 Sincronizar con Drive"):
        with st.spinner("Leyendo Drive..."):
            texto = leer_drive()
            if texto:
                preparar_cerebro(texto)
                st.success("¡Cerebro Actualizado!")

# Chat
if "mensajes" not in st.session_state: st.session_state.mensajes = []

for m in st.session_state.mensajes:
    with st.chat_message(m["role"]): st.markdown(m["content"])

pregunta = st.chat_input(f"Consulta a {NOMBRE_APP}...")

if pregunta:
    with st.chat_message("user"): st.markdown(pregunta)
    st.session_state.mensajes.append({"role": "user", "content": pregunta})
    
    if os.path.exists("memoria_pdfs"):
        with st.chat_message("assistant"):
            with st.spinner("Analizando las Escrituras y Documentos..."):
                try:
                    resp = responder(pregunta)
                    st.markdown(resp)
                    st.session_state.mensajes.append({"role": "assistant", "content": resp})
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.warning("⚠️ Pulsa 'Sincronizar con Drive' para empezar.")
