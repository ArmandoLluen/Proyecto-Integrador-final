import warnings
import streamlit as st
from services.ocr_service import OCRService
from services.detection_service import DetectionService
from services.nlp_service import NLPService
from services.excel_service import ExcelService
from utils.models import MessageResult
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor
from services.common.common import logger, log_

warnings.filterwarnings("ignore", message=r".*use_container_width.*")

# ---------------- CONFIG ----------------
st.set_page_config(page_title="Chat Analyzer - Acoso Detector", layout="centered")
st.title("Chat Analyzer — Detecta acoso en capturas de chat")

# ---------------- SERVICIOS ----------------
# Inicializar servicios solo una vez
if "ocr_service" not in st.session_state:
    st.session_state.ocr_service = OCRService()
if "detector_service" not in st.session_state:
    st.session_state.detector_service = DetectionService()
if "nlp_service" not in st.session_state:
    st.session_state.nlp_service = NLPService()
if "excel_service" not in st.session_state:
    st.session_state.excel_service = ExcelService(output_dir=".")
if "context_done" not in st.session_state:
    st.session_state.context_done = False
if "context_result" not in st.session_state:
    st.session_state.context_result = None

ocr = st.session_state.ocr_service
detector = st.session_state.detector_service
nlp = st.session_state.nlp_service
excel = st.session_state.excel_service

# ---------------- SESSION STATE ----------------
session_vars = [
    ("uploaded_img", None),
    ("analysis_done", False),
    ("buffer", []),
    ("last_analysis", None),
    ("img_counter", 0),
    ("force_analyze", False),
]
for var, default in session_vars:
    if var not in st.session_state:
        st.session_state[var] = default

# ---------------- 1) SUBIR IMAGEN ----------------
if st.session_state.uploaded_img is None:
    uploaded = st.file_uploader(
        "Sube una captura de chat (png/jpg/jpeg)",
        type=["png", "jpg", "jpeg"]
    )
    if uploaded is not None:
        st.session_state.uploaded_img = uploaded.read()
        st.session_state.analysis_done = False
        st.session_state.last_analysis = None
        st.rerun()
    st.stop()

# ---------------- 2) MOSTRAR IMAGEN ----------------
st.success(f"Cantidad de imagenes en memoria: {st.session_state.img_counter}")
st.image(st.session_state.uploaded_img, caption="Imagen subida", use_container_width=True)

# ---------------- 3) OCR ----------------
with st.spinner("Detectando texto..."):
    try:
        ocr_res = ocr.image_to_lines(st.session_state.uploaded_img)
    except Exception as e:
        log_("error", logger, f" Detección de texto falló en el OCR: {e}")
        st.error(f"Detección de texto falló: {e}")
        st.stop()

# ---------------- 4) DETECTAR CHAT ----------------
file_bytes = np.frombuffer(st.session_state.uploaded_img, np.uint8)
img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
lines = ocr_res.get("lines", [])

if not lines and not st.session_state.force_analyze:
    st.warning("No se detectó texto en la imagen.")
    if st.button("Subir otra imagen"):
        st.session_state.uploaded_img = None
        st.session_state.force_analyze = False
        st.rerun()
    st.stop()

if "bubbles" not in st.session_state or st.session_state.force_analyze or st.session_state.bubbles is None:
    st.session_state.bubbles = detector.messages(ocr_res, img)

bubbles = st.session_state.bubbles
 
if len(bubbles) < 2 and not st.session_state.force_analyze:
    st.warning("La imagen no parece ser una captura de chat.")
    colA, colB = st.columns(2)
    with colA:
        if st.button("Forzar análisis"):
            st.session_state.force_analyze = True
            st.rerun()
    with colB:
        if st.button("Subir otra imagen"):
            st.session_state.uploaded_img = None
            st.session_state.force_analyze = False
            st.rerun()
    st.stop()

if len(bubbles) >= 2:
    st.success(f"Se detectaron {len(bubbles)} burbujas de chat. Parece un chat ✅")
    for i, msg in enumerate([b["text"] for b in bubbles], start=1):
        st.markdown(f"**{i}.** {msg}")
else:
    st.info(f"Se detectaron {len(bubbles)} burbujas de chat. Forzando análisis ⚠️")

# ---------------- 5) DETECTAR MENSAJES (lado izquierdo) ----------------
with st.spinner("Detectando mensajes..."):
    left_msgs = detector.get_received_messages(bubbles)
if not left_msgs:
    st.warning("No se detectaron mensajes recibidos.")
    if st.button("Subir otra imagen"):
            st.session_state.uploaded_img = None
            st.session_state.force_analyze = False
            st.rerun()
else:
    st.subheader("Mensajes recibidos detectados")
    for i, msg in enumerate(left_msgs, start=1):
        st.markdown(f"**{i}.** {msg}")

# ---------------- 6) BOTÓN ANALIZAR ----------------
def bool_to_si_no(value):
    if isinstance(value, str):
        return value
    return "sí" if value else "no"

def analyze_message(nlp_service, m, nro_img):
    analysis = nlp_service.analyze_text(m)
    return MessageResult(
        nro_imagen=nro_img,
        texto=m,
        hostilidad=analysis.get("hostilidad"),
        discriminacion=analysis.get("discriminacion"),
        insultos=analysis.get("insultos"),
        lenguaje_soez=analysis.get("lenguaje_soez"),
        contenido_sexual=analysis.get("contenido_sexual"),
        amenazas=analysis.get("amenazas"),
        humillacion=analysis.get("humillacion"),
        coercion=analysis.get("coercion"),
        es_acoso="probablemente sí" if analysis.get("es_acoso") == "sí" else "no",
        razon=analysis.get("razon", ""),
        probabilidad=analysis.get("probabilidad", 0.0)
    )

if left_msgs and st.button("Analizar mensajes"):
    st.session_state.img_counter += 1
    nro_img = st.session_state.img_counter
    analyses = []

    with st.spinner("Analizando mensajes ..."):
        # Paralelizar de forma segura
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = executor.map(lambda m: analyze_message(nlp, m, nro_img), left_msgs)
            for r in results:
                st.session_state.buffer.append(r)
                analyses.append(r)

    st.session_state.last_analysis = analyses
    st.session_state.analysis_done = True
    st.success(f"Análisis completado.\n Cantidad de mensajes: {len(analyses)}.")
    st.rerun()

# ---------------- 7) MOSTRAR RESUMEN ----------------
if st.session_state.analysis_done and st.session_state.last_analysis:
    st.subheader("Resumen del análisis")
    for i, r in enumerate(st.session_state.last_analysis, start=1):
        st.markdown(f"### Mensaje {i}")
        st.write(f"**Texto:** {r.texto}")

        st.write(f"- Hostilidad: {r.hostilidad}")
        st.write(f"- Insultos: {r.insultos}")
        st.write(f"- Discriminación: {r.discriminacion}")
        st.write(f"- Lenguaje soez: {r.lenguaje_soez}")
        st.write(f"- Contenido sexual: {r.contenido_sexual}")
        st.write(f"- Amenazas: {r.amenazas}")
        st.write(f"- Humillación: {r.humillacion}")
        st.write(f"- Coerción: {r.coercion}")
        st.write(f"- Es acoso: {r.es_acoso}")
        st.write(f"- Probabilidad: {r.probabilidad*100:.2f}%")

        if r.razon:
            st.write(f"- **Razón:** {r.razon}")

        st.markdown("---")

    # ---------------- ANALIZAR CONTEXTO GENERAL ----------------
    if (
        st.session_state.analysis_done 
        and st.session_state.last_analysis 
        and not st.session_state.context_done
    ):

        conversation_texts = [
            (
                "Persona B: " + b["text"]
                if b["text"] in left_msgs
                else "Persona A: " + b["text"]
            )
            for b in bubbles
        ]

        # Analizar contexto general
        conversation_summary = nlp.analyze_conversation_context(conversation_texts)

        resumen_general = MessageResult(
            nro_imagen=st.session_state.img_counter,
            texto="resumen general de imagen",
            hostilidad=conversation_summary.get("hostilidad", "no"),
            discriminacion=conversation_summary.get("discriminacion", "no"),
            insultos=conversation_summary.get("insultos", "no"),
            lenguaje_soez=conversation_summary.get("lenguaje_soez", "no"),
            contenido_sexual=conversation_summary.get("contenido_sexual", "no"),
            amenazas=conversation_summary.get("amenazas", "no"),
            humillacion=conversation_summary.get("humillacion", "no"),
            coercion=conversation_summary.get("coercion", "no"),
            es_acoso=conversation_summary.get("es_acoso", "no"),
            razon=conversation_summary.get("razon", "") + " " + conversation_summary.get("reflexion", ""),
            probabilidad=conversation_summary.get("probabilidad", 0.0)
        )

        # Guardar el resultado contextual sin mezclarlo con mensajes individuales
        st.session_state.context_result = resumen_general
        st.session_state.context_summary_raw = conversation_summary  # guardar para UI
        st.session_state.context_done = True

        # ---------------- MOSTRAR CONTEXTO GENERAL ----------------
        if st.session_state.context_result:
            conversation_summary = st.session_state.context_summary_raw
            st.session_state.buffer.append(resumen_general)

            st.markdown("### Contexto general de la imagen")

            if conversation_summary.get("es_acoso", "no") == "sí":
                st.warning(f"El análisis indica que hay acoso en la conversación. Probabilidad: {conversation_summary.get('probabilidad', 0.0)*100:.2f}%")
            else:
                st.success(f"El análisis indica que no hay acoso en la conversación. Probabilidad {conversation_summary.get('probabilidad', 0.0)*100:.2f}%")

            if conversation_summary.get("reflexion"):
                st.write(conversation_summary["reflexion"])

            st.markdown("---")

# ---------------- 8) BOTONES FINALES ----------------
if st.session_state.analysis_done:
    st.subheader("¿Qué deseas hacer ahora?")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Continuar con otra imagen"):
            st.session_state.uploaded_img = None
            st.session_state.last_analysis = None
            st.session_state.analysis_done = False
            st.session_state.force_analyze = False
            st.session_state.context_done = False
            st.session_state.bubbles = None
            st.rerun()
    with col2:
        if st.button("Descargar Excel") and st.session_state.buffer:
            path = excel.export_messages(st.session_state.buffer)
            with open(path, "rb") as f:
                st.download_button(
                    "Descargar archivo",
                    data=f,
                    file_name="analisis_chat.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            st.session_state.buffer = []
            st.success("Buffer vaciado.")
            st.session_state.uploaded_img = None
            st.session_state.last_analysis = None
            st.session_state.analysis_done = False
            st.session_state.force_analyze = False
            st.session_state.img_counter = 0
            st.session_state.context_done = False
            st.session_state.bubbles = None
            st.rerun()
