import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import tempfile
import numpy as np
import os
from collections import deque

# Configurazione pagina
st.set_page_config(page_title="Barbell Tracker Pro", layout="wide")

# Funzione per processare il video
def process_video(input_path, output_path, init_bbox):
    cap = cv2.VideoCapture(input_path)
    
    # Prepara il writer per salvare il video
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    # Codec per il web (H264 è lo standard, ma usiamo mp4v per compatibilità OpenCV base)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    # Inizializza Tracker
    tracker = cv2.TrackerCSRT_create()
    
    # Leggi primo frame per inizializzare
    ret, frame = cap.read()
    if not ret: return
    
    tracker.init(frame, init_bbox)

    # Variabili per la scia
    trajectory_points = []
    # Smoothing factor (media mobile)
    recent_points = deque(maxlen=5) 

    # Barra di progresso Streamlit
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_bar = st.progress(0)
    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            center_x = int(x + w/2)
            center_y = int(y + h/2)
            
            # Smoothing
            recent_points.append((center_x, center_y))
            avg_x = int(sum(p[0] for p in recent_points) / len(recent_points))
            avg_y = int(sum(p[1] for p in recent_points) / len(recent_points))
            
            trajectory_points.append((avg_x, avg_y))

            # Disegna il box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Disegna la scia sul video
            if len(trajectory_points) > 1:
                for i in range(1, len(trajectory_points)):
                    # Effetto sfumatura colore
                    color_intensity = int(255 * (i / len(trajectory_points)))
                    # BGR format -> Verde sfumato
                    cv2.line(frame, trajectory_points[i-1], trajectory_points[i], (0, color_intensity, 255), 3)
            
            # Pallino centrale
            cv2.circle(frame, (avg_x, avg_y), 5, (0, 0, 255), -1)

        # Scrivi il frame nel file di output
        out.write(frame)
        
        # Aggiorna barra progresso
        frame_count += 1
        progress_bar.progress(min(frame_count / total_frames, 1.0))

    cap.release()
    out.release()
    progress_bar.empty()

# --- INTERFACCIA UTENTE ---

st.title("🏋️ Barbell Tracker AI - Web Edition")
st.markdown("1. Carica il video. 2. Disegna un rettangolo sul bilanciere. 3. Goditi il risultato.")

uploaded_file = st.file_uploader("Carica Video", type=["mp4", "mov"])

if uploaded_file is not None:
    # Salva il file temporaneamente
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    video_path = tfile.name

    # --- FASE 1: SELEZIONE (CANVAS) ---
    st.subheader("🛠️ Setup: Seleziona il Bilanciere")
    
    # Estraiamo il primo frame per farlo vedere all'utente
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()
    
    if ret:
        # Convertiamo colore da BGR (OpenCV) a RGB (Web)
        first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(first_frame_rgb)

        # Creiamo il Canvas per disegnare
        st.info("Disegna un rettangolo preciso attorno al disco del bilanciere.")
        
        # Calcoliamo una larghezza adatta per il web
        canvas_width = 700
        canvas_height = int(pil_image.height * (canvas_width / pil_image.width))
        
        canvas_result = st_canvas(
            fill_color="rgba(255, 165, 0, 0.3)",  # Colore riempimento semitrasparente
            stroke_width=2,
            stroke_color="#FF0000",
            background_image=pil_image,
            update_streamlit=True,
            height=canvas_height,
            width=canvas_width,
            drawing_mode="rect", # Modalità rettangolo
            key="canvas",
        )

        # Se l'utente ha disegnato qualcosa...
        if canvas_result.json_data is not None:
            objects = canvas_result.json_data["objects"]
            if len(objects) > 0:
                # Prendiamo l'ultimo rettangolo disegnato
                rect = objects[-1]
                
                # Le coordinate del canvas sono scalate, dobbiamo riportarle alle dimensioni reali del video
                scale_factor = pil_image.width / canvas_width
                
                real_x = int(rect["left"] * scale_factor)
                real_y = int(rect["top"] * scale_factor)
                real_w = int(rect["width"] * scale_factor)
                real_h = int(rect["height"] * scale_factor)
                
                bbox = (real_x, real_y, real_w, real_h)
                
                st.success(f"Target selezionato! Coordinate: {bbox}")
                
                # --- FASE 2: ELABORAZIONE ---
                if st.button("🚀 AVVIA ANALISI"):
                    output_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                    output_path = output_file.name
                    
                    with st.spinner('L\'intelligenza artificiale sta analizzando la tua alzata...'):
                        process_video(video_path, output_path, bbox)
                    
                    st.success("Fatto! Ecco il risultato:")
                    
                    # Mostra video risultante
                    # Nota: A volte i browser faticano con i video generati da OpenCV raw.
                    # Se non si vede, bisogna convertirlo (ma proviamo così per ora).
                    st.video(output_path)
                    
                    # Tasto download
                    with open(output_path, "rb") as file:
                        btn = st.download_button(
                            label="Scarica Video Analizzato",
                            data=file,
                            file_name="barbell_analysis.mp4",
                            mime="video/mp4"
                        )   