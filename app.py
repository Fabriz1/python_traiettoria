import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import tempfile
import numpy as np
import os
from collections import deque

# --- CONFIGURAZIONE PAGINA ---
st.set_page_config(page_title="Barbell Tracker Big Screen", layout="wide")

# Funzione per disegnare il grafico
def draw_graph(trajectory, width, height):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    cx, cy = int(width/2), int(height/2)
    cv2.line(img, (cx, 0), (cx, height), (30, 30, 30), 1)
    cv2.line(img, (0, cy), (width, cy), (30, 30, 30), 1)
    
    for i in range(0, width, 50):
        cv2.line(img, (i, 0), (i, height), (15, 15, 15), 1)
    for i in range(0, height, 50):
        cv2.line(img, (0, i), (width, i), (15, 15, 15), 1)
    
    if len(trajectory) > 1:
        cv2.polylines(img, [np.array(trajectory)], False, (0, 255, 0), 4, lineType=cv2.LINE_AA)
            
    if len(trajectory) > 0:
        cv2.circle(img, trajectory[-1], 8, (0, 0, 255), -1)
        
    return img

def process_video_zoom(input_path, output_path, init_bbox, perspective_points, zoom_factor):
    cap = cv2.VideoCapture(input_path)
    
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    matrix = None
    if perspective_points and len(perspective_points) == 4:
        pts1 = np.float32(perspective_points)
        pts2 = np.float32([[0,0], [400,0], [400,600], [0,600]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

    graph_w = w_orig 
    out_w = w_orig + graph_w
    out_h = max(h_orig, 600)
    
    graph_cx = int(graph_w / 2)
    graph_cy = int(out_h / 2)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    tracker = cv2.TrackerCSRT_create()
    ret, frame = cap.read()
    if not ret: return
    tracker.init(frame, init_bbox)

    traj_raw = []
    traj_zoomed = [] 
    recent_points = deque(maxlen=5) 
    
    start_x_raw, start_y_raw = 0, 0
    first_point_found = False

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = st.progress(0)
    frame_cnt = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if h_orig < out_h:
            frame = cv2.copyMakeBorder(frame, 0, out_h - h_orig, 0, 0, cv2.BORDER_CONSTANT)

        success, bbox = tracker.update(frame)
        graph_img = draw_graph([], graph_w, out_h)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            cx, cy = int(x + w/2), int(y + h/2)
            
            recent_points.append((cx, cy))
            avg_x = int(sum(p[0] for p in recent_points) / len(recent_points))
            avg_y = int(sum(p[1] for p in recent_points) / len(recent_points))
            
            traj_raw.append((avg_x, avg_y))

            current_real_x, current_real_y = avg_x, avg_y
            
            if matrix is not None:
                pt_array = np.array([[[avg_x, avg_y]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(pt_array, matrix)
                current_real_x = dst[0][0][0]
                current_real_y = dst[0][0][1]

            if not first_point_found:
                start_x_raw = current_real_x
                start_y_raw = current_real_y
                first_point_found = True
            
            delta_x = (current_real_x - start_x_raw)
            delta_y = (current_real_y - start_y_raw)
            
            zoomed_x = int(graph_cx + (delta_x * zoom_factor))
            zoomed_y = int(graph_cy + (delta_y * zoom_factor))
            
            traj_zoomed.append((zoomed_x, zoomed_y))

            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            if len(traj_raw) > 1:
                cv2.polylines(frame, [np.array(traj_raw)], False, (0, 0, 255), 2)

            graph_img = draw_graph(traj_zoomed, graph_w, out_h)
            
            cv2.putText(graph_img, f"ZOOM: {zoom_factor}x", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            cv2.putText(graph_img, "TRAIETTORIA PURA", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        combined = np.hstack((frame, graph_img))
        out.write(combined)
        
        frame_cnt += 1
        pbar.progress(min(frame_cnt / total_frames, 1.0))

    cap.release()
    out.release()
    pbar.empty()

# --- INTERFACCIA ---

st.title("🏋️ Barbell Tracker - Big Screen")

with st.sidebar:
    st.header("Impostazioni")
    st.write("Se la linea è troppo piccola, aumenta lo Zoom.")
    zoom_val = st.slider("Livello Zoom", min_value=1.0, max_value=6.0, value=3.0, step=0.5)
    st.info(f"Moltiplicatore attuale: {zoom_val}x")

uploaded_file = st.file_uploader("Carica Video", type=["mp4", "mov"])

if uploaded_file is not None:
    tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
    tfile.write(uploaded_file.read())
    video_path = tfile.name
    
    cap = cv2.VideoCapture(video_path)
    ret, first_frame = cap.read()
    cap.release()
    
    if ret:
        frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(frame_rgb)
        
        col1, col2 = st.columns(2)
        perspective_pts = []
        target_bbox = None
        
        with col1:
            st.info("1. Punti Prospettiva (Opzionale)")
            c_width = 400
            c_height = int(pil_img.height * (c_width / pil_img.width))
            
            canvas_p = st_canvas(
                fill_color="rgba(0, 255, 0, 0.3)",
                stroke_width=3,
                stroke_color="#00FF00",
                background_image=pil_img,
                update_streamlit=True,
                height=c_height,
                width=c_width,
                drawing_mode="point",
                key="canvas_persp",
            )
            if canvas_p.json_data:
                pts = [o for o in canvas_p.json_data["objects"] if o["type"] == "circle"]
                if len(pts) == 4:
                    scale = pil_img.width / c_width
                    perspective_pts = [[p["left"]*scale, p["top"]*scale] for p in pts]
                    st.success("✅ Punti OK")

        with col2:
            st.info("2. Seleziona Bilanciere")
            canvas_t = st_canvas(
                fill_color="rgba(255, 0, 0, 0.3)",
                stroke_width=2,
                stroke_color="#FF0000",
                background_image=pil_img,
                update_streamlit=True,
                height=c_height,
                width=c_width,
                drawing_mode="rect",
                key="canvas_track",
            )
            if canvas_t.json_data:
                objs = canvas_t.json_data["objects"]
                if objs:
                    r = objs[-1]
                    s = pil_img.width / c_width
                    target_bbox = (int(r["left"]*s), int(r["top"]*s), int(r["width"]*s), int(r["height"]*s))
                    st.success("✅ Target OK")

        st.divider()
        if target_bbox:
            if st.button("🚀 AVVIA ANALISI ZOOM", type="primary"):
                out_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                with st.spinner(f"Elaborazione con Zoom {zoom_val}x..."):
                    process_video_zoom(video_path, out_file, target_bbox, perspective_pts, zoom_val)
                
                # --- QUI C'ERA L'ERRORE: HO RIMESSO IL CODICE DI DOWNLOAD ---
                st.success("Analisi Completata! Scarica il video qui sotto.")
                
                # Leggiamo il file generato
                with open(out_file, "rb") as f:
                    # Tasto Download
                    st.download_button(
                        label="📥 SCARICA VIDEO ANALIZZATO",
                        data=f,
                        file_name="analisi_zoom.mp4",
                        mime="video/mp4"
                    )
                
                # Provo comunque a mostrarlo (ma se resta nero, usa il download)
                st.video(out_file)