import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import cv2
import tempfile
import numpy as np
import pandas as pd
from collections import deque
import os

# --- CONFIGURAZIONE ---
st.set_page_config(page_title="Barbell Tracker Wizard", layout="wide")

# --- GESTIONE STATO ---
if 'step' not in st.session_state: st.session_state.step = 0
if 'processed' not in st.session_state: st.session_state.processed = False
if 'video_file_path' not in st.session_state: st.session_state.video_file_path = None
if 'persp_pts' not in st.session_state: st.session_state.persp_pts = []
if 'plate_rect' not in st.session_state: st.session_state.plate_rect = None
if 'track_rect' not in st.session_state: st.session_state.track_rect = None
if 'df_res' not in st.session_state: st.session_state.df_res = None
if 'vid_res' not in st.session_state: st.session_state.vid_res = None

# --- FUNZIONI GRAFICHE ---
def draw_side_graph(trajectory, velocities, width, height, current_phase, start_point):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    cx, cy = int(width/2), int(height/2)
    
    cv2.line(img, (cx, 0), (cx, height), (40, 40, 40), 1)
    cv2.line(img, (0, cy), (width, cy), (40, 40, 40), 1)
    
    if len(trajectory) > 1 and start_point:
        start_x, start_y = start_point
        zoom = 3.0
        pts = []
        for (px, py) in trajectory:
            dx = int((px - start_x) * zoom)
            dy = int((py - start_y) * zoom)
            pts.append((cx + dx, cy + dy))
        
        cv2.polylines(img, [np.array(pts)], False, (0, 200, 0), 2, lineType=cv2.LINE_AA)
        if pts: cv2.circle(img, pts[-1], 5, (0, 0, 255), -1)

    if velocities:
        vel = velocities[-1]
        abs_v = abs(vel)
        bar_h = int(abs_v * 250) 
        if bar_h > 250: bar_h = 250
        col = (0, 255, 0) if current_phase == "Concentrica" else (0, 0, 255)
        if current_phase == "Statico": col = (100, 100, 100)
        
        cv2.rectangle(img, (20, height-20), (50, height-20-bar_h), col, -1)
        cv2.putText(img, f"{abs_v:.2f} m/s", (60, height-20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

    return img

def run_analysis(input_path, output_path, plate_bbox, track_bbox, perspective_points, exercise_type, plate_diam_cm):
    cap = cv2.VideoCapture(input_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    if fps == 0: fps = 30
    w_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 1. Prospettiva
    matrix = None
    if perspective_points and len(perspective_points) == 4:
        pts1 = np.float32(perspective_points)
        pts2 = np.float32([[0,0], [400,0], [400,600], [0,600]])
        matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # 2. Calibrazione
    raw_h = plate_bbox[3]
    if matrix is not None:
        top_pt = np.array([[[plate_bbox[0], plate_bbox[1]]]], dtype=np.float32)
        bot_pt = np.array([[[plate_bbox[0], plate_bbox[1] + raw_h]]], dtype=np.float32)
        t_top = cv2.perspectiveTransform(top_pt, matrix)
        t_bot = cv2.perspectiveTransform(bot_pt, matrix)
        corrected_h = abs(t_bot[0][0][1] - t_top[0][0][1])
    else:
        corrected_h = raw_h

    if corrected_h < 1: corrected_h = 1
    meters_per_px = (plate_diam_cm / 100.0) / corrected_h

    # Output
    graph_w = 400
    out_w = w_orig + graph_w
    out_h = max(h_orig, 600)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(output_path, fourcc, fps, (out_w, out_h))

    tracker = cv2.TrackerCSRT_create()
    ret, frame = cap.read()
    tracker.init(frame, track_bbox)

    vel_buf = deque(maxlen=5)
    pos_x_buf = deque(maxlen=5)
    pos_y_buf = deque(maxlen=5)
    
    traj_corrected = [] 
    vel_history = []
    
    start_point_corrected = None
    
    # LOGICA ROBUSTA (ISTERESI / DISTANZA)
    current_phase = "Statico"
    rep_counter = 0
    phase_start_frame = 0
    
    # Picchi Raggiunti (in Metri)
    # Importante: In coordinate CV, Y cresce scendendo.
    # Quindi MIN_Y = Punto più Alto (Top), MAX_Y = Punto più Basso (Bottom)
    peak_top_y_m = float('inf') 
    peak_bottom_y_m = float('-inf')
    
    # Soglia Movimento: 6 cm
    ROM_THRESH_M = 0.06 
    
    temp_stats = {"max": 0, "sum": 0, "cnt": 0}
    rep_results = []
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    pbar = st.progress(0)
    frame_cnt = 0

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        if h_orig < out_h:
            frame = cv2.copyMakeBorder(frame, 0, out_h - h_orig, 0, 0, cv2.BORDER_CONSTANT)

        success, bbox = tracker.update(frame)
        
        if success:
            x, y, w, h = [int(v) for v in bbox]
            cx, cy = int(x + w/2), int(y + h/2)
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 1)
            
            # Coordinate Raddrizzate
            curr_x, curr_y = cx, cy
            if matrix is not None:
                pt = np.array([[[cx, cy]]], dtype=np.float32)
                dst = cv2.perspectiveTransform(pt, matrix)
                curr_x = dst[0][0][0]
                curr_y = dst[0][0][1]
            
            pos_x_buf.append(curr_x)
            pos_y_buf.append(curr_y)
            smooth_x = int(sum(pos_x_buf)/len(pos_x_buf))
            smooth_y = int(sum(pos_y_buf)/len(pos_y_buf))
            
            if start_point_corrected is None:
                start_point_corrected = (smooth_x, smooth_y)
                curr_y_m = smooth_y * meters_per_px
                peak_top_y_m = curr_y_m
                peak_bottom_y_m = curr_y_m
            
            traj_corrected.append((smooth_x, smooth_y))

            # Y in Metri
            curr_y_m = smooth_y * meters_per_px
            
            # Calcolo Velocità Istantanea (basata su delta posizionale)
            # Se siamo al primo frame, velocità 0
            instant_vel = 0
            if len(traj_corrected) > 1:
                # Delta rispetto al frame precedente
                prev_y_m = traj_corrected[-2][1] * meters_per_px
                # Invertiamo segno: se Y diminuisce (sale), velocità positiva
                delta_m = prev_y_m - curr_y_m
                instant_vel = delta_m * fps
            
            vel_buf.append(instant_vel)
            smooth_vel = sum(vel_buf) / len(vel_buf)
            vel_history.append(smooth_vel)
            abs_vel = abs(smooth_vel)

            # --- LOGICA ISTERESI ---
            
            # Aggiorna estremi
            if curr_y_m < peak_top_y_m: peak_top_y_m = curr_y_m       # Nuovo massimo in alto
            if curr_y_m > peak_bottom_y_m: peak_bottom_y_m = curr_y_m # Nuovo massimo in basso
            
            new_phase = current_phase
            
            # Calcolo distanza percorsa dall'ultimo estremo
            dist_from_top = curr_y_m - peak_top_y_m      # Quanto siamo scesi (positivo)
            dist_from_bottom = peak_bottom_y_m - curr_y_m # Quanto siamo saliti (positivo)
            
            # Logica Transizioni
            if current_phase == "Statico":
                if dist_from_top > ROM_THRESH_M:
                    new_phase = "Eccentrica"
                    peak_bottom_y_m = curr_y_m # Reset fondo
                elif dist_from_bottom > ROM_THRESH_M:
                    new_phase = "Concentrica"
                    peak_top_y_m = curr_y_m # Reset cima
            
            elif current_phase == "Eccentrica": # Stiamo scendendo
                if dist_from_bottom > ROM_THRESH_M: # Se siamo risaliti di X cm
                    new_phase = "Concentrica"
                    peak_top_y_m = curr_y_m # Il nuovo top parte da qui
            
            elif current_phase == "Concentrica": # Stiamo salendo
                if dist_from_top > ROM_THRESH_M: # Se siamo riscesi di X cm
                    new_phase = "Eccentrica"
                    peak_bottom_y_m = curr_y_m # Il nuovo bottom parte da qui

            # CAMBIO FASE
            if new_phase != current_phase:
                # Salva dati fase precedente
                if temp_stats["cnt"] > 5:
                    avg_v = temp_stats["sum"] / temp_stats["cnt"]
                    peak_v = temp_stats["max"]
                    duration = (frame_cnt - phase_start_frame) / fps
                    
                    save = True
                    if exercise_type == "Stacco (Deadlift)" and current_phase == "Eccentrica": save = False
                    
                    if save and current_phase != "Statico":
                        r_num = rep_counter
                        if current_phase == "Concentrica":
                            rep_counter += 1
                            r_num = rep_counter
                        elif current_phase == "Eccentrica":
                            r_num = rep_counter + 1
                        
                        rep_results.append({
                            "Rep": r_num, "Fase": current_phase,
                            "V_Media": round(abs(avg_v), 2), 
                            "V_Picco": round(abs(peak_v), 2),
                            "Durata (s)": round(duration, 2)
                        })
                
                # Reset
                current_phase = new_phase
                phase_start_frame = frame_cnt
                temp_stats = {"max": 0, "sum": 0, "cnt": 0}
                
                # Reset estremi opposti per tracciare la prossima inversione pulita
                if new_phase == "Eccentrica": peak_bottom_y_m = curr_y_m
                if new_phase == "Concentrica": peak_top_y_m = curr_y_m

            # Accumulo stats
            if current_phase != "Statico":
                temp_stats["cnt"] += 1
                temp_stats["sum"] += abs_vel
                if abs_vel > temp_stats["max"]: temp_stats["max"] = abs_vel

            cv2.putText(frame, f"Vel: {abs_vel:.2f} m/s", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Fase: {current_phase}", (20, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)
            cv2.putText(frame, f"Reps: {rep_counter}", (20, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

        graph_img = draw_side_graph(traj_corrected, vel_history, graph_w, out_h, current_phase, start_point_corrected)
        final = np.hstack((frame, graph_img))
        out.write(final)
        frame_cnt += 1
        pbar.progress(min(frame_cnt / total_frames, 1.0))

    cap.release()
    out.release()
    pbar.empty()
    return rep_results

# --- INTERFACCIA WIZARD ---

st.title("🏋️ Barbell Tracker Wizard")

# Selettori sempre visibili
with st.sidebar:
    st.header("⚙️ Impostazioni")
    exercise = st.selectbox("Esercizio", ["Panca Piana", "Squat", "Stacco (Deadlift)"])
    diam = st.number_input("Diametro Disco (cm)", value=45.0, step=0.5)
    if st.button("🔄 Reset"):
        st.session_state.clear()
        st.rerun()

# 0. CARICAMENTO
if st.session_state.video_file_path is None:
    uploaded = st.file_uploader("Carica Video", type=["mp4", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') 
        tfile.write(uploaded.read())
        st.session_state.video_file_path = tfile.name
        st.session_state.step = 1
        st.rerun()

else:
    # Carica Frame
    cap = cv2.VideoCapture(st.session_state.video_file_path)
    ret, frame = cap.read()
    cap.release()
    pil_img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    
    # --- STEP 1: PROSPETTIVA ---
    if st.session_state.step == 1:
        st.subheader("Step 1/3: Prospettiva")
        st.info("Clicca 4 punti (Rettangolo Verticale). Se è già dritto, premi AVANTI.")
        
        cw = 700
        ch = int(pil_img.height * (cw / pil_img.width))
        
        cv1 = st_canvas(
            fill_color="rgba(0,255,0,0.3)", stroke_width=1, point_display_radius=4,
            stroke_color="#00FF00", background_image=pil_img,
            height=ch, width=cw, drawing_mode="point", key="cv1"
        )
        
        col1, col2 = st.columns(2)
        if col1.button("Avanti ➡️"):
            # Salva punti se ci sono
            if cv1.json_data:
                pts = [o for o in cv1.json_data["objects"] if o["type"]=="circle"]
                if len(pts)==4:
                    scale = pil_img.width / cw
                    st.session_state.persp_pts = [[p["left"]*scale, p["top"]*scale] for p in pts]
            st.session_state.step = 2
            st.rerun()
            
    # --- STEP 2: CALIBRAZIONE ---
    elif st.session_state.step == 2:
        st.subheader(f"Step 2/3: Calibrazione (Disco {diam}cm)")
        st.info("Disegna un rettangolo ESATTO attorno al disco.")
        
        cw = 700
        ch = int(pil_img.height * (cw / pil_img.width))
        
        cv2_c = st_canvas(
            fill_color="rgba(0,0,255,0.2)", stroke_width=1, stroke_color="#0000FF",
            background_image=pil_img, height=ch, width=cw, drawing_mode="rect", key="cv2"
        )
        
        col1, col2 = st.columns(2)
        if col1.button("⬅️ Indietro"):
            st.session_state.step = 1
            st.rerun()
        if col2.button("Avanti ➡️"):
            if cv2_c.json_data:
                objs = cv2_c.json_data["objects"]
                if objs:
                    r = objs[-1]
                    s = pil_img.width / cw
                    st.session_state.plate_rect = (int(r["left"]*s), int(r["top"]*s), int(r["width"]*s), int(r["height"]*s))
                    st.session_state.step = 3
                    st.rerun()
            else:
                st.error("Devi selezionare il disco!")

    # --- STEP 3: TRACKING ---
    elif st.session_state.step == 3:
        st.subheader("Step 3/3: Tracking")
        st.info("Disegna un rettangolo sulla PUNTA del bilanciere.")
        
        cw = 700
        ch = int(pil_img.height * (cw / pil_img.width))
        
        cv3 = st_canvas(
            fill_color="rgba(255,0,0,0.2)", stroke_width=1, stroke_color="#FF0000",
            background_image=pil_img, height=ch, width=cw, drawing_mode="rect", key="cv3"
        )
        
        col1, col2 = st.columns(2)
        if col1.button("⬅️ Indietro"):
            st.session_state.step = 2
            st.rerun()
        if col2.button("🚀 AVVIA ANALISI", type="primary"):
            if cv3.json_data:
                objs = cv3.json_data["objects"]
                if objs:
                    r = objs[-1]
                    s = pil_img.width / cw
                    st.session_state.track_rect = (int(r["left"]*s), int(r["top"]*s), int(r["width"]*s), int(r["height"]*s))
                    
                    # ESEGUI ANALISI
                    outfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
                    with st.spinner("Elaborazione in corso..."):
                        data = run_analysis(
                            st.session_state.video_file_path, outfile,
                            st.session_state.plate_rect, st.session_state.track_rect,
                            st.session_state.persp_pts, exercise, diam
                        )
                    
                    st.session_state.df_res = pd.DataFrame(data) if data else None
                    st.session_state.vid_res = outfile
                    st.session_state.processed = True
                    st.rerun()
            else:
                st.error("Seleziona il tracker!")

    # --- RISULTATI ---
    if st.session_state.processed:
        st.divider()
        st.subheader("📊 Report Finale")
        
        if st.session_state.df_res is not None:
            df = st.session_state.df_res
            def color_phase(val):
                return 'color: green' if val == 'Concentrica' else 'color: red'
            st.dataframe(df.style.applymap(color_phase, subset=['Fase']), use_container_width=True)
        else:
            st.warning("Nessuna rep valida (movimento < 6cm).")
        
        st.video(st.session_state.vid_res)
        with open(st.session_state.vid_res, "rb") as f:
            st.download_button("📥 Scarica Video", f, "analisi_wizard.mp4")
        
        if st.button("🔄 Nuova Analisi"):
            st.session_state.clear()
            st.rerun()