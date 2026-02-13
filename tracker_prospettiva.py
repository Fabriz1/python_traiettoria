import cv2
import numpy as np
from collections import deque

# --- CONFIGURAZIONE ---
# Quanti frame usare per la media? 
# 1 = Nessuna modifica (tremolante)
# 5 = Normale (consigliato)
# 10 = Molto morbido (ma con leggero ritardo)
SMOOTHING_FACTOR = 5 

# Variabili globali
punti_click = []

def click_event(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(punti_click) < 4:
            punti_click.append((x, y))
            cv2.circle(param, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(param, str(len(punti_click)), (x+10, y), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow("Calibrazione", param)

def main():
    video_path = 'video.mp4' 
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Errore video")
        return

    ret, frame = cap.read()
    if not ret: return

    # Ridimensiona
    scale_percent = 50 
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)
    
    frame_calibrazione = frame.copy()

    # --- CALIBRAZIONE ---
    print("\n--- ISTRUZIONI ---")
    print("Clicca 4 punti per il rettangolo verticale di riferimento.")
    print("Ordine: 1.Alto-Sx, 2.Alto-Dx, 3.Basso-Dx, 4.Basso-Sx")
    
    cv2.imshow("Calibrazione", frame_calibrazione)
    cv2.setMouseCallback("Calibrazione", click_event, frame_calibrazione)

    while True:
        k = cv2.waitKey(1)
        if len(punti_click) == 4:
            print("Punti presi. Premi un tasto...")
            cv2.waitKey(0) 
            break
        if k == 27: return 

    cv2.destroyWindow("Calibrazione")

    # Matrice Prospettiva
    pts1 = np.float32(punti_click)
    pts2 = np.float32([[0,0], [300,0], [300,500], [0,500]])
    matrix = cv2.getPerspectiveTransform(pts1, pts2)

    # --- TRACKING ---
    print("Seleziona il bilanciere e premi INVIO")
    bbox = cv2.selectROI("Tracker", frame, False)
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)
    cv2.destroyWindow("Tracker")

    # Liste per i punti
    trajectory_points = []
    
    # Buffer per la media mobile (Smoothing)
    recent_points_buffer = deque(maxlen=SMOOTHING_FACTOR)
    
    offset_x = 0
    offset_y = 0
    first_point_found = False

    while True:
        ret, frame = cap.read()
        if not ret: break
        
        frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
        
        success, bbox = tracker.update(frame)

        if success:
            x, y, w, h = [int(v) for v in bbox]
            center_x = int(x + w/2)
            center_y = int(y + h/2)

            # 1. Calcolo coordinate raddrizzate (Raw - Grezze)
            original_point = np.array([[[center_x, center_y]]], dtype=np.float32)
            transformed_point = cv2.perspectiveTransform(original_point, matrix)
            
            tx_raw = transformed_point[0][0][0]
            ty_raw = transformed_point[0][0][1]

            # 2. Aggiungi al buffer per fare la media
            recent_points_buffer.append((tx_raw, ty_raw))

            # 3. Calcola la media degli ultimi N punti (Smoothing)
            avg_x = sum(p[0] for p in recent_points_buffer) / len(recent_points_buffer)
            avg_y = sum(p[1] for p in recent_points_buffer) / len(recent_points_buffer)

            # 4. Centratura (usando il punto medio)
            if not first_point_found:
                offset_x = 200 - avg_x
                offset_y = 300 - avg_y
                first_point_found = True
            
            final_x = int(avg_x + offset_x)
            final_y = int(avg_y + offset_y)
            
            trajectory_points.append((final_x, final_y))

            # Disegna box su video originale
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            
            # --- DISEGNO SU SFONDO NERO ---
            view_2d = np.zeros((600, 400, 3), dtype=np.uint8)
            
            # Linee griglia
            cv2.line(view_2d, (0, 300), (400, 300), (30, 30, 30), 1) 
            cv2.line(view_2d, (200, 0), (200, 600), (30, 30, 30), 1)

            # Disegna traiettoria morbida
            if len(trajectory_points) > 1:
                # Disegna la linea punto per punto
                for i in range(1, len(trajectory_points)):
                    # Sfumatura colore (dal verde scuro al verde acceso)
                    # Opzionale: rende la scia più bella
                    color_intensity = int(255 * (i / len(trajectory_points)))
                    color = (0, color_intensity, 0) # BGR
                    
                    cv2.line(view_2d, trajectory_points[i-1], trajectory_points[i], color, 2)
            
            cv2.circle(view_2d, (final_x, final_y), 5, (0, 0, 255), -1)
            cv2.imshow("Traiettoria (Smooth)", view_2d)

        cv2.imshow("Video Originale", frame)

        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()