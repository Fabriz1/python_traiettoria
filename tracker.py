import cv2
import numpy as np

def main():
    # 1. Carica il video (sostituisci con il nome del tuo file)
    video_path = 'video.mp4' 
    cap = cv2.VideoCapture(video_path)

    # Verifica se il video è stato aperto correttamente
    if not cap.isOpened():
        print("Errore: Impossibile aprire il video.")
        return

    # 2. Leggi il primo frame
    ret, frame = cap.read()
    if not ret:
        print("Errore: Impossibile leggere il frame.")
        return

    # Ridimensioniamo il video se è troppo grande (opzionale, per velocità)
    scale_percent = 50 # percent of original size
    width = int(frame.shape[1] * scale_percent / 100)
    height = int(frame.shape[0] * scale_percent / 100)
    dim = (width, height)
    frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

    # 3. Seleziona l'oggetto da tracciare (ROI - Region of Interest)
    # Si aprirà una finestra: seleziona il disco del bilanciere col mouse e premi INVIO
    print("Seleziona il disco del bilanciere e premi INVIO o SPAZIO")
    bbox = cv2.selectROI("Tracking", frame, False)
    
    # 4. Inizializza il Tracker
    # Usiamo CSRT perché è molto preciso anche se lento. 
    # Per velocità si potrebbe usare KCF, ma CSRT è meglio per le traiettorie.
    tracker = cv2.TrackerCSRT_create()
    tracker.init(frame, bbox)

    # Lista per memorizzare i punti della traiettoria
    trajectory_points = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break # Fine del video

        # Ridimensiona il frame corrente (stessa scala di prima)
        frame = cv2.resize(frame, dim, interpolation = cv2.INTER_AREA)

        # 5. Aggiorna il tracker
        success, bbox = tracker.update(frame)

        if success:
            # Ottieni le coordinate del box tracciato
            x, y, w, h = [int(v) for v in bbox]
            
            # Calcola il centro del box (centro del disco)
            center_x = int(x + w / 2)
            center_y = int(y + h / 2)
            
            # Aggiungi il punto alla lista della traiettoria
            trajectory_points.append((center_x, center_y))

            # Disegna il rettangolo attorno al disco (opzionale)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            
            # Disegna il centro attuale
            cv2.circle(frame, (center_x, center_y), 5, (0, 0, 255), -1)

        # 6. Disegna la linea della traiettoria
        if len(trajectory_points) > 1:
            # Disegna una linea che collega tutti i punti storici
            for i in range(1, len(trajectory_points)):
                # Spessore 2, Colore Rosso (B, G, R) -> (0, 0, 255)
                cv2.line(frame, trajectory_points[i - 1], trajectory_points[i], (0, 0, 255), 2)

        # Mostra il risultato
        cv2.imshow("Barbell Tracker", frame)

        # Premi 'q' per uscire prima della fine
        if cv2.waitKey(30) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()