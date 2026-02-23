import cv2
import mediapipe as mp
import time
import screen_brightness_control as sbc
import winsound
from scipy.spatial import distance as dist

class Blinky:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(refine_landmarks=True)
        
        # --- CONFIGURATION ---
        self.EAR_THRESHOLD = 0.16
        self.FROWN_THRESHOLD = 0.052 
        self.REQUIRED_BPM = 15
        self.NORMAL_BR = 50
        self.DIMMED_BR = 20
        
        # --- LOGIC STATE ---
        self.is_closed = False
        self.dimmed = False
        self.last_blink_start = 0
        self.current_blink_duration = 0
        self.frown_start_time = None
        self.beep_cooldown = 0
        self.minute_start_time = time.time()
        
        # --- COUNTERS ---
        self.total_session_blinks = 0 
        self.deep_blink_count = 0
        self.blink_timestamps = []
        self.frown_history = []

    def get_ear(self, mesh_points):
        l_v = dist.euclidean((mesh_points[159].x, mesh_points[159].y), (mesh_points[145].x, mesh_points[145].y))
        l_h = dist.euclidean((mesh_points[33].x, mesh_points[33].y), (mesh_points[133].x, mesh_points[133].y))
        r_v = dist.euclidean((mesh_points[386].x, mesh_points[386].y), (mesh_points[374].x, mesh_points[374].y))
        r_h = dist.euclidean((mesh_points[362].x, mesh_points[362].y), (mesh_points[263].x, mesh_points[263].y))
        return ((l_v/l_h) + (r_v/r_h)) / 2

    def get_frown_score(self, mesh_points):
        dist_l = dist.euclidean((mesh_points[107].x, mesh_points[107].y), (mesh_points[6].x, mesh_points[6].y))
        dist_r = dist.euclidean((mesh_points[336].x, mesh_points[336].y), (mesh_points[6].x, mesh_points[6].y))
        return (dist_l + dist_r) / 2

    def run(self):
        cap = cv2.VideoCapture(0)
        sbc.set_brightness(self.NORMAL_BR)
        running = True 

        try:
            while running:
                success, frame = cap.read()
                if not success: break
                
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = self.face_mesh.process(rgb_frame)
                now = time.time()
                
                # Check for 'q' immediately to stop processing
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    running = False
                    break

                if now - self.minute_start_time >= 60:
                    self.minute_start_time = now

                if results.multi_face_landmarks:
                    mesh = results.multi_face_landmarks[0].landmark
                    ear = self.get_ear(mesh)
                    frown_score = self.get_frown_score(mesh)

                    # --- BLINK ENGINE ---
                    if ear < self.EAR_THRESHOLD:
                        if not self.is_closed:
                            self.is_closed = True
                            self.last_blink_start = now
                        self.current_blink_duration = now - self.last_blink_start
                    else:
                        if self.is_closed:
                            duration = now - self.last_blink_start
                            self.blink_timestamps.append(now)
                            self.total_session_blinks += 1
                            if self.dimmed and duration >= 0.5:
                                self.deep_blink_count += 1
                                winsound.Beep(1200, 100)
                            self.is_closed = False
                            self.current_blink_duration = 0

                    # --- FROWN ENGINE ---
                    is_frowning = frown_score < self.FROWN_THRESHOLD
                    if is_frowning:
                        self.frown_history.append(now)
                        if self.frown_start_time is None: self.frown_start_time = now
                        if (now - self.frown_start_time > 5) and not self.dimmed:
                            sbc.set_brightness(self.DIMMED_BR)
                            self.dimmed = True
                    else:
                        self.frown_start_time = None

                    # --- HEALTH CHECKS ---
                    self.blink_timestamps = [t for t in self.blink_timestamps if now - t < 60]
                    self.frown_history = [t for t in self.frown_history if now - t < 60]
                    
                    if not self.dimmed and (now - self.minute_start_time > 30):
                        if len(self.blink_timestamps) < self.REQUIRED_BPM:
                            sbc.set_brightness(self.DIMMED_BR)
                            self.dimmed = True

                    # Beep logic: only fires if 'running' is still true
                    if running and len(self.frown_history) > 450 and (now - self.beep_cooldown > 10):
                        winsound.Beep(800, 300)
                        self.beep_cooldown = now

                    # --- RESTORE ENGINE ---
                    if self.dimmed and self.deep_blink_count >= 5:
                        winsound.Beep(1500, 500)
                        sbc.set_brightness(self.NORMAL_BR)
                        self.dimmed = False
                        self.blink_timestamps, self.frown_history = [], []
                        self.deep_blink_count = 0
                        self.frown_start_time = None
                        self.minute_start_time = now

                    # === DASHBOARD DISPLAY ===
                    h, w, _ = frame.shape
                    
                    # DRAW DETECTING DOTS (New/Restored)
                    # Eye dots (Green)
                    for i in [159, 145, 33, 133, 386, 374, 362, 263]:
                        p = (int(mesh[i].x * w), int(mesh[i].y * h))
                        cv2.circle(frame, p, 1, (0, 255, 0), -1)
                    # Brow scowl dots (Red if stressed, Green if relaxed)
                    dot_color = (0, 0, 255) if is_frowning else (0, 255, 0)
                    for i in [107, 336, 6]:
                        p = (int(mesh[i].x * w), int(mesh[i].y * h))
                        cv2.circle(frame, p, 3, dot_color, -1)

                    cv2.rectangle(frame, (0, 0), (w, 140), (25, 25, 25), -1) 
                    
                    # Column 1: Blinks
                    cv2.putText(frame, f"Session: {self.total_session_blinks}", (20, 35), 1, 1.2, (0, 255, 0), 2)
                    cv2.putText(frame, f"BPM: {len(self.blink_timestamps)}/15", (20, 70), 1, 1.2, (255, 255, 255), 2)
                    if self.is_closed:
                        cv2.putText(frame, f"HOLD: {round(self.current_blink_duration, 1)}s", (20, 110), 1, 1.2, (255, 255, 0), 2)

                    # Column 2: Frown Dashboard
                    col2_x = w//2 - 100 
                    f_color = (0, 0, 255) if is_frowning else (0, 255, 0)
                    cv2.putText(frame, f"Frown: {round(frown_score, 3)}", (col2_x, 35), 1, 1.2, f_color, 2)
                    cv2.putText(frame, f"Scowl: {len(self.frown_history)//30}s/10s", (col2_x, 70), 1, 1.2, (255, 255, 0), 2)
                    cv2.putText(frame, "STRESSED" if is_frowning else "RELAXED", (col2_x, 105), 1, 1.2, f_color, 2)

                    # Column 3: Timer
                    cv2.putText(frame, f"Unlocks: {self.deep_blink_count}/5", (w-210, 35), 1, 1.4, (0, 255, 255), 2)
                    cv2.putText(frame, f"Reset: {int(60 - (now - self.minute_start_time))}s", (w-210, 80), 1, 1.8, (255, 255, 255), 3)

                    #if is_frowning:
                        #cv2.putText(frame, "RELAX FACE!", (w//2-110, h//2), 1, 2.5, (0, 0, 255), 4)
                    if self.dimmed:
                        cv2.rectangle(frame, (w//2-210, h-70), (w//2+210, h-20), (0, 0, 0), -1)
                        cv2.putText(frame, "LOCKED: DO 5 DEEP BLINKS", (w//2-180, h-35), 1, 1.2, (0, 255, 255), 2)

                    cv2.imshow("Blinky v3.2", frame)

        finally:
            # This ensures everything stops properly and NO beeps can trigger
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    app = Blinky()
    app.run()