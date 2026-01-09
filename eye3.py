import cv2
import time
import os
import requests
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# ================= CONFIG =================
ESP32_IP = "192.168.15.200"   # 🔴 change this
ESP32_URL = f"http://{ESP32_IP}"
EYE_CLOSED_TIME = 0.5

# MediaPipe Tasks model config
FACE_LANDMARKER_MODEL_PATH = "face_landmarker.task"
FACE_LANDMARKER_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
)
# ==========================================


def get_face_landmarker():
    """Create (and lazily download) the MediaPipe FaceLandmarker."""
    if not os.path.exists(FACE_LANDMARKER_MODEL_PATH):
        try:
            print("Downloading face_landmarker model...")
            resp = requests.get(FACE_LANDMARKER_MODEL_URL, timeout=30)
            resp.raise_for_status()
            with open(FACE_LANDMARKER_MODEL_PATH, "wb") as f:
                f.write(resp.content)
            print("Model downloaded.")
        except Exception as e:
            raise RuntimeError(f"Failed to download face_landmarker model: {e}") from e

    base_options = python.BaseOptions(model_asset_path=FACE_LANDMARKER_MODEL_PATH)
    options = vision.FaceLandmarkerOptions(
        base_options=base_options,
        running_mode=vision.RunningMode.VIDEO,
        num_faces=1,
    )
    return vision.FaceLandmarker.create_from_options(options)


face_landmarker = get_face_landmarker()

cap = cv2.VideoCapture(0)

eye_closed_start = None
last_command = ""

def send_command(cmd):
    global last_command
    if cmd == last_command:
        return
    try:
        requests.get(f"{ESP32_URL}/{cmd}", timeout=0.7)
        print("SENT →", cmd)
        last_command = cmd
    except:
        print("ESP32 not reachable")

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    return (A + B) / (2.0 * C)

LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

frame_idx = 0

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Wrap frame in MediaPipe Image and run the FaceLandmarker (tasks API).
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
    result = face_landmarker.detect_for_video(mp_image, frame_idx)
    frame_idx += 1

    if result.face_landmarks:
        landmarks = result.face_landmarks[0]
        h, w, _ = frame.shape

        # Convert normalized landmarks to pixel coordinates.
        left_eye = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in LEFT_EYE]
        )
        right_eye = np.array(
            [[landmarks[i].x * w, landmarks[i].y * h] for i in RIGHT_EYE]
        )

        ear = (eye_aspect_ratio(left_eye) + eye_aspect_ratio(right_eye)) / 2

        if ear < 0.22:
            if eye_closed_start is None:
                eye_closed_start = time.time()
            elif time.time() - eye_closed_start > EYE_CLOSED_TIME:
                send_command("stop")
                cv2.putText(frame, "EYE CLOSED → STOP",
                            (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                            (0, 0, 255), 2)
        else:
            eye_closed_start = None
            send_command("forward")
            cv2.putText(frame, "EYE OPEN → FORWARD",
                        (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow("AI Eye Control", frame)
    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()