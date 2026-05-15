import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import joblib
import numpy as np
import pyttsx3

MODEL_PATH = "isl_model.pkl"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

def create_landmarker(model_path="hand_landmarker.task"):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=1,
        running_mode=VisionRunningMode.IMAGE,
    )
    return HandLandmarker.create_from_options(options)

def landmarks_to_vector(landmarks):
    vec = []
    for lm in landmarks:
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec).reshape(1, -1)

def main():
    clf = joblib.load(MODEL_PATH)

    engine = pyttsx3.init()
    engine.setProperty("rate", 170)

    landmarker = create_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    last_label = None
    same_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame")
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        result = landmarker.detect(mp_image)

        predicted_label = ""

        if result.hand_landmarks:
            h, w, _ = frame.shape
            hand = result.hand_landmarks[0]

            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            vec = landmarks_to_vector(hand)
            predicted_label = clf.predict(vec)[0]

            if predicted_label == last_label:
                same_count += 1
            else:
                same_count = 1
                last_label = predicted_label

            if same_count == 10:
                engine.say(predicted_label)
                engine.runAndWait()

        cv2.putText(
            frame,
            f"Prediction: {predicted_label}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )
        cv2.putText(
            frame,
            "Press 'q' to quit",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        cv2.imshow("ISL Realtime Prediction", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()