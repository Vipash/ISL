import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import json
import pyttsx3
import time

MODEL_JSON = "isl_simple_model.json"
HAND_MODEL_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


def create_landmarker(model_path=HAND_MODEL_PATH):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=1,
        running_mode=VisionRunningMode.IMAGE,
    )
    return HandLandmarker.create_from_options(options)


def load_means():
    with open(MODEL_JSON, "r") as f:
        means = json.load(f)
    return {label: np.array(vec, dtype=np.float32) for label, vec in means.items()}


def landmarks_to_vector(landmarks):
    vec = []
    for lm in landmarks:
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec, dtype=np.float32)


def predict_label(vec, means):
    best_label = None
    best_dist = None
    for label, mean_vec in means.items():
        if vec.shape != mean_vec.shape:
            continue
        dist = np.linalg.norm(vec - mean_vec)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label


def speak_text(text):
    if not text:
        return
    engine = pyttsx3.init(driverName="sapi5")
    engine.setProperty("rate", 170)
    engine.setProperty("volume", 1.0)
    voices = engine.getProperty("voices")
    if len(voices) > 0:
        engine.setProperty("voice", voices[0].id)
    engine.say(text)
    engine.runAndWait()
    engine.stop()


def collapse_consecutive(words):
    out = []
    prev = None
    for w in words:
        if w != prev:
            out.append(w)
        prev = w
    return out


def main():
    means = load_means()
    any_vec = next(iter(means.values()))
    print("Loaded labels:", list(means.keys()))
    print("Vector dim from JSON:", any_vec.shape)

    landmarker = create_landmarker()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    last_spoken_label = None
    last_spoken_time = 0.0
    current_label = ""

    SPEAK_INTERVAL = 2.0
    auto_speak = False  # still available if you ever want it

    # Sentence recording state
    sentence_mode = False        # controlled by key '2'
    sentence_words = []          # words explicitly added with key '3'

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

        if result.hand_landmarks:
            h, w, _ = frame.shape
            hand = result.hand_landmarks[0]

            # draw landmarks
            for lm in hand:
                cx, cy = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

            vec = landmarks_to_vector(hand)

            if vec.shape != any_vec.shape:
                print("Shape mismatch:", vec.shape, "vs", any_vec.shape)

            label = predict_label(vec, means)
            current_label = label if label is not None else ""

            now = time.time()

            # Optional auto-speak (currently disabled)
            if auto_speak and label is not None:
                should_speak = False
                if label != last_spoken_label:
                    should_speak = True
                elif now - last_spoken_time >= SPEAK_INTERVAL:
                    should_speak = True

                if should_speak:
                    print("Auto speak:", label)
                    speak_text(label)
                    last_spoken_label = label
                    last_spoken_time = now
        else:
            current_label = ""

        # -------- UI overlay --------
        overlay = frame.copy()
        bar_height = 110
        cv2.rectangle(overlay, (0, 0), (frame.shape[1], bar_height), (0, 0, 0), -1)
        alpha = 0.5
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)

        status_color = (0, 255, 0) if current_label else (0, 0, 255)

        cv2.putText(
            frame,
            f"Prediction: {current_label if current_label else 'None'}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
            cv2.LINE_AA
        )

        cv2.putText(
            frame,
            f"Sentence mode: {'ON' if sentence_mode else 'OFF'}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
            cv2.LINE_AA
        )

        if sentence_words:
            preview = " ".join(sentence_words[-5:])
        else:
            preview = "(none)"

        cv2.putText(
            frame,
            f"Sentence: {preview}",
            (10, 75),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            1,
            cv2.LINE_AA
        )

        h_frame = frame.shape[0]
        cv2.putText(
            frame,
            "Keys: '1'=speak word, '2'=toggle sentence, '3'=add word, 'q'=quit",
            (10, h_frame - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            1,
            cv2.LINE_AA
        )
        # -------- end UI overlay --------

        cv2.imshow("ISL Realtime Prediction (Simple)", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        # '1' -> speak current word
        if key == ord("1") and current_label:
            print("Speak word (1):", current_label)
            speak_text(current_label)
            last_spoken_label = current_label
            last_spoken_time = time.time()

        # '2' -> toggle sentence mode
        if key == ord("2"):
            sentence_mode = not sentence_mode

            if sentence_mode:
                # starting a new sentence
                sentence_words = []
                print("Sentence mode: ON")
            else:
                # finishing sentence: speak what has been added so far
                print("Sentence mode: OFF")
                if sentence_words:
                    words_clean = collapse_consecutive(sentence_words)
                    sentence = " ".join(words_clean)
                    print("Speak sentence (2 off):", sentence)
                    speak_text(sentence)
                else:
                    print("Sentence empty, nothing to speak.")

        # '3' -> add current word to sentence (only if in sentence mode)
        if key == ord("3") and sentence_mode and current_label:
            sentence_words.append(current_label)
            print("Added to sentence (3):", current_label)

        # You could also add a key to clear the sentence if needed

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()