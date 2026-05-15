import csv
import os
import time

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# === CONFIG ===
OUTPUT_CSV = "isl_dataset.csv"

# Map keys to labels (edit this after you decide your words)
KEY_TO_LABEL = {
    ord('0'): "HELLO",
    ord('1'): "YES",
    ord('2'): "NO",
    ord('3'): "THANKYOU",
    ord('4'): "I",
    ord('5'): "You",
    ord('6'): "Want",
    ord('7'): "Eat",
    ord('8'): "Drink",
    ord('9'): "Go",
}

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

def landmarks_to_row(landmarks, label):
    # landmarks: list of 21 points; each has x,y,z
    row = [label]
    for lm in landmarks:
        row.extend([lm.x, lm.y, lm.z])
    return row

def ensure_csv_header(path):
    if not os.path.exists(path):
        # 21 landmarks * 3 coords = 63 + 1 label
        header = ["label"]
        for i in range(21):
            header += [f"x_{i}", f"y_{i}", f"z_{i}"]
        with open(path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def main():
    # Ensure CSV has header
    ensure_csv_header(OUTPUT_CSV)

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Cannot open webcam")
        return

    landmarker = create_landmarker()

    print("Instructions:")
    print("- Show a gesture and press 0,1,2,... to save a sample for that label.")
    print("- Press 'q' to quit.")

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

        # Draw landmarks for feedback
        if result.hand_landmarks:
            h, w, _ = frame.shape
            for hand in result.hand_landmarks:
                for lm in hand:
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (cx, cy), 4, (0, 255, 0), -1)

        cv2.putText(
            frame,
            "Press 0/1/2... to save, q to quit",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2
        )

        cv2.imshow("Collect ISL Data", frame)
        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):
            break

        if key in KEY_TO_LABEL and result.hand_landmarks:
            label = KEY_TO_LABEL[key]
            hand = result.hand_landmarks[0]
            row = landmarks_to_row(hand, label)
            with open(OUTPUT_CSV, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(row)
            print(f"Saved one sample for label: {label}")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()