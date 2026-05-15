import streamlit as st
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import json

MODEL_JSON = "isl_simple_model.json"
HAND_MODEL_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode


@st.cache_resource(show_spinner=False)
def load_means():
    with open(MODEL_JSON, "r") as f:
        means = json.load(f)
    return {label: np.array(vec, dtype=np.float32) for label, vec in means.items()}


@st.cache_resource(show_spinner=False)
def create_landmarker(model_path=HAND_MODEL_PATH):
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
    return best_label, best_dist


def process_image(image_bytes, landmarker, means):
    # Read raw bytes into a numpy array
    data = image_bytes.read()
    if not data:
        return None, None, None

    # MediaPipe expects RGB image data as a numpy array
    # Streamlit provides image in standard formats; we can use Mediapipe's internal loader
    np_data = np.frombuffer(data, dtype=np.uint8)

    # Decode image using MediaPipe Image API
    # Note: mp.Image has a create_from_file method but we have bytes,
    # so we use the data argument with SRGB format.
    mp_image = mp.Image(
        image_format=mp.ImageFormat.SRGB,
        data=np_data
    )

    result = landmarker.detect(mp_image)

    if not result.hand_landmarks:
        return None, None, None

    hand = result.hand_landmarks[0]

    vec = landmarks_to_vector(hand)
    any_vec = next(iter(means.values()))
    if vec.shape != any_vec.shape:
        return None, None, None

    label, dist = predict_label(vec, means)
    return mp_image, label, dist


def main():
    st.set_page_config(page_title="ISL Hand Sign Translator", layout="wide")

    st.title("ISL Hand Sign Translator (Image Demo - MediaPipe Only)")
    st.write(
        "Upload a hand image (or capture from camera) with a single ISL sign. "
        "The app will detect the hand landmarks and predict the closest label using the ISL model."
    )

    means = load_means()
    landmarker = create_landmarker()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.subheader("1. Upload or capture an image")

        uploaded_file = st.file_uploader(
            "Choose an image file (jpg, png)", type=["jpg", "jpeg", "png"]
        )

        camera_image = st.camera_input("Or capture from your camera")

        source_image = None
        source_name = None
        if camera_image is not None:
            source_image = camera_image
            source_name = "Camera"
        elif uploaded_file is not None:
            source_image = uploaded_file
            source_name = "Upload"

        if source_image is not None:
            st.info(f"Using image from: {source_name}")
        else:
            st.warning("Upload an image or capture one using the camera above.")

    with col_right:
        st.subheader("2. Result")

        if source_image is not None:
            mp_image, label, dist = process_image(source_image, landmarker, means)

            if mp_image is None:
                st.error("No hand detected or vector shape mismatch. Try a clearer image.")
                return

            # Show the original image (Streamlit can display uploaded/camera images directly)
            st.image(source_image, caption="Input image", use_column_width=True)

            if label is None:
                st.error("Could not compute a valid prediction. Try a different image.")
            else:
                st.success(f"Predicted label: **{label}**")
                st.write(f"Distance to mean vector: `{dist:.4f}`")
        else:
            st.info("Prediction will appear here after you provide an image.")


if __name__ == "__main__":
    main()