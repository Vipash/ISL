from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
import numpy as np
import mediapipe as mp
from mediapipe.tasks.python import vision
import json
import uvicorn

MODEL_JSON = "isl_simple_model.json"
HAND_MODEL_PATH = "hand_landmarker.task"

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

app = FastAPI(title="ISL Hand Sign API")


def load_means():
    with open(MODEL_JSON, "r") as f:
        means = json.load(f)
    return {label: np.array(vec, dtype=np.float32) for label, vec in means.items()}


def create_landmarker(model_path=HAND_MODEL_PATH):
    options = HandLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        num_hands=1,
        running_mode=VisionRunningMode.IMAGE,
    )
    return HandLandmarker.create_from_options(options)


means = load_means()
landmarker = create_landmarker()


def landmarks_to_vector(landmarks):
    vec = []
    for lm in landmarks:
        vec.extend([lm.x, lm.y, lm.z])
    return np.array(vec, dtype=np.float32)


def predict_label(vec, means_dict):
    best_label = None
    best_dist = None
    for label, mean_vec in means_dict.items():
        if vec.shape != mean_vec.shape:
            continue
        dist = np.linalg.norm(vec - mean_vec)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_label = label
    return best_label, best_dist


@app.get("/")
async def root():
    return {"message": "ISL Hand Sign API. Use POST /predict with an image file."}


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        data = await file.read()
        if not data:
            raise HTTPException(status_code=400, detail="Empty file")

        np_data = np.frombuffer(data, dtype=np.uint8)

        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np_data
        )

        result = landmarker.detect(mp_image)

        if not result.hand_landmarks:
            raise HTTPException(status_code=400, detail="No hand detected")

        hand = result.hand_landmarks[0]
        vec = landmarks_to_vector(hand)
        any_vec = next(iter(means.values()))
        if vec.shape != any_vec.shape:
            raise HTTPException(status_code=500, detail="Vector shape mismatch")

        label, dist = predict_label(vec, means)
        if label is None:
            raise HTTPException(status_code=500, detail="No valid prediction")

        return JSONResponse(
            content={"label": label, "distance": float(dist)}
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("api_app:app", host="0.0.0.0", port=8000, reload=True)