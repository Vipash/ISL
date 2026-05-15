import streamlit as st

st.set_page_config(
    page_title="ISL 2-Way Translator",
    layout="wide",
)

st.title("ISL 2-Way Hand Sign Translator")
st.write(
    "This app demonstrates my final-year project: a real-time Indian Sign Language "
    "hand sign recognizer using MediaPipe, OpenCV, and a custom mean-vector classifier."
)

st.markdown("### What this project does")
st.markdown(
    """
- Uses **MediaPipe Hand Landmarker** to detect 3D hand landmarks.
- Converts landmarks to a vector and classifies against mean vectors stored in `isl_simple_model.json`.
- Supports **real-time webcam prediction** in a desktop app.
- Adds **sentence mode** with hotkeys:
  - `1` → speak current word
  - `2` → toggle sentence mode
  - `3` → add current word to the sentence
"""
)

st.markdown("### How to run the realtime app locally")

st.code(
    """
git clone https://github.com/your-username/your-repo.git
cd your-repo

# create and activate venv (Windows example)
python -m venv venv
venv\\Scripts\\activate

pip install -r requirements_local.txt

# run once.py to build isl_simple_model.json
python once.py

# run the realtime webcam app
python realtime_predict_simple.py
""",
    language="bash",
)

st.info(
    "Note: The full realtime app uses OpenCV and MediaPipe tasks, which are not "
    "currently compatible with Streamlit Community Cloud's Python/OpenCV environment. "
    "Run it locally for full functionality."
)

st.markdown("### Repository link")
st.markdown(
    "[View the complete source code on GitHub](https://github.com/Vipash/ISL)"
)

st.markdown("### Screenshots / Demo")
