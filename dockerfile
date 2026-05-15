FROM python:3.11-slim

# (Optional) Install system deps if Mediapipe complains about GL libs
# Uncomment if you see errors about libGL or libglib:
# RUN apt-get update && apt-get install -y \
#     libgl1 \
#     libglib2.0-0 \
#     && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies for Railway
COPY requirements_railway.txt .

RUN pip install --no-cache-dir -r requirements_railway.txt

# Copy the rest of the project
COPY . .

# Railway provides PORT env var; default to 8000 for local
ENV PORT=8000

# Run FastAPI app
CMD ["uvicorn", "api_app:app", "--host", "0.0.0.0", "--port", "8000"]