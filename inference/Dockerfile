FROM python:3.10-slim

WORKDIR /opt/program

RUN apt-get update && apt-get install -y ffmpeg libsndfile1 && rm -rf /var/lib/apt/lists/*

# (requirements.txt will only need: fastapi, uvicorn, onnxruntime, torchaudio, python-multipart)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY ./app /opt/program/app
COPY ./weights/sepacap_model.onnx /opt/program/weights/

# Expose port 8080 (SageMaker requires this specific port)
EXPOSE 8080

CMD ["uvicorn", "app.serve:app", "--host", "0.0.0.0", "--port", "8080"]