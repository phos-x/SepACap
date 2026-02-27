import io
import os
import zipfile
import logging
from contextlib import asynccontextmanager

import yaml
import numpy as np
import onnxruntime as ort
import torchaudio
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Load deploy_config.yaml securely
CONFIG_PATH = os.environ.get("SEPACAP_DEPLOY_CONFIG", "/opt/program/app/deploy_config.yaml")
with open(CONFIG_PATH, 'r') as f:
    config = yaml.safe_load(f)

# Global variables to hold the loaded ONNX session
ort_session = None
STEM_NAMES = config['model']['stems']
SAMPLE_RATE = config['model']['sample_rate']
MAX_FILE_SIZE_MB = config['security']['max_upload_size_mb']

# ---------------------------------------------------------------------------
# LIFESPAN (Resource Management)
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Loads the ONNX model into RAM *once* when the server starts.
    If we loaded this inside the /invocations route, the server would crash from Memory Leaks.
    """
    global ort_session
    weights_path = config['paths']['model_weights']
    
    logger.info(f"Booting ONNX Runtime Engine. Loading: {weights_path}")
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"ONNX model not found at {weights_path}")

    # Initialize ONNX Runtime with CPU or GPU providers dynamically
    providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
    ort_session = ort.InferenceSession(weights_path, providers=providers)
    logger.info("ONNX Engine loaded successfully. Ready for inference.")
    
    yield
    
    logger.info("Shutting down ONNX Engine...")
    ort_session = None

app = FastAPI(lifespan=lifespan)

# ---------------------------------------------------------------------------
# API ROUTES 
# ---------------------------------------------------------------------------

@app.get("/ping")
async def ping():
    """
    AWS SageMaker Health Check Route. Must return 200 OK.
    """
    if ort_session is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet.")
    return {"status": "Healthy"}

@app.post("/invocations")
async def predict(file: UploadFile = File(...)):
    """
    The main inference route. Receives a WAV, runs ONNX, returns a ZIP of stems.
    """
    logger.info(f"Received request. File: {file.filename}, Type: {file.content_type}")

    if not file.filename.endswith(('.wav', '.mp3')):
        raise HTTPException(status_code=415, detail="Unsupported media type. Send .wav or .mp3.")

    file_bytes = await file.read()
    file_size_mb = len(file_bytes) / (1024 * 1024)
    if file_size_mb > MAX_FILE_SIZE_MB:
        raise HTTPException(status_code=413, detail=f"File exceeds {MAX_FILE_SIZE_MB}MB limit.")

    try:
        waveform, sr = torchaudio.load(io.BytesIO(file_bytes))
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sr != SAMPLE_RATE:
            resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
            waveform = resampler(waveform)

        input_data = waveform.unsqueeze(0).numpy()
        
        ort_inputs = {ort_session.get_inputs()[0].name: input_data}

        logger.info("Executing ONNX Graph...")
        ort_outs = ort_session.run(None, ort_inputs)
        
        separated_stems = ort_outs[0][0] 
        # In-Memory ZIP File Generation
        # Creating a zip file purely in RAM ensures the server handles high concurrency 
        # without running out of Docker container disk space.
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, "a", zipfile.ZIP_DEFLATED, False) as zip_file:
            for i, stem_name in enumerate(STEM_NAMES):
                stem_audio = torch.from_numpy(separated_stems[i]).unsqueeze(0) # [1, Time]
                
                audio_buffer = io.BytesIO()
                torchaudio.save(audio_buffer, stem_audio, SAMPLE_RATE, format="wav")
                
                zip_file.writestr(f"{stem_name}.wav", audio_buffer.getvalue())

        zip_buffer.seek(0)

        logger.info("Inference complete. Streaming ZIP response.")
        return StreamingResponse(
            zip_buffer, 
            media_type="application/zip",
            headers={"Content-Disposition": f"attachment; filename=separated_stems.zip"}
        )

    except Exception as e:
        logger.error(f"Inference failed: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal Server Error during separation.")