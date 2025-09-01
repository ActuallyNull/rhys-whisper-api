import uvicorn
from whisper_jax import FlaxWhisperPipline
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np
import time

print("--- Server starting, loading model... ---")
# Initialize the pipeline
pipeline = FlaxWhisperPipline("openai/whisper-tiny")
print("--- Model loaded successfully. Server is ready. ---")

# Initialize FastAPI app
app = FastAPI()

# --- Add CORS middleware ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    print("[/transcribe] - Request received.")
    try:
        start_time = time.time()
        
        print("[/transcribe] - Reading file contents...")
        contents = await file.read()
        read_time = time.time()
        print(f"[/transcribe] - File read in {read_time - start_time:.2f} seconds.")

        audio_array = np.frombuffer(contents, dtype=np.int16)
        
        print("[/transcribe] - Starting transcription...")
        transcript = pipeline({"array": audio_array, "sampling_rate": 16000})
        transcribe_time = time.time()
        print(f"[/transcribe] - Transcription finished in {transcribe_time - read_time:.2f} seconds.")
        
        return {"transcript": transcript["text"]}

    except Exception as e:
        print(f"[/transcribe] - An error occurred: {str(e)}")
        return {"error": str(e)}

# Health check endpoint
@app.get("/")
def read_root():
    print("[/] - Health check requested.")
    return {"status": "ok"}

# Run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
