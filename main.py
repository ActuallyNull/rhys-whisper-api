import uvicorn
from whisper_jax import FlaxWhisperPipline
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import base64
import time
import os

# Define the request body model using Pydantic
class TranscriptionRequest(BaseModel):
    audio_chunk: str  # This will be the base64 encoded string

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
async def transcribe(request: TranscriptionRequest):
    print("[/transcribe] - JSON request received.")
    try:
        start_time = time.time()
        
        # Decode the base64 string to bytes
        print("[/transcribe] - Decoding base64 chunk...")
        contents = base64.b64decode(request.audio_chunk)
        decode_time = time.time()
        print(f"[/transcribe] - Chunk decoded in {decode_time - start_time:.2f} seconds.")

        audio_array = np.frombuffer(contents, dtype=np.int16)
        
        print("[/transcribe] - Starting transcription...")
        transcript = pipeline({"array": audio_array, "sampling_rate": 16000})
        transcribe_time = time.time()
        print(f"[/transcribe] - Transcription finished in {transcribe_time - decode_time:.2f} seconds.")
        print(f"[/transcribe] - Returned transcript: '{transcript["text"]}'")
        
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
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
