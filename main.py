import uvicorn
from whisper_jax import FlaxWhisperPipline
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import numpy as np

# Initialize the pipeline
# Using "openai/whisper-tiny" is a good starting point for CPU instances.
pipeline = FlaxWhisperPipline("openai/whisper-tiny")

# Initialize FastAPI app
app = FastAPI()

# --- Add CORS middleware --- 
# This is the key change to allow requests from your web app's domain.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Read the audio file from the upload
        contents = await file.read()
        
        # Convert the byte string to a numpy array
        audio_array = np.frombuffer(contents, dtype=np.int16)

        # Transcribe using the recommended dictionary format
        transcript = pipeline({"array": audio_array, "sampling_rate": 16000})
        
        # Return the transcript
        return {"transcript": transcript["text"]}

    except Exception as e:
        return {"error": str(e)}

# Health check endpoint
@app.get("/")
def read_root():
    return {"status": "ok"}

# Run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
