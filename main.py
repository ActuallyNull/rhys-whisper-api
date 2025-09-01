import uvicorn
from whisper_jax import FlaxWhisperPipline
from fastapi import FastAPI, File, UploadFile
import numpy as np

# Initialize the pipeline
# Using "openai/whisper-tiny" is a good starting point for CPU instances.
# It's fast and the quality is still very good.
# For higher quality on a paid Render instance, you could use "openai/whisper-base"
# or "openai/whisper-small".
pipeline = FlaxWhisperPipline("openai/whisper-tiny")

# Initialize FastAPI app
app = FastAPI()

@app.post("/transcribe")
async def transcribe(file: UploadFile = File(...)):
    try:
        # Read the audio file from the upload
        contents = await file.read()
        
        # Convert the byte string to a numpy array
        audio_array = np.frombuffer(contents, dtype=np.int16)

        # Transcribe
        transcript = pipeline(audio_array)
        
        # Return the transcript
        return {"transcript": transcript["text"]}

    except Exception as e:
        return {"error": str(e)}

# Run the app with uvicorn
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=10000)
