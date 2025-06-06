from services.kokoroo import GetAudio
import io
from fastapi import FastAPI,HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware  # ✅ Add this
import uvicorn
import torchaudio
import torch


app = FastAPI()

# ✅ Add this CORS middleware config immediately after creating the FastAPI app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # For development, use "*" (any domain). For production, set this to your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TextRequest(BaseModel):
    text: str

@app.post("/generate-audio")
def generate_audio_wav(request: TextRequest):
    try:
        # Generate audio chunks
        synth = GetAudio(request.text)
        audio_generator = synth.generate_audio()

        # Combine all audio chunks into one tensor
        audio_chunks = []
        for audio in audio_generator:
            audio_chunks.append(torch.tensor(audio))  # Ensure it's a tensor

        # Concatenate along time dimension
        full_audio = torch.cat(audio_chunks, dim=0)

        # Save to BytesIO as WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, full_audio.unsqueeze(0).cpu(), 24000, format="wav")
        buffer.seek(0)

        return Response(content=buffer.read(), media_type="audio/wav")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7000)
