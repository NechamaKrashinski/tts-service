import os
import subprocess
import tempfile
from pathlib import Path
import google.auth

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
from elevenlabs.client import ElevenLabs
from elevenlabs.types import DialogueInput
from google.cloud import storage
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from elevenlabs.types.to_dialogue_settings_response_model import ToDialogueSettingsResponseModel

app = FastAPI()

BUCKET = os.environ["AUDIO_BUCKET"]
SECRET = os.environ["SERVICE_SECRET"]
DRIVE_FOLDER_ID = os.environ["DRIVE_FOLDER_ID"]
stability=float(os.getenv("TTS_STABILITY", "0.35"))

def _require_secret(value:str) -> None:
    if value != SECRET:
        raise HTTPException(status_code=403, detail="bad service secret")



class Line(BaseModel):
    text:str
    voice_id: str

class SynthesizeRequest(BaseModel):
    chunk_index: int
    job_id: str
    lines: list[Line]

@app.post("/synthesize")
def synthesize(req: SynthesizeRequest, x_service_secret: str = Header("")):
    _require_secret(x_service_secret)
    client = ElevenLabs()
    inputs = [DialogueInput(text=l.text,voice_id=l.voice_id) for l in req.lines]
    audio = client.text_to_dialogue.convert(
        inputs=inputs,
        model_id="eleven_v3",
        settings=ToDialogueSettingsResponseModel(stability=stability)
    )
    data = b"".join(audio)
    blob = storage.Client().bucket(BUCKET).blob(f"jobs/{req.job_id}/{req.chunk_index:04d}.mp3")
    blob.upload_from_string(data, content_type="audio/mpeg")
    return {"job_id": req.job_id, "chunk_index": req.chunk_index, "bytes": len(data)}


class FinalizeRequest(BaseModel):
    filename: str
    job_id: str

@app.post("/finalize")
def finalize(req: FinalizeRequest, x_service_secret: str = Header("")):
    _require_secret(x_service_secret)
    bucket = storage.Client().bucket(BUCKET)
    blobs = sorted(bucket.list_blobs(prefix=f"jobs/{req.job_id}/"), key=lambda b: b.name)
    if not blobs:
        raise HTTPException(status_code=404, detail="no chunks for this job")

    with tempfile.TemporaryDirectory() as tmp:
        paths = []
        for b in blobs:
            path = Path(tmp) / Path(b.name).name
            b.download_to_filename(path)
            paths.append(path)

        listfile = Path(tmp) / "list.txt"
        listfile.write_text("\n".join(f"file '{p}'" for p in paths))
        out = Path(tmp) / req.filename
        subprocess.run(
            ["ffmpeg", "-f", "concat", "-safe", "0",
             "-i", str(listfile), "-c", "copy", str(out)],
            check=True, capture_output=True,
        )

        creds, _ = google.auth.default(
            scopes=["https://www.googleapis.com/auth/drive"]
        )
        drive = build("drive", "v3", credentials=creds)
        file = drive.files().create(
            body={"name": req.filename, "parents": [DRIVE_FOLDER_ID]},
            media_body=MediaFileUpload(str(out), mimetype="audio/mpeg"),
            fields="id, webViewLink",
        ).execute()

    for b in blobs:
        b.delete()
    return {"url": file["webViewLink"]}
