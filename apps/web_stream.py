"""
VieNeu-TTS v3 Turbo (int8) — CPU streaming demo (FastAPI).
==========================================================
Stream 48 kHz audio ngay khi generate, qua `V3TurboVieNeuTTS.infer_stream` (đường
ONNX/CPU int8 mặc định). Vì RTF < 1 (int8 nhanh hơn realtime), stream chạy mượt
không underrun — chỉ cần player prebuffer ~300–500ms.

    uv run python -m apps.web_stream        # http://127.0.0.1:8001

Public API dùng ở đây:
    vieneu = Vieneu(backend="onnx")                    # v3 Turbo int8, ép CPU/ONNX
    for chunk in vieneu.infer_stream(text, voice="Minh Đức"):
        ...                                         # np.float32 @ 48kHz, phát/ghi dần
"""
import io
import json
import time
import wave
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse, Response
from pydantic import BaseModel
import uvicorn

SAMPLE_RATE = 48_000
app = FastAPI()
vieneu = None
_model_error: Optional[str] = None
_model_lock = Lock()

ROOT_DIR = Path(__file__).resolve().parents[1]
CLIENT_HTML_PATH = ROOT_DIR / "client" / "client.html"
VOICES_PATH = ROOT_DIR / "src" / "vieneu" / "assets" / "voices_v3_turbo.json"


def fallback_voices():
    """Return the curated preset names without loading the TTS model."""
    try:
        data = json.loads(VOICES_PATH.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            presets = data.get("presets", data.get("voices", []))
            if isinstance(presets, dict):
                return [{"id": str(name), "name": str(name)} for name in presets]
            data = presets
        if isinstance(data, list):
            return [
                {"id": str(item.get("id", item.get("name", ""))), "name": str(item.get("name", item.get("id", "")))}
                for item in data
                if isinstance(item, dict) and (item.get("id") or item.get("name"))
            ]
    except (OSError, json.JSONDecodeError, TypeError):
        pass
    return [{"id": "Minh Đức", "name": "Minh Đức"}]


def load_model():
    global vieneu, _model_error
    if vieneu is not None:
        return vieneu
    with _model_lock:
        if vieneu is not None:
            return vieneu
        try:
            print("[v0] Loading VieNeu-TTS v3 Turbo (int8, CPU)...")
            from vieneu import Vieneu
            vieneu = Vieneu(backend="onnx")
            print("[v0] VieNeu-TTS model ready")
            return vieneu
        except Exception as exc:
            _model_error = str(exc)
            print(f"[v0] VieNeu-TTS model failed to load: {_model_error}")
            raise HTTPException(status_code=503, detail="TTS model is unavailable") from exc


@app.get("/")
async def ui():
    if CLIENT_HTML_PATH.exists():
        return FileResponse(CLIENT_HTML_PATH, media_type="text/html")
    return Response("client.html not found", status_code=404, media_type="text/plain")


@app.get("/favicon.ico")
async def favicon():
    svg = '<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 100 100"><text y="82" font-size="82">🐆</text></svg>'
    return Response(svg, media_type="image/svg+xml")


@app.get("/voices")
async def voices():
    # Voice discovery must not make the entire UI unusable when a serverless
    # runtime cannot download/load the model during a cold start.
    if vieneu is None:
        return fallback_voices()
    try:
        vs = vieneu.list_preset_voices()
        out = []
        for item in vs:
            if isinstance(item, (tuple, list)) and len(item) == 2:
                label, vid = item
                out.append({"id": vid, "name": label})
            else:
                out.append({"id": str(item), "name": str(item)})
        return out or fallback_voices()
    except Exception:
        return fallback_voices()


@app.get("/health")
async def health():
    return {"ok": True, "model_loaded": vieneu is not None, "model_error": _model_error}


def _pcm16(audio_f32) -> bytes:
    import numpy as np

    return (np.asarray(audio_f32) * 32767).clip(-32768, 32767).astype(np.int16).tobytes()


@app.get("/stream")
async def stream(text: str, voice_id: Optional[str] = None):
    """Stream 48 kHz WAV: header rồi PCM16 theo từng chunk."""
    engine = load_model()

    def gen():
        # WAV header 48 kHz mono; nframes huge → browser phát liền khi data tới.
        h = io.BytesIO()
        with wave.open(h, "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(SAMPLE_RATE)
            w.setnframes(1_000_000_000)
        yield h.getvalue()

        t0 = time.perf_counter()
        first_at = None
        n_chunks = 0
        emitted = 0
        for chunk in engine.infer_stream(text, voice=voice_id or None):
            if chunk is None or len(chunk) == 0:
                continue
            if first_at is None:
                first_at = time.perf_counter() - t0
                print(f"⚡ TTFA (time-to-first-audio): {first_at*1000:.0f} ms")
            n_chunks += 1
            emitted += len(chunk)
            yield _pcm16(chunk)
        if first_at is not None:
            gen_time = time.perf_counter() - t0
            audio_s = emitted / SAMPLE_RATE
            rtf = gen_time / audio_s if audio_s else 0
            print(f"✅ {n_chunks} chunks | audio {audio_s:.2f}s | gen {gen_time:.2f}s "
                  f"| RTF {rtf:.3f} ({1/rtf:.1f}x realtime)" if rtf else "")

    return StreamingResponse(gen(), media_type="audio/wav")


class StreamReq(BaseModel):
    text: str
    voice_id: Optional[str] = None


@app.post("/stream")
async def stream_post(req: StreamReq):
    return await stream(req.text, req.voice_id)


def main():
    print("🌍 Mở http://localhost:8001 để test VieNeu v3 Turbo (int8) streaming (CPU)")
    uvicorn.run(app, host="127.0.0.1", port=8001)


if __name__ == "__main__":
    main()
