"""FastAPI server for Faster Qwen3-TTS — compatible with livetalk TTS WebSocket protocol.

WebSocket endpoint: ws://host:port/tts/ws
Upload endpoint:    POST /upload_voice

Env:
  QWEN3_TTS_MODEL   - Model path or HuggingFace ID (default: Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice)
  TTS_DATA_DIR      - Base dir for voice storage (default: project root)
  HOST              - Bind address (default: 0.0.0.0)
  PORT              - Port (default: 10086)
  DEBUG_TTS               - Set to 1 for debug logging
  CUDA_MEMORY_FRACTION    - Fraction of GPU memory this process may use (e.g. 0.15 for 15%).
                            Useful when running multiple instances on one GPU.
  AWS_ACCESS_KEY_ID       - S3 credentials (optional, for voice S3 upload/download)
  AWS_SECRET_ACCESS_KEY
  AWS_REGION
  VOICE_S3_BUCKET         - S3 bucket for voice WAV files (default: tts-voice-audio)
  VOICE_DB_S3_BUCKET      - S3 bucket for voice DB backup (default: tts-voice-id)
"""

import asyncio
import json
import logging
import os
import sqlite3
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import soundfile as sf
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from faster_qwen3_tts import FasterQwen3TTS

# ═══════════════════════════════════════════════════════════
# Constants
# ═══════════════════════════════════════════════════════════

TARGET_SAMPLE_RATE = 24000
_BASE_DIR = Path(os.environ.get("TTS_DATA_DIR", Path(__file__).resolve().parent))
VOICE_STORAGE_DIR = (_BASE_DIR / "voices").resolve()
VOICE_DB_PATH = (_BASE_DIR / "data" / "voices.db").resolve()
VOICE_S3_BUCKET = os.environ.get("VOICE_S3_BUCKET", "tts-voice-audio")
VOICE_DB_S3_BUCKET = os.environ.get("VOICE_DB_S3_BUCKET", "tts-voice-id")

# ═══════════════════════════════════════════════════════════
# Logging
# ═══════════════════════════════════════════════════════════

logger = logging.getLogger(__name__)
if not logging.getLogger().handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s", datefmt="%Y-%m-%d %H:%M:%S"
    ))
    logging.getLogger().addHandler(_h)
    logging.getLogger().setLevel(
        logging.DEBUG if os.environ.get("DEBUG_TTS") else logging.INFO
    )

# ═══════════════════════════════════════════════════════════
# Voice storage (SQLite)
# ═══════════════════════════════════════════════════════════

_voice_db_conn = None


def ensure_voice_storage():
    global _voice_db_conn
    VOICE_STORAGE_DIR.mkdir(parents=True, exist_ok=True)
    VOICE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _voice_db_conn is None:
        _voice_db_conn = sqlite3.connect(str(VOICE_DB_PATH), check_same_thread=False)
        _voice_db_conn.row_factory = sqlite3.Row
    _voice_db_conn.execute("""
        CREATE TABLE IF NOT EXISTS voices (
            voice_id TEXT PRIMARY KEY, user_id TEXT, filename TEXT NOT NULL,
            duration_sec REAL, sample_rate INTEGER, size_bytes INTEGER,
            format TEXT NOT NULL, created_at TEXT NOT NULL
        )""")
    _voice_db_conn.commit()


def _voice_db():
    if _voice_db_conn is None:
        ensure_voice_storage()
    return _voice_db_conn


def resolve_voice_file(voice_id: str):
    """Return local Path for voice_id, or None if not found. Falls back to S3 download."""
    ensure_voice_storage()
    local = VOICE_STORAGE_DIR / f"{voice_id}.wav"
    if local.exists():
        return local
    # Try S3 download
    try:
        s3 = _get_s3_client()
        logger.info("Downloading voice %s from S3 bucket %s ...", voice_id, VOICE_S3_BUCKET)
        tmp = str(local) + ".tmp"
        with open(tmp, "wb") as fh:
            s3.download_fileobj(VOICE_S3_BUCKET, f"{voice_id}.wav", fh)
        os.replace(tmp, str(local))
        logger.info("Voice %s downloaded from S3 and cached locally.", voice_id)
        return local
    except Exception as e:
        logger.warning("S3 download failed for voice_id=%s: %s", voice_id, e)
        return None


def _get_s3_client():
    import boto3
    ak = os.environ.get("AWS_ACCESS_KEY_ID")
    sk = os.environ.get("AWS_SECRET_ACCESS_KEY")
    region = os.environ.get("AWS_REGION", "us-east-1")
    if not ak or not sk:
        raise RuntimeError("Missing AWS credentials (AWS_ACCESS_KEY_ID / AWS_SECRET_ACCESS_KEY)")
    return boto3.client("s3", region_name=region, aws_access_key_id=ak, aws_secret_access_key=sk)


def _get_s3_client_for_api():
    """Same as _get_s3_client but raises HTTPException (for use inside FastAPI route handlers)."""
    try:
        return _get_s3_client()
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


def upload_file_to_s3(path, bucket, key, content_type=None):
    s3 = _get_s3_client()
    extra = {"ContentType": content_type} if content_type else {}
    s3.upload_file(str(path), bucket, key, ExtraArgs=extra if extra else None)


def _resolve_voice_s3_key(voice_s3_key: str):
    """Download voice from an S3 key (s3://bucket/key or bucket/key), return local Path or None."""
    try:
        s3 = _get_s3_client()
    except Exception:
        return None
    bucket = VOICE_S3_BUCKET
    key = voice_s3_key
    if key.startswith("s3://"):
        key = key[5:]
    if "/" in key:
        bucket, key = key.split("/", 1)
    try:
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        tmp.close()
        s3.download_file(bucket, key, tmp.name)
        return Path(tmp.name)
    except Exception as e:
        logger.warning("Failed to download voice from S3 key %s: %s", voice_s3_key, e)
        return None


def load_audio_to_mono_24k(path, max_seconds=30.0):
    """Load audio → mono float32 @ 24 kHz. Returns (array, duration_sec)."""
    cleanup = None
    try:
        try:
            audio, sr = sf.read(str(path), always_2d=False)
        except Exception:
            cleanup = _ffmpeg_decode(path)
            audio, sr = sf.read(cleanup, always_2d=False)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        audio = audio.astype(np.float32)
        if sr != TARGET_SAMPLE_RATE:
            n = int(round(len(audio) * TARGET_SAMPLE_RATE / sr))
            audio = np.interp(
                np.linspace(0, len(audio) - 1, n), np.arange(len(audio)), audio
            ).astype(np.float32)
        if max_seconds:
            audio = audio[: int(TARGET_SAMPLE_RATE * max_seconds)]
        return audio, len(audio) / TARGET_SAMPLE_RATE
    finally:
        if cleanup and os.path.exists(cleanup):
            os.remove(cleanup)


def _ffmpeg_decode(src):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    subprocess.run(
        ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(TARGET_SAMPLE_RATE), tmp.name],
        check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    return tmp.name


# ═══════════════════════════════════════════════════════════
# Audio helpers
# ═══════════════════════════════════════════════════════════

def _float_to_pcm16(wav: np.ndarray) -> bytes:
    return (np.clip(wav, -1.0, 1.0) * 32767).astype(np.int16).tobytes()


def _resample(wav: np.ndarray, orig_sr: int, target_sr: int = TARGET_SAMPLE_RATE) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    wav = wav.flatten().astype(np.float32)
    n = int(round(len(wav) * target_sr / orig_sr))
    if n == 0:
        return wav
    return np.interp(
        np.linspace(0, len(wav) - 1, n), np.arange(len(wav)), wav
    ).astype(np.float32)


def _chunk_to_pcm16(audio_chunk, sr: int) -> bytes:
    """Convert float32 numpy chunk (or list of chunks) → PCM16 bytes @ TARGET_SAMPLE_RATE."""
    if isinstance(audio_chunk, list):
        parts = [np.array(a, dtype=np.float32).flatten() for a in audio_chunk if len(a) > 0]
        wav = np.concatenate(parts) if parts else np.zeros(0, dtype=np.float32)
    else:
        wav = np.array(audio_chunk, dtype=np.float32).flatten()
    if sr != TARGET_SAMPLE_RATE:
        wav = _resample(wav, sr, TARGET_SAMPLE_RATE)
    return _float_to_pcm16(wav)


# ═══════════════════════════════════════════════════════════
# Model
# ═══════════════════════════════════════════════════════════

_model: FasterQwen3TTS = None
_generation_lock: asyncio.Lock = None


def _model_name() -> str:
    return os.environ.get("QWEN3_TTS_MODEL", "Qwen/Qwen3-TTS-12Hz-0.6B-Base")


def _get_model_type() -> str:
    """Returns 'base', 'custom_voice', or 'voice_design' (or None if not loaded)."""
    if _model is None:
        return None
    try:
        return _model.model.model.tts_model_type
    except Exception:
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _generation_lock
    _generation_lock = asyncio.Lock()

    model_name = _model_name()
    logger.info(f"Loading model: {model_name}")

    def _load():
        # Optional per-process VRAM limit. E.g. CUDA_MEMORY_FRACTION=0.15 caps at
        # 15% of total GPU memory, useful when running multiple instances on one GPU.
        frac = os.environ.get("CUDA_MEMORY_FRACTION")
        if frac:
            try:
                torch.cuda.set_per_process_memory_fraction(float(frac), device=0)
                logger.info(f"CUDA memory fraction set to {frac}")
            except Exception as e:
                logger.warning(f"Failed to set CUDA memory fraction: {e}")
        return FasterQwen3TTS.from_pretrained(
            model_name, device="cuda", dtype=torch.bfloat16
        )

    _model = await asyncio.to_thread(_load)
    logger.info(f"Model loaded (type={_get_model_type()}), warming up CUDA graphs...")

    def _warmup():
        model_type = _get_model_type()
        try:
            if model_type == "custom_voice":
                speakers = _model.model.get_supported_speakers() or ["Vivian"]
                for _ in _model.generate_custom_voice_streaming(
                    text="Hello.", speaker=speakers[0], language="English", chunk_size=12
                ):
                    pass
            elif model_type == "voice_design":
                for _ in _model.generate_voice_design_streaming(
                    text="Hello.", instruct="neutral voice", language="English", chunk_size=12
                ):
                    pass
            else:
                # Base model: CUDA graphs captured on first real call (requires ref audio)
                logger.info("Base model: CUDA graph warmup deferred to first request")
        except Exception as e:
            logger.warning(f"Warmup failed (non-fatal, will retry on first request): {e}")

    await asyncio.to_thread(_warmup)
    logger.info("Server ready.")
    yield


# ═══════════════════════════════════════════════════════════
# Voice resolution
# ═══════════════════════════════════════════════════════════

async def _download_voice(voice_url: str) -> str:
    """Download voice audio from URL → temp file, return local path."""
    import urllib.request

    def _dl():
        parsed = urlparse(voice_url)
        if parsed.scheme not in ("http", "https"):
            raise ValueError("voice_url must use http or https")
        suffix = Path(parsed.path).suffix or ".wav"
        req = urllib.request.Request(
            voice_url, headers={"User-Agent": "faster-qwen3-tts", "Accept": "*/*"}
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(resp.read())
                return tmp.name

    return await asyncio.to_thread(_dl)


async def _resolve_voice(payload: dict):
    """
    Parse voice fields from payload.

    Returns (ref_audio_path, ref_text, speaker, x_vector_only_mode, download_tmp).
    - ref_audio_path: local path for Base model voice clone (or None)
    - speaker: named speaker string for CustomVoice model (or None)
    - download_tmp: temp path to clean up after generation (or None)
    """
    voice = payload.get("voice", "Vivian")
    voice_id = payload.get("voice_id")
    voice_url = payload.get("voice_url")
    voice_s3_key = payload.get("voice_s3_key")
    ref_text = payload.get("reference_text") or payload.get("ref_text") or ""
    x_raw = payload.get("x_vector_only_mode")
    x_vector_only_mode = bool(x_raw) if x_raw is not None else True

    ref_audio = None
    download_tmp = None

    if voice_id:
        candidate = resolve_voice_file(str(voice_id))
        if candidate is None:
            raise HTTPException(status_code=400, detail=f"Unknown voice_id: {voice_id}")
        ref_audio = str(candidate)
        voice = None
    elif voice_url:
        download_tmp = await _download_voice(voice_url)
        ref_audio = download_tmp
        voice = None
    elif voice_s3_key:
        resolved = await asyncio.to_thread(_resolve_voice_s3_key, voice_s3_key)
        if resolved is None:
            raise HTTPException(status_code=400, detail=f"Failed to resolve voice_s3_key: {voice_s3_key}")
        ref_audio = str(resolved)
        download_tmp = str(resolved)
        voice = None

    return ref_audio, ref_text, voice, x_vector_only_mode, download_tmp


# ═══════════════════════════════════════════════════════════
# Synchronous generation (runs in a thread)
# ═══════════════════════════════════════════════════════════

def _run_generation(
    payload: dict,
    loop: asyncio.AbstractEventLoop,
    queue: asyncio.Queue,
    ref_audio: str = None,
    ref_text: str = "",
    speaker: str = None,
    x_vector_only_mode: bool = True,
    download_tmp: str = None,
):
    """Run the synchronous streaming generator and push PCM16 chunks to an asyncio queue.

    Sends raw bytes for audio chunks, None as sentinel on completion,
    or an Exception on error.
    """
    def _put(item):
        loop.call_soon_threadsafe(queue.put_nowait, item)

    try:
        if _model is None:
            _put(RuntimeError("Model not loaded"))
            return

        text = payload.get("text") or payload.get("input") or ""
        language = payload.get("language", "Auto")
        chunk_size = int(payload.get("chunk_size") or 12)
        instructions = payload.get("instructions") or payload.get("instruct") or ""
        model_type = _get_model_type()

        if model_type == "custom_voice":
            gen = _model.generate_custom_voice_streaming(
                text=text,
                speaker=speaker or "Vivian",
                language=language,
                instruct=instructions or None,
                chunk_size=chunk_size,
            )
        elif model_type == "voice_design":
            gen = _model.generate_voice_design_streaming(
                text=text,
                instruct=instructions,
                language=language,
                chunk_size=chunk_size,
            )
        else:
            # Base model — requires ref audio
            if not ref_audio:
                _put(RuntimeError(
                    "Base model requires a voice reference. "
                    "Provide voice_id (from /upload_voice) or voice_url."
                ))
                return
            gen = _model.generate_voice_clone_streaming(
                text=text,
                language=language,
                ref_audio=ref_audio,
                ref_text=ref_text,
                xvec_only=x_vector_only_mode,
                chunk_size=chunk_size,
            )

        for audio_chunk, sr, timing in gen:
            pcm = _chunk_to_pcm16(audio_chunk, sr)
            if pcm:
                _put(pcm)

        _put(None)  # sentinel: done

    except Exception as e:
        logger.error(f"Generation error: {e}", exc_info=True)
        _put(e)
    finally:
        if download_tmp and os.path.exists(download_tmp):
            try:
                os.remove(download_tmp)
            except Exception:
                pass


# ═══════════════════════════════════════════════════════════
# FastAPI app
# ═══════════════════════════════════════════════════════════

app = FastAPI(title="Faster Qwen3-TTS Server", version="0.2.2", lifespan=lifespan)


@app.get("/health")
async def health():
    if _model is None:
        return JSONResponse({"status": "error", "detail": "model not loaded"}, status_code=503)
    return {"status": "ok", "model": _model_name(), "model_type": _get_model_type()}


@app.websocket("/tts/ws")
async def websocket_tts(websocket: WebSocket):
    """WebSocket streaming TTS — livetalk/vllm-omni compatible protocol.

    Client sends JSON:
        {"text": "...", "voice": "Vivian", "voice_id": "...", "voice_url": "...",
         "reference_text": "...", "x_vector_only_mode": true,
         "language": "English", "chunk_size": 12, "instructions": "..."}

    Server responds:
        {"type": "metadata", "sample_rate": 24000, "channels": 1, "format": "pcm16"}
        <binary PCM16 frames> ...
        {"type": "complete", "chunks": N, "total_time": float}
        {"type": "error", "message": "..."}  (on error)
    """
    IDLE_TIMEOUT = 30
    await websocket.accept()

    try:
        while True:
            # ── Receive request ────────────────────────────────────────
            try:
                payload = await asyncio.wait_for(
                    websocket.receive_json(), timeout=IDLE_TIMEOUT
                )
            except asyncio.TimeoutError:
                await websocket.send_json({"type": "error", "message": "idle timeout"})
                await websocket.close()
                return
            except WebSocketDisconnect:
                return
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"Invalid JSON: {e}"})
                await websocket.close()
                return

            text = payload.get("text") or payload.get("input") or ""
            if not text:
                await websocket.send_json({"type": "error", "message": "text parameter required"})
                continue

            # ── Resolve voice ──────────────────────────────────────────
            try:
                ref_audio, ref_text, speaker, x_vector_only_mode, download_tmp = (
                    await _resolve_voice(payload)
                )
            except HTTPException as e:
                await websocket.send_json({"type": "error", "message": e.detail})
                continue
            except Exception as e:
                await websocket.send_json({"type": "error", "message": f"Voice resolution failed: {e}"})
                continue

            # ── Send metadata ──────────────────────────────────────────
            await websocket.send_json({
                "type": "metadata",
                "sample_rate": TARGET_SAMPLE_RATE,
                "channels": 1,
                "format": "pcm16",
            })

            # ── Stream generation (thread → asyncio.Queue → WebSocket) ─
            loop = asyncio.get_running_loop()
            queue: asyncio.Queue = asyncio.Queue()

            async with _generation_lock:
                thread = threading.Thread(
                    target=_run_generation,
                    args=(payload, loop, queue),
                    kwargs=dict(
                        ref_audio=ref_audio,
                        ref_text=ref_text,
                        speaker=speaker,
                        x_vector_only_mode=x_vector_only_mode,
                        download_tmp=download_tmp,
                    ),
                    daemon=True,
                )
                thread.start()

                t0 = time.time()
                chunk_count = 0
                error_msg = None

                try:
                    while True:
                        item = await asyncio.wait_for(queue.get(), timeout=120)
                        if item is None:
                            break
                        if isinstance(item, Exception):
                            error_msg = str(item)
                            break
                        await websocket.send_bytes(item)
                        chunk_count += 1
                except asyncio.TimeoutError:
                    error_msg = "generation timeout"
                except WebSocketDisconnect:
                    return

                thread.join(timeout=5)

            if error_msg:
                try:
                    await websocket.send_json({"type": "error", "message": error_msg})
                except Exception:
                    pass
            else:
                try:
                    await websocket.send_json({
                        "type": "complete",
                        "chunks": chunk_count,
                        "total_time": round(time.time() - t0, 3),
                    })
                except Exception:
                    pass

    finally:
        try:
            await websocket.close()
        except Exception:
            pass


@app.post("/upload_voice")
async def upload_voice(
    voice_file: UploadFile = File(None),
    voice_url: str = Form(None),
):
    """Upload a voice sample → normalise to 24 kHz mono WAV → store locally → return voice_id.

    The returned voice_id can be passed as voice_id in /tts/ws requests.
    """
    ensure_voice_storage()
    if voice_url is None and voice_file is None:
        raise HTTPException(status_code=400, detail="voice_url or voice_file is required")

    source_path = None
    download_tmp = None
    dest_path = None

    try:
        if voice_url:
            download_tmp = await _download_voice(voice_url)
            source_path = download_tmp
        else:
            suffix = Path(voice_file.filename or "").suffix or ".wav"
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(await voice_file.read())
                source_path = tmp.name

        audio, duration = await asyncio.to_thread(
            load_audio_to_mono_24k, source_path, 600.0
        )

        voice_id = uuid.uuid4().hex
        dest_path = VOICE_STORAGE_DIR / f"{voice_id}.wav"
        sf.write(str(dest_path), audio, samplerate=TARGET_SAMPLE_RATE, subtype="PCM_16")

        size_bytes = dest_path.stat().st_size
        created_at = datetime.utcnow().isoformat() + "Z"
        _voice_db().execute(
            "INSERT INTO voices "
            "(voice_id,user_id,filename,duration_sec,sample_rate,size_bytes,format,created_at) "
            "VALUES (?,?,?,?,?,?,?,?)",
            (voice_id, "default", dest_path.name, duration,
             TARGET_SAMPLE_RATE, size_bytes, "wav", created_at),
        )
        _voice_db().commit()

        # S3 upload (best-effort, non-blocking)
        try:
            upload_file_to_s3(dest_path, VOICE_S3_BUCKET, dest_path.name, "audio/wav")
            upload_file_to_s3(VOICE_DB_PATH, VOICE_DB_S3_BUCKET, VOICE_DB_PATH.name, "application/octet-stream")
            logger.info(f"Voice synced to S3: {dest_path.name}")
        except Exception as e:
            logger.warning(f"S3 upload failed (voice still saved locally): {e}")

        logger.info(f"Voice uploaded: voice_id={voice_id}, duration={duration:.1f}s")
        return JSONResponse({"voice_id": voice_id, "created_at": created_at})

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Failed to process voice: {e}")
    finally:
        for p in {download_tmp, source_path}:
            if p and dest_path and os.path.abspath(str(p)) == str(dest_path):
                continue
            if p and os.path.exists(str(p)):
                try:
                    os.remove(p)
                except Exception:
                    pass


@app.get("/")
async def root():
    return {
        "name": "Faster Qwen3-TTS Server",
        "model": _model_name(),
        "model_type": _get_model_type(),
        "endpoints": [
            "WS  /tts/ws",
            "POST /upload_voice",
            "GET  /health",
        ],
    }


if __name__ == "__main__":
    import uvicorn
    host = os.environ.get("HOST", "0.0.0.0")
    port = int(os.environ.get("PORT", "10086"))
    uvicorn.run(app, host=host, port=port)
