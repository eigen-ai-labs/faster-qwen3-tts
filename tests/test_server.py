"""Integration tests for ./server.py — all 9 public endpoints.

Run against a live server:
    TTS_SERVER_URL=http://localhost:10086 pytest tests/test_server.py -v

Env vars:
    TTS_SERVER_URL   Base URL of the running server (default: http://localhost:10086)
    TEST_VOICE_FILE  Path to a reference WAV for voice-clone tests (required for base model)
"""

import asyncio
import json
import os

import httpx
import pytest
import websockets

BASE = os.environ.get("TTS_SERVER_URL", "http://localhost:10086")
WS_BASE = BASE.replace("http://", "ws://").replace("https://", "wss://")
TEST_VOICE_FILE = os.environ.get("TEST_VOICE_FILE", "")


# ── Session fixtures ────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def client():
    with httpx.Client(base_url=BASE, timeout=180) as c:
        yield c


@pytest.fixture(scope="session")
def model_type(client):
    r = client.get("/health")
    assert r.status_code == 200, f"/health returned {r.status_code}"
    return r.json().get("model_type")


@pytest.fixture(scope="session")
def first_speaker(client):
    r = client.get("/v1/audio/voices")
    voices = r.json().get("voices", []) if r.status_code == 200 else []
    return voices[0] if voices else "Vivian"


@pytest.fixture(scope="session")
def voice_id(client, model_type):
    """Upload TEST_VOICE_FILE once per session; used by base-model tests.

    For custom_voice / voice_design models this returns None (not needed).
    For base model without TEST_VOICE_FILE the fixture returns None and each
    test that depends on it must skip itself.
    """
    if model_type != "base":
        return None
    if not TEST_VOICE_FILE or not os.path.exists(TEST_VOICE_FILE):
        return None
    with open(TEST_VOICE_FILE, "rb") as f:
        r = client.post("/upload_voice", files={"voice_file": ("ref.wav", f, "audio/wav")})
    assert r.status_code == 200, f"/upload_voice failed: {r.text}"
    return r.json()["voice_id"]


# ── Helpers ─────────────────────────────────────────────────────────────────────

def _is_wav(data: bytes) -> bool:
    return len(data) > 44 and data[:4] == b"RIFF" and data[8:12] == b"WAVE"


def _require_voice(model_type: str, voice_id: str | None) -> None:
    """Skip the test if a base model has no uploaded voice available."""
    if model_type == "base" and not voice_id:
        pytest.skip("Base model requires TEST_VOICE_FILE env var to run this test")


def _speech_request_payload(
    model_type: str,
    first_speaker: str,
    voice_id: str | None,
    text: str = "Hello, this is a test.",
) -> dict:
    """Build a JSON payload for /v1/audio/speech and /v1/audio/speech/stream.

    These endpoints use SpeechRequest (Pydantic), which supports ref_audio as
    a local file path directly — no upload needed.
    """
    payload: dict = {"text": text}
    if model_type == "custom_voice":
        payload["voice"] = first_speaker
    elif model_type == "voice_design":
        payload["instructions"] = "neutral voice"
    else:  # base
        _require_voice(model_type, voice_id)
        payload["ref_audio"] = TEST_VOICE_FILE
    return payload


def _voice_resolution_payload(
    model_type: str,
    first_speaker: str,
    voice_id: str | None,
    text: str = "Hello, this is a test.",
) -> dict:
    """Build a JSON payload for /v1/text-to-speech and WS /tts/ws.

    These endpoints use _resolve_voice(), which only accepts voice_id /
    voice_url / voice_s3_key for reference audio — not a bare local path.
    """
    payload: dict = {"text": text}
    if model_type == "custom_voice":
        payload["voice"] = first_speaker
    elif model_type == "voice_design":
        payload["instructions"] = "neutral voice"
    else:  # base
        _require_voice(model_type, voice_id)
        payload["voice_id"] = voice_id
    return payload


# ── 1. GET / ────────────────────────────────────────────────────────────────────

def test_root(client):
    r = client.get("/")
    assert r.status_code == 200
    body = r.json()
    assert "name" in body
    assert "endpoints" in body


# ── 2. GET /health ──────────────────────────────────────────────────────────────

def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body.get("status") == "ok"
    assert "model" in body
    assert "model_type" in body


# ── 3. GET /v1/models ───────────────────────────────────────────────────────────

def test_v1_models(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body.get("object") == "list"
    assert isinstance(body.get("data"), list)
    assert len(body["data"]) > 0
    assert "id" in body["data"][0]


# ── 4. GET /v1/audio/voices ─────────────────────────────────────────────────────

def test_v1_audio_voices(client):
    r = client.get("/v1/audio/voices")
    assert r.status_code == 200
    body = r.json()
    assert "voices" in body
    assert isinstance(body["voices"], list)


# ── 5. POST /v1/audio/speech ────────────────────────────────────────────────────

def test_v1_audio_speech_wav(client, model_type, first_speaker, voice_id):
    payload = _speech_request_payload(model_type, first_speaker, voice_id)
    payload["response_format"] = "wav"
    r = client.post("/v1/audio/speech", json=payload)
    assert r.status_code == 200
    assert "audio/wav" in r.headers.get("content-type", "")
    assert _is_wav(r.content)
    assert r.headers.get("Sample-Rate") == "24000"


def test_v1_audio_speech_pcm(client, model_type, first_speaker, voice_id):
    payload = _speech_request_payload(model_type, first_speaker, voice_id, text="PCM format.")
    payload["response_format"] = "pcm"
    r = client.post("/v1/audio/speech", json=payload)
    assert r.status_code == 200
    assert "audio/L16" in r.headers.get("content-type", "")
    assert len(r.content) > 0
    assert len(r.content) % 2 == 0  # PCM16 = 2 bytes per sample


def test_v1_audio_speech_missing_text(client):
    r = client.post("/v1/audio/speech", json={"voice": "Vivian"})
    assert r.status_code == 400


# ── 6. POST /v1/audio/speech/stream ─────────────────────────────────────────────

def test_v1_audio_speech_stream(client, model_type, first_speaker, voice_id):
    payload = _speech_request_payload(model_type, first_speaker, voice_id, text="Streaming audio test.")
    chunks = []
    with client.stream("POST", "/v1/audio/speech/stream", json=payload) as resp:
        assert resp.status_code == 200
        assert "audio/L16" in resp.headers.get("content-type", "")
        assert resp.headers.get("Sample-Rate") == "24000"
        for chunk in resp.iter_bytes(chunk_size=4096):
            if chunk:
                chunks.append(chunk)
    assert len(chunks) > 0
    total = sum(len(c) for c in chunks)
    assert total > 0
    assert total % 2 == 0  # PCM16


def test_v1_audio_speech_stream_missing_text(client):
    r = client.post("/v1/audio/speech/stream", json={"voice": "Vivian"})
    assert r.status_code == 400


# ── 7. POST /v1/text-to-speech ──────────────────────────────────────────────────

def test_v1_text_to_speech(client, model_type, first_speaker, voice_id):
    # Uses _resolve_voice() — must supply voice_id for base model, voice name for custom_voice
    payload = _voice_resolution_payload(model_type, first_speaker, voice_id, text="Text to speech endpoint.")
    payload["response_format"] = "wav"
    r = client.post("/v1/text-to-speech", json=payload)
    assert r.status_code == 200
    assert _is_wav(r.content)
    assert r.headers.get("Sample-Rate") == "24000"


def test_v1_text_to_speech_missing_text(client):
    r = client.post("/v1/text-to-speech", json={"voice": "Vivian"})
    assert r.status_code == 400


def test_v1_text_to_speech_invalid_voice_id(client, model_type):
    if model_type != "base":
        pytest.skip("voice_id validation only applies to base model")
    r = client.post("/v1/text-to-speech", json={
        "text": "Test.",
        "voice_id": "nonexistent_id_000000",
    })
    assert r.status_code == 400


# ── 8. WS /tts/ws ───────────────────────────────────────────────────────────────

def test_websocket_tts(model_type, first_speaker, voice_id):
    # Uses _resolve_voice() — same payload rules as /v1/text-to-speech
    payload = _voice_resolution_payload(model_type, first_speaker, voice_id, text="WebSocket TTS test.")
    payload["language"] = "English"

    async def _run():
        async with websockets.connect(f"{WS_BASE}/tts/ws") as ws:
            await ws.send(json.dumps(payload))

            raw = await asyncio.wait_for(ws.recv(), timeout=30)
            meta = json.loads(raw)
            assert meta["type"] == "metadata"
            assert meta["sample_rate"] == 24000
            assert meta["format"] == "pcm16"

            audio_bytes = 0
            while True:
                msg = await asyncio.wait_for(ws.recv(), timeout=120)
                if isinstance(msg, bytes):
                    audio_bytes += len(msg)
                else:
                    data = json.loads(msg)
                    if data["type"] == "complete":
                        assert data["chunks"] > 0
                        break
                    elif data["type"] == "error":
                        pytest.fail(f"Server error: {data['message']}")
            assert audio_bytes > 0
            assert audio_bytes % 2 == 0  # PCM16

    asyncio.run(_run())


def test_websocket_empty_text():
    async def _run():
        async with websockets.connect(f"{WS_BASE}/tts/ws") as ws:
            await ws.send(json.dumps({"text": ""}))
            msg = await asyncio.wait_for(ws.recv(), timeout=15)
            data = json.loads(msg)
            assert data["type"] == "error"

    asyncio.run(_run())


# ── 9. POST /upload_voice ────────────────────────────────────────────────────────

def test_upload_voice_file(client):
    if not TEST_VOICE_FILE or not os.path.exists(TEST_VOICE_FILE):
        pytest.skip("Set TEST_VOICE_FILE=path/to/ref.wav to run this test")

    with open(TEST_VOICE_FILE, "rb") as f:
        r = client.post("/upload_voice", files={"voice_file": ("ref.wav", f, "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert "voice_id" in body
    assert "created_at" in body


def test_upload_voice_no_input(client):
    r = client.post("/upload_voice", data={})
    assert r.status_code in (400, 422)
