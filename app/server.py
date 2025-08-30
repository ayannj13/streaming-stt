import asyncio, base64, json, time, collections, traceback
from pathlib import Path

import numpy as np
import webrtcvad
from fastapi import FastAPI, WebSocket
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from vosk import Model, KaldiRecognizer
from starlette.websockets import WebSocketDisconnect

SAMPLE_RATE = 16000
FRAME_MS = 20
FRAME_SAMPLES = int(SAMPLE_RATE * FRAME_MS / 1000)
VAD_AGGRESSIVENESS = 2
END_SIL_MS = 1200                    
MODEL_DIR = "models/vosk-en"
LOG_DIR = Path("logs")

app = FastAPI()
app.mount("/static", StaticFiles(directory="app/static"), name="static")

@app.get("/")
def root():
    return FileResponse("app/static/index.html")

LOG_DIR.mkdir(exist_ok=True)
session_file = LOG_DIR / f"session_{int(time.time()*1000)}.jsonl"

def log_event(event: dict):
    event["t"] = time.time()
    with open(session_file, "a") as f:
        f.write(json.dumps(event) + "\n")

MODEL = Model(MODEL_DIR)

def pcm16_from_b64(b64str: str) -> np.ndarray:
    raw = base64.b64decode(b64str)
    return np.frombuffer(raw, dtype=np.int16)

def chunk_bytes(int16arr: np.ndarray):
    for i in range(0, len(int16arr), FRAME_SAMPLES):
        frame = int16arr[i:i+FRAME_SAMPLES]
        if len(frame) == FRAME_SAMPLES:
            yield frame.tobytes()

@app.websocket("/ws")
async def ws_asr(ws: WebSocket):
    await ws.accept()
    print("WS: client connected")
    log_event({"event": "session_start"})

    vad = webrtcvad.Vad(VAD_AGGRESSIVENESS)
    rec = KaldiRecognizer(MODEL, SAMPLE_RATE)
    rec.SetWords(True)

    silence_ms = 0
    last_partial = ""
    recent_partials = collections.deque(maxlen=3)

    first_audio_ts = None
    first_partial_ts = None
    utter_start_ts = None  

    try:
        while True:
            try:
                msg = await ws.receive_text()
            except WebSocketDisconnect as e:
                print("WS: client closed normally", e.code)
                break
            except Exception as e:
                print("WS: receive error ->", repr(e))
                break

            try:
                data = json.loads(msg)
            except Exception as e:
                print("WS: bad json ->", msg[:120], repr(e))
                continue

            if data.get("type") != "audio":
                continue

            if first_audio_ts is None:
                first_audio_ts = time.time()
                log_event({"event": "first_audio"})

            pcm = pcm16_from_b64(data["pcm16le"])
            for fb in chunk_bytes(pcm):
                voiced = vad.is_speech(fb, SAMPLE_RATE)
                if voiced:
                    silence_ms = 0
                else:
                    silence_ms += FRAME_MS

                try:
                    has_final = rec.AcceptWaveform(fb)
                except Exception as e:
                    print("WS: recognizer error ->", repr(e))
                    print(traceback.format_exc())
                    await ws.close()
                    return

                if has_final:
                    res = json.loads(rec.Result())
                    final_text = (res.get("text") or "").strip()
                    if final_text:
                        await ws.send_text(json.dumps({
                            "type": "final",
                            "text": final_text,
                            "ts": time.time()
                        }))
                        log_event({"event": "final", "text": final_text})

                        if utter_start_ts is not None:
                            log_event({
                                "event": "time_to_final",
                                "time_to_final_ms": int((time.time() - utter_start_ts) * 1000)
                            })

                        last_partial = ""
                        recent_partials.clear()
                        utter_start_ts = None
                        silence_ms = 0
                else:
                    part = json.loads(rec.PartialResult()).get("partial", "").strip()
                    if part and part != last_partial:
                        if first_partial_ts is None and first_audio_ts is not None:
                            first_partial_ts = time.time()
                            log_event({
                                "event": "first_partial",
                                "latency_ms": int((first_partial_ts - first_audio_ts) * 1000)
                            })
                        if utter_start_ts is None:
                            utter_start_ts = time.time()
                        last_partial = part
                        recent_partials.append(part)
                        await ws.send_text(json.dumps({
                            "type": "partial",
                            "text": part,
                            "ts": time.time()
                        }))
                        log_event({"event": "partial", "text": part})

                if utter_start_ts and silence_ms >= END_SIL_MS and len(set(recent_partials)) <= 1:
                    endpoint_ts = time.time()
                    res = json.loads(rec.FinalResult())
                    final_text = (res.get("text") or "").strip()
                    if final_text:
                        await ws.send_text(json.dumps({
                            "type": "final",
                            "text": final_text,
                            "ts": time.time()
                        }))
                        log_event({"event": "final", "text": final_text})
                        log_event({
                            "event": "time_to_final",
                            "time_to_final_ms": int((time.time() - utter_start_ts) * 1000)
                        })
                    last_partial = ""
                    recent_partials.clear()
                    utter_start_ts = None
                    silence_ms = 0

    except Exception as e:
        print("WS: fatal ->", repr(e))
        print(traceback.format_exc())
    finally:
        print("WS: client disconnected")
        try:
            await ws.close()
        except Exception:
            pass
