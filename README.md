# Streaming STT Demo with FastAPI + WebSocket + Vosk

A minimal **real-time Speech-to-Text** system:
- Browser captures mic audio and streams to server over **WebSocket**
- Server chunks audio (20 ms), runs **VAD (webrtcvad)** to handle real-time speech, feeds frames into **Vosk** recognizer
- Sends **partials** while you speak and **finals** after short pauses

## Quick Start to test the system

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt

**download a small English Vosk model and place the unzipped folder inside the "models" folder: models/vosk-en/  (contains am/, graph/, conf/, etc.)**

**Then run the server:**

uvicorn app.server:app --reload --host 0.0.0.0 --port 8000

Open http://localhost:8000 
Start the mic and speak. When you're done press the 'Stop' button.

## Project Structure
app/
  server.py          # FastAPI + WebSocket + VAD + Vosk
  static/
    index.html       # Mic capture + streaming client UI
models/
  vosk-en/           # (downloaded model, it's not in repo)
logs/                # JSONL session logs 

## System Design 

Client (browser): Web Audio API + WebSocket.

Transport: 16 kHz PCM16 chunks (~20 ms frames) - base64 JSON messages.

Server: FastAPI WebSocket endpoint /ws
webrtcvad for silence detection
Vosk streaming recognizer for partial and final hypotheses

## Real-time behavior:

**Partials** appear while speaking (gray).

**Finals** appear on short pause (bold).

**Metrics (logged in logs/*.jsonl):**

first_partial.latency_ms – time from first audio : first partial text

time_to_final.time_to_final_ms – time from utterance start : final text


## How to evaluate

**Latency**: Grep the latest session log:

ls -lt logs | head
tail -n +1 logs/<LATEST>.jsonl | grep -E 'first_partial|time_to_final' 

** replace the <LATEST> with the session and its id together eg.(session_1756567347137.jsonl)**

For evaluating **WER**: Put what you wanted to say as a text in refs.json, and what is recognized by the system in hyps.json, and in the terminal run:

python eval_wer.py


**Example results (This is what the final results will look like):**

First partial latency: ~1523 ms

Time to final: ~927 ms

WER: 0.0

**Notes**

- Tunables: VAD_AGGRESSIVENESS, END_SIL_MS (latency, stability)
- Consider AudioWorklet on the client for lower callback latency