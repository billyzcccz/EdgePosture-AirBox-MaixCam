import base64
import json
import os
import time
import requests


def text_to_tts_payload(text: str) -> dict:
    """Encode UTF-8 text into GB2312-Base64 JSON payload for the LAN IP speaker."""
    gb_bytes = text.encode("gb2312", errors="ignore")
    b64_str = base64.b64encode(gb_bytes).decode("ascii")
    return {"tts": {"data": b64_str}}


def send_tts(
    text: str,
    host: str = None,
    port: int = None,
    delay_seconds: float = 0,
):
    """Send posture feedback text to the LAN IP speaker for TTS playback."""
    host = host or os.environ.get("IP_SPEAKER_HOST", "192.168.3.29")
    port = int(port or os.environ.get("IP_SPEAKER_PORT", "80"))

    payload_dict = text_to_tts_payload(text)
    payload_str = json.dumps(payload_dict, ensure_ascii=False)
    payload_bytes = payload_str.encode("gb2312", errors="ignore")
    url = f"http://{host}:{port}/v3"
    headers = {"Content-Type": "application/json"}

    if delay_seconds > 0:
        time.sleep(delay_seconds)

    try:
        resp = requests.post(url, headers=headers, data=payload_bytes, timeout=10)
        resp.raise_for_status()
        return resp.json()
    except requests.RequestException as e:
        print(f"IP speaker request failed ({url}): {e}")
        return None
