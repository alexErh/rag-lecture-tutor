"""Ollama-Lebenszyklus: lokales Modell beim API-Start laden, beim Stop entladen.

Nur relevant im lokalen Modus (LOCAL=True). Datei absichtlich NICHT 'ollama.py'
genannt, um das gleichnamige PyPI-Paket nicht zu überschatten.
"""

import os

from dotenv import load_dotenv

# .env laden (LOCAL, LOCAL_BASE_URL, LOCAL_MODEL_NAME)
load_dotenv()

# LOCAL=true -> lokales Ollama-Modell wird genutzt (dann Warmup/Unload sinnvoll).
LOCAL = os.getenv("LOCAL", "true").strip().lower() in ("1", "true", "yes", "ja")


def _ollama_keep_alive(keep_alive, timeout: int = 300) -> str:
    """Sendet eine Leer-Anfrage an Ollama, um das Modell zu laden bzw. zu entladen.

    keep_alive=-1 lädt das Modell und hält es im Speicher; keep_alive=0 entlädt es
    sofort. Gibt den Modellnamen zurück.
    """
    import json
    import urllib.request

    base_url = os.getenv("LOCAL_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("LOCAL_MODEL_NAME", "llama3.2:3b")
    payload = json.dumps(
        {"model": model_name, "prompt": "", "keep_alive": keep_alive}
    ).encode("utf-8")
    request = urllib.request.Request(
        f"{base_url}/api/generate",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as resp:
        resp.read()
    return model_name


def warmup_ollama() -> None:
    """Lädt das lokale Ollama-Modell beim Start in den Speicher (keep_alive=-1).

    So ist das Modell schon vor der ersten /ask-Anfrage bereit. Nur relevant im
    lokalen Modus (LOCAL=True); scheitert leise, falls Ollama (noch) nicht läuft.
    """
    if not LOCAL:
        return
    try:
        print("[startup] Lade Ollama-Modell vor ...")
        model_name = _ollama_keep_alive(-1, timeout=300)
        print(f"[startup] Ollama-Modell '{model_name}' geladen und bereit.")
    except Exception as exc:
        print(f"[startup] Warmup übersprungen (läuft Ollama?): {exc}")


def unload_ollama() -> None:
    """Entlädt das lokale Ollama-Modell beim Herunterfahren aus dem Speicher.

    Gibt den RAM wieder frei (keep_alive=0). Nur im lokalen Modus; scheitert leise.
    """
    if not LOCAL:
        return
    try:
        print("[shutdown] Entlade Ollama-Modell ...")
        model_name = _ollama_keep_alive(0, timeout=30)
        print(f"[shutdown] Ollama-Modell '{model_name}' entladen.")
    except Exception as exc:
        print(f"[shutdown] Entladen übersprungen: {exc}")
