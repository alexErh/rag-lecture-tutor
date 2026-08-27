"""Zentraler Schalter: HuggingFace/Transformers in den Offline-Modus zwingen.

Steuert HF_HUB_OFFLINE und TRANSFORMERS_OFFLINE, sodass ALLE HF-basierten Modelle
im Prozess (docling-Layout/Formel, das Embedding-Modell in database.py und jedes
künftig hinzukommende HF-Modell, z. B. für weitere Chunking-Methoden) ausschließlich
aus dem lokalen Cache laden – keine Anfragen an den HF Hub, keine "unauthenticated
requests"-Warnung, kein Netz nötig.

Die Werte kommen aus der .env (Schlüssel HF_HUB_OFFLINE / TRANSFORMERS_OFFLINE).
Deshalb wird hier die .env ZUERST geladen – noch bevor huggingface_hub/transformers
importiert werden, denn diese lesen die Variablen EINMALIG beim Import in Konstanten.
Regel im Projekt: jedes Modul, das ein HF-Modell lädt (pdf_to_markdown, database),
und jeder Einstiegspunkt (api) importiert `hf_offline` als ALLERERSTES.

Reihenfolge/Priorität:
  1. echte Umgebungsvariable (falls gesetzt) – gewinnt immer
  2. Wert aus der .env
  3. Fallback hier: "1" (offline), damit ohne Konfiguration nichts ins Netz geht.

Um einmalig ein noch nicht gecachtes Modell herunterzuladen: in der .env
HF_HUB_OFFLINE=0 (und ggf. TRANSFORMERS_OFFLINE=0) setzen.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

# .env aus dem Projektwurzelverzeichnis (server/..) laden, unabhängig vom cwd.
# override=False -> eine echte Umgebungsvariable hat weiterhin Vorrang.
load_dotenv(Path(__file__).resolve().parent.parent / ".env", override=False)

# Fallback, falls in .env (und Umgebung) nicht gesetzt: standardmäßig offline.
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
