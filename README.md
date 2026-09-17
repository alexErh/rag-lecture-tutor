# RAG-Vorlesungstutor

Ein Retrieval-Augmented-Generation-System für Vorlesungsmaterialien: PDFs werden nach
Markdown konvertiert, in Chunks zerlegt, in einer Vektordatenbank abgelegt und über einen
LLM-Agenten mit Quellenangabe beantwortet. Das Projekt vergleicht dabei vier
Chunking-Strategien (recursive, markdown, semantic, late).

## Setup

```bash
python -m venv .venv
.venv/Scripts/activate          # Windows:  .venv\Scripts\activate
pip install -r requirements.txt
```

Konfiguration anlegen (Vorlage kopieren und Werte anpassen):

```bash
cp .env.example .env            # Windows (cmd):  copy .env.example .env
```

## Umgebungsvariablen (.env)

| Variable | Standard | Bedeutung |
|---|---|---|
| `LOCAL` | `true` | `true` = lokales Ollama-Modell, `false` = Nvidia-NIM-Cloud (dann `NVIDIA_API_KEY` nötig) |
| `NVIDIA_API_KEY` | – | API-Schlüssel für Nvidia NIM (nur bei `LOCAL=false`). Schlüssel: https://build.nvidia.com/ |
| `LOCAL_BASE_URL` | `http://localhost:11434` | Adresse des Ollama-Servers (bei `LOCAL=true`) |
| `LOCAL_MODEL_NAME` | `llama3.2:3b` | Name des Ollama-Modells (muss vorher `ollama pull …` geladen sein) |
| `OLLAMA_WARMUP` | `true` | Ollama-Modell beim API-Start vorladen (schnellere erste Antwort) |
| `HF_HUB_OFFLINE` | `1` | HuggingFace offline – Modelle nur aus dem lokalen Cache. `0` nur, um ein noch nicht gecachtes Modell einmalig zu laden |
| `TRANSFORMERS_OFFLINE` | `1` | dito für die Transformers-Bibliothek |
| `EMBEDDING_MODEL` | `intfloat/multilingual-e5-base` | Gemeinsames Embedding-Modell für **alle** Chunking-Methoden (mehrsprachig, Langkontext) |
| `LATE_EMBEDDING_MODEL` | `= EMBEDDING_MODEL` | Modell speziell für Late Chunking (überschreibbar) |
| `LATE_CHUNK_SIZE` | `512` | Chunk-Größe (Tokens) für Late Chunking |
| `DYNAMIC_RETREIVAL` | `false` | `true` = Retrieval per Distanz-Schwelle statt fester Top-n |
| `DEBUG` | `false` | `true` = ausführliche Retrieval-Logs in der Konsole |

Hinweis: Das gewählte Embedding-Modell muss beim ersten Einsatz einmalig heruntergeladen
werden – dazu kurz `HF_HUB_OFFLINE=0` setzen, danach wieder auf `1`.

## Bedienung

**Alle Befehle aus dem Ordner `server/` ausführen**, mit dem Projekt-Root in `PYTHONPATH`
(der Code mischt Skript- und Paket-Importe). Einmalig pro Terminal-Sitzung setzen:

```bash
cd server
export PYTHONPATH=..             # PowerShell:  $env:PYTHONPATH=".."
```

### 1. PDFs bereitstellen
PDF-Dateien nach `server/data/raw/` legen.

### 2. PDF → Markdown
```bash
python pdf_to_markdown.py
```
Erzeugt Markdown (inkl. Seiten-Marker und LaTeX-Formeln) in `server/data/processed/`.
Hinweis: Läuft auf CPU und ist bei formelreichen PDFs langsam.

### 3. In die Vektordatenbank laden
```bash
python database.py
```
Chunkt alle Markdown-Dateien mit **allen vier Methoden** und speichert sie in ChromaDB
(`server/VectorDB/`). Erneuter Aufruf ersetzt nur die Chunks derselben Datei + Methode.

### 4. API starten
```bash
uvicorn api:app --reload
```
Läuft auf `http://localhost:8000`.

### 5. Fragen stellen
Am einfachsten über das Notebook `main.ipynb` (siehe nächster Abschnitt) oder direkt über
die Endpunkte:
- `POST /ask` – Frage an den Agenten (Chat-Verlauf + Chunking-Methode)
- `POST /query` – direkte Ähnlichkeitssuche in der DB
- `POST /ingest` – einzelne PDF hochladen, konvertieren und indexieren
- `POST /document` – Inhalt eines Dokuments per Dateiname abrufen

## Bedienung über das Jupyter-Notebook (`main.ipynb`)

Das Notebook `main.ipynb` (im Projekt-Root) ist ein dünner Client: Agent, Retrieval und
LLM laufen im Server, das Notebook lädt nur PDFs hoch und stellt Fragen – alles über HTTP.

**Voraussetzung:** Die API muss laufen (siehe Schritt 4: `uvicorn api:app --reload` bzw.
`python api.py` aus `server/`).

Dann die Zellen der Reihe nach ausführen:

1. **Setup-Zelle:** Setzt `API_URL` (Standard `http://127.0.0.1:8000`) und die Variable
   `METHOD` – hier die gewünschte Chunking-Methode wählen (`RECURSIVE`, `MARKDOWN`,
   `SEMANTIC` oder `LATE`). Sie gilt für Upload **und** Fragen.
2. **PDFs hochladen (`/ingest`):** Legt die PDFs aus dem Ordner `input_pdfs/` (neben dem
   Notebook) beim Backend ab; dort werden sie konvertiert, gechunkt und indexiert. Der
   erste Upload lädt einmalig die docling-Modelle → das kann dauern. *(Alternativ kann die
   DB auch vorab per `python database.py` befüllt werden – dann ist dieser Schritt
   optional.)*
3. **Konversation starten:** Die Zelle mit `messages = []` einmal ausführen (setzt den
   Chat-Verlauf zurück).
4. **Frage stellen:** Die letzte Zelle ausführen – sie fragt die Eingabe ab, schickt sie an
   `/ask` und zeigt die Antwort. Für **Folgefragen** einfach diese Zelle erneut ausführen;
   der Verlauf in `messages` bleibt erhalten.

Zum Wechseln der Chunking-Methode `METHOD` in der Setup-Zelle ändern und die Setup-Zelle
erneut ausführen.

## Evaluation (optional)

```bash
python evaluation.py             # vergleicht die vier Methoden (Precision/Recall/IoU/F1 + Coverage)
python threshold_calibration.py  # bestimmt die faire Retrieval-Schwelle für den dynamischen Modus
```
