"""PDF-Dateien mit docling einlesen und nach Markdown umwandeln.

Liest PDFs aus  server/data/raw/  und speichert das Ergebnis als Markdown in
server/data/processed/. Diese Markdown-Dateien können anschließend von database.py
(add_document) gechunkt und in die Vektor-DB geschrieben werden.

Vorerst OHNE Bilder: die Standard-docling-Konvertierung extrahiert Text und
Struktur (Überschriften, Listen, Tabellen); Abbildungen werden nicht beschrieben.

Mathematische Formeln: mit aktivierter Formel-Anreicherung erkennt docling
Formel-Bereiche und wandelt sie in LaTeX um, das als $$...$$ ins Markdown
eingebettet wird. Standardmäßig eingeschaltet (enable_formulas=True) – für
Vorlesungsmaterial mit vielen Formeln sinnvoll.

Nutzung (aus dem server/-Ordner):
    python pdf_to_markdown.py            # alle PDFs aus data/raw konvertieren
"""

import os
import re
from pathlib import Path

# Vollständig lokaler Betrieb: die docling-Modelle (Layout, CodeFormulaV2) sind
# nach dem ersten Lauf im HuggingFace-Cache. Offline-Modus verhindert die Hub-
# Anfragen beim ersten Ingest ("unauthenticated requests to the HF Hub"-Warnung)
# und garantiert, dass kein Netz benötigt wird. setdefault -> per Umgebungsvariable
# überschreibbar, falls doch mal ein neues Modell geladen werden soll (HF_HUB_OFFLINE=0).
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

# Pfade relativ zu dieser Datei (server/), damit der Aufruf vom cwd unabhängig ist.
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ---- Formel-Bereinigung -----------------------------------------------------
# docling liest Formeln meist mathematisch korrekt, schleppt aber Layout-Artefakte
# mit: Nachbartext, der in die Formel blutet – oft als Fließtext Buchstabe für
# Buchstabe transkribiert ("Darstellung..." -> "D a r t o l l u n s p a r t i v o p"),
# dazu Gleichungsnummern aus Skripten und führende \cdot/\quad-Reste.
#
# Die Bereinigung ist bewusst HOCHPRÄZISE statt vollständig: sie fasst mehrzeilige
# Struktur-Formeln (Matrizen, aligned, cases, ...) NIE an und schneidet Bleed nur an
# eindeutigen Signalen ab. Lieber etwas Müll stehen lassen, als eine echte Formel
# beschädigen. Nur aktiv, wenn convert_pdf(clean_formulas=True) gesetzt ist.

_FORMULA_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)

# Struktur-Umgebungen: enthält die Formel eine davon, bleibt sie komplett unangetastet.
_MATH_ENV_RE = re.compile(
    r"\\begin\{(?:array|matrix|pmatrix|bmatrix|vmatrix|Vmatrix|smallmatrix"
    r"|aligned|align|alignat|cases|split|gathered|gather)\}"
)
# Gleichungsnummer ( N ) am Formelende – typisch für Skripte; danach folgt Nachbartext.
_EQ_NUM_RE = re.compile(r"\(\s*\d+\s*\)")
# Prosa-Signatur: >=5 aufeinanderfolgende, allein stehende Einzelbuchstaben. Echte
# Mathematik erzeugt das praktisch nie (max. 2-4, z. B. "Bild" -> B i l d), der
# Fließtext-Bleed schon ("jedes" -> j e d e s, "Darstellung..." -> D a r t o ...).
# Auf dem realen Material trennt die Schwelle 5 sauber: echte Inhalte bleiben <=4.
_PROSE_RUN_RE = re.compile(r"(?:(?<![A-Za-z0-9])[A-Za-z](?![A-Za-z0-9])\s+){5,}")
# Führende Spacing-/Operator-Reste (dangling \cdot am Formelanfang etc.).
_LEADING_ART_RE = re.compile(r"^(?:\s*(?:\\cdot|\\quad|\\,|\\;))+")


def _normalize_ws(s: str) -> str:
    """Restliche \\-Zeilenumbrüche zu Leerzeichen, Mehrfach-Whitespace zusammenfassen."""
    s = s.replace("\\\\", " ")
    return re.sub(r"\s+", " ", s).strip()


def _clean_formula_block(inner: str) -> str:
    s = inner.strip()

    # A) Mehrzeilige Struktur-Umgebungen (Matrizen, aligned, cases, ...) NIE anfassen –
    #    die Schnitte unten würden sie zerstören. Das macht die Bereinigung risikofrei
    #    für alle legitimen mehrzeiligen Formeln.
    if _MATH_ENV_RE.search(s):
        return s

    # B) Bleed am Ende abschneiden – zwei sichere Anker:
    #    1. Gleichungsnummer ( N ) (Skripte): alles danach ist Nachbartext.
    match = _EQ_NUM_RE.search(s)
    if match:
        s = s[: match.end()]
    #    2. Folien-Bleed: nur an einer \\-Naht schneiden, deren Folgesegment die
    #       Prosa-Signatur trägt. Ohne \\-Naht wird NICHT geschnitten – so kann ein
    #       legitimes Einzelbuchstaben-Produkt niemals fälschlich gekürzt werden.
    prose = _PROSE_RUN_RE.search(s)
    if prose:
        brk = s.rfind("\\\\", 0, prose.start())
        if brk > 0:
            head = s[:brk].strip()
            if head:
                s = head

    # C) Führende Spacing-/Operator-Reste entfernen.
    s = _LEADING_ART_RE.sub("", s)

    # D) \\ zu Leerzeichen (hier sicher – Struktur-Umgebungen oben ausgeschlossen)
    #    und Whitespace normalisieren.
    return _normalize_ws(s)


def _clean_formula_markdown(markdown: str) -> str:
    """Bereinigt alle $$...$$-Formelblöcke im Markdown (No-op ohne Formeln)."""
    return _FORMULA_RE.sub(lambda m: f"$${_clean_formula_block(m.group(1))}$$", markdown)


def _build_converter(
    enable_formulas: bool = True,
    enable_ocr: bool = False,
    layout_preset: str = "layout_egret_xlarge",
    formula_scale: float = 3.0,
    num_threads: int | None = None,
):
    """Erzeugt einen docling-DocumentConverter (Import lazy, mit klarer Fehlermeldung).

    Die Einstellungen zielen auf formelreiche, teils zweispaltige/Folien-PDFs mit
    echtem Text-Layer (verifiziert gegen docling 2.121.0):

    * layout_preset: präziseres Layout-Modell als der Default (Heron). "layout_egret_large"
        zieht engere Formel-Bounding-Boxen -> weniger Nachbartext-Bleed und weniger
        überflüssige \\begin{array}-Wrapper. Alternativen: "layout_egret_medium"
        (leichter/schneller) oder "layout_egret_xlarge" (genauer, langsamer).
    * formula_scale: Crop-Auflösung fürs Formel-Modell (CodeFormulaV2). Default 2.0;
        höher (2.5-3.0) verbessert kleine Symbole/Indizes. Betrifft NUR die
        Enrichment-Crops – nicht zu verwechseln mit pipeline images_scale.
    * enable_ocr: für digitale PDFs (echter Text-Layer) AUS lassen; OCR macht das
        Ergebnis hier schlechter und langsamer. Nur für gescannte PDFs einschalten.

    Args:
        enable_formulas: Formel-Anreicherung aktivieren (LaTeX als $$...$$).
            Rechenintensiv (VLM), auf CPU langsam.
        enable_ocr: OCR aktivieren – nur für gescannte PDFs sinnvoll.
        layout_preset: docling-Layout-Preset (siehe oben).
        formula_scale: Crop-Skalierung fürs Formel-Modell (siehe oben).
        num_threads: CPU-Threads (nur Tempo, keine Qualität). None = docling-Default
            (bzw. OMP_NUM_THREADS / DOCLING_NUM_THREADS aus der Umgebung).
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import (
            CodeFormulaVlmOptions,
            LayoutObjectDetectionOptions,
            PdfPipelineOptions,
        )
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise ImportError(
            "docling ist nicht installiert. Bitte ausführen:\n"
            "    pip install docling"
        ) from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_formula_enrichment = enable_formulas
    pipeline_options.do_ocr = enable_ocr
    # do_table_structure ist bereits True mit mode=ACCURATE (docling-Default).

    # Hebel 1: präziseres Layout-Modell -> engere Formel-Boxen, weniger Bleed/array.
    pipeline_options.layout_options = LayoutObjectDetectionOptions.from_preset(layout_preset)

    # Hebel 2: höhere Crop-Auflösung fürs Formel-Modell -> Symbole/Indizes.
    if enable_formulas:
        cf = CodeFormulaVlmOptions.from_preset("codeformulav2")
        cf.scale = formula_scale
        pipeline_options.code_formula_options = cf

    # CPU: Threads setzen (nur Tempo). None -> docling entscheidet.
    if num_threads is not None:
        from docling.datamodel.accelerator_options import (
            AcceleratorDevice,
            AcceleratorOptions,
        )

        pipeline_options.accelerator_options = AcceleratorOptions(
            num_threads=num_threads, device=AcceleratorDevice.CPU
        )

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


# Seiten-Marker: docling verwirft im flachen Markdown die Seiteninformation. Wir
# bauen das Markdown deshalb SEITENWEISE auf und stellen jeder Seite einen HTML-
# Kommentar mit der echten Seitenzahl voran. chunking.py liest diese Marker wieder
# aus, um jedem Chunk seine Seite zuzuordnen – das Format muss dort übereinstimmen.
PAGE_MARKER = "<!-- page: {} -->"


def _export_markdown_with_pages(document) -> str:
    """Exportiert ein DoclingDocument nach Markdown, jede Seite mit Seiten-Marker.

    Fällt auf den normalen Export zurück, falls das Dokument keine Seiten meldet.
    """
    pages = sorted(document.pages)
    if not pages:
        return document.export_to_markdown()
    parts = [
        f"{PAGE_MARKER.format(page_no)}\n\n{document.export_to_markdown(page_no=page_no)}"
        for page_no in pages
    ]
    return "\n\n".join(parts)


def convert_pdf(
    pdf_path: str,
    converter=None,
    overwrite: bool = False,
    enable_formulas: bool = True,
    clean_formulas: bool = True,
) -> Path:
    """Konvertiert eine einzelne PDF nach Markdown.

    Args:
        pdf_path: Pfad zur PDF-Datei.
        converter: optional ein bereits erzeugter docling-Converter
            (spart bei mehreren Dateien die wiederholte Initialisierung).
        overwrite: vorhandene Markdown-Datei überschreiben.
        enable_formulas: Formeln als LaTeX einbetten (nur relevant, wenn kein
            eigener converter übergeben wird).
        clean_formulas: nachträgliche Regex-Bereinigung der $$...$$-Blöcke.
            Standard False – die verbesserten docling-Einstellungen (Egret-Layout,
            höhere Crop-Auflösung) sollen die Artefakte an der Wurzel reduzieren.
            Zum A/B-Vergleich auf True setzen.

    Returns:
        Pfad zur erzeugten Markdown-Datei.
    """
    pdf_path = Path(pdf_path)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    out_path = PROCESSED_DIR / f"{pdf_path.stem}.md"

    # Bereits konvertierte Dateien nicht erneut verarbeiten.
    if out_path.exists() and not overwrite:
        print(f"übersprungen (existiert): {pdf_path.name}")
        return out_path

    converter = converter or _build_converter(enable_formulas=enable_formulas)
    result = converter.convert(str(pdf_path))
    markdown = _export_markdown_with_pages(result.document)  # inkl. Seiten-Marker
    if clean_formulas:
        markdown = _clean_formula_markdown(markdown)  # Formel-Artefakte glätten (optional)
    out_path.write_text(markdown, encoding="utf-8")

    print(f"{pdf_path.name} -> {out_path.name} ({len(markdown)} Zeichen)")
    return out_path


def convert_all(
    overwrite: bool = False,
    enable_formulas: bool = True,
    clean_formulas: bool = False,
) -> list[Path]:
    """Konvertiert alle PDFs aus server/data/raw/ nach Markdown."""
    if not RAW_DIR.exists():
        raise FileNotFoundError(f"Eingabeordner nicht gefunden: {RAW_DIR}")

    pdf_files = sorted(RAW_DIR.rglob("*.pdf"))
    if not pdf_files:
        print(f"Keine PDF-Dateien in {RAW_DIR} gefunden.")
        return []

    # docling-Converter nur einmal aufbauen und wiederverwenden.
    converter = _build_converter(enable_formulas=enable_formulas)

    outputs = []
    for i, pdf_path in enumerate(pdf_files, start=1):
        print(f"[{i}/{len(pdf_files)}] {pdf_path.name}")
        try:
            outputs.append(
                convert_pdf(
                    pdf_path,
                    converter=converter,
                    overwrite=overwrite,
                    clean_formulas=clean_formulas,
                )
            )
        except Exception as exc:
            print(f"    [FEHLER] {pdf_path.name}: {exc}")
    return outputs


if __name__ == "__main__":
    results = convert_all()
    print(f"\nFertig: {len(results)} Markdown-Datei(en) in {PROCESSED_DIR}")
