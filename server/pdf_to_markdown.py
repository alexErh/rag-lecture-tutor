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

import re
from pathlib import Path

# Pfade relativ zu dieser Datei (server/), damit der Aufruf vom cwd unabhängig ist.
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"


# ---- Formel-Bereinigung -----------------------------------------------------
# docling erkennt Formeln im (oft zweispaltigen) Layout meist mathematisch korrekt,
# schleppt aber Artefakte mit: führende \cdot/\quad, angehängte Textfragmente aus
# der Nachbarspalte (\\ \text{...}), überflüssige \begin{array}-Wrapper und in die
# Formel geratene Gleichungsnummern. Diese Heuristik glättet die häufigsten Fälle.

_FORMULA_RE = re.compile(r"\$\$(.*?)\$\$", re.DOTALL)


def _clean_formula_block(inner: str) -> str:
    s = inner

    # 1) Überflüssigen \begin{array}{...} ... \end{array}-Wrapper entpacken.
    if "\\begin{array}" in s:
        s = re.sub(r"\\begin\{array\}\s*\{[^}]*\}", " ", s)
        s = s.replace("\\end{array}", " ").replace("&", " ")

    # 2) Layout-Bleed am Ende abschneiden: alles nach der Gleichungsnummer ( N ).
    match = re.search(r"\(\s*\d+\s*\)", s)
    if match:
        s = s[: match.end()]

    # 3) Führende Artefakte entfernen (\cdot, \quad, kurze \text{..}-Fragmente).
    s = re.sub(r"^(?:\s*(?:\\cdot|\\quad|\\,|\\;|\\text\s*\{[^}]{0,4}\}))+", "", s)

    # 4) Verbliebene Zeilenumbrüche (\\) zu Leerzeichen, Whitespace normalisieren.
    s = s.replace("\\\\", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _clean_formula_markdown(markdown: str) -> str:
    """Bereinigt alle $$...$$-Formelblöcke im Markdown (No-op ohne Formeln)."""
    return _FORMULA_RE.sub(lambda m: f"$${_clean_formula_block(m.group(1))}$$", markdown)


def _build_converter(enable_formulas: bool = True, enable_ocr: bool = True):
    """Erzeugt einen docling-DocumentConverter (Import lazy, mit klarer Fehlermeldung).

    Args:
        enable_formulas: Formel-Anreicherung aktivieren – erkennt mathematische
            Formeln und wandelt sie in LaTeX um (im Markdown als $$...$$).
            Rechenintensiv (Transformer-Modell), auf CPU langsam.
        enable_ocr: OCR aktivieren – nötig für gescannte PDFs, für digitale
            (mit echtem Text-Layer) überflüssig und langsam.
    """
    try:
        from docling.datamodel.base_models import InputFormat
        from docling.datamodel.pipeline_options import PdfPipelineOptions
        from docling.document_converter import DocumentConverter, PdfFormatOption
    except ImportError as exc:
        raise ImportError(
            "docling ist nicht installiert. Bitte ausführen:\n"
            "    pip install docling"
        ) from exc

    pipeline_options = PdfPipelineOptions()
    pipeline_options.do_formula_enrichment = enable_formulas
    pipeline_options.do_ocr = enable_ocr

    return DocumentConverter(
        format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
    )


def convert_pdf(
    pdf_path: str,
    converter=None,
    overwrite: bool = False,
    enable_formulas: bool = True,
) -> Path:
    """Konvertiert eine einzelne PDF nach Markdown.

    Args:
        pdf_path: Pfad zur PDF-Datei.
        converter: optional ein bereits erzeugter docling-Converter
            (spart bei mehreren Dateien die wiederholte Initialisierung).
        overwrite: vorhandene Markdown-Datei überschreiben.
        enable_formulas: Formeln als LaTeX einbetten (nur relevant, wenn kein
            eigener converter übergeben wird).

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
    markdown = result.document.export_to_markdown()
    markdown = _clean_formula_markdown(markdown)  # Formel-Artefakte glätten
    out_path.write_text(markdown, encoding="utf-8")

    print(f"{pdf_path.name} -> {out_path.name} ({len(markdown)} Zeichen)")
    return out_path


def convert_all(overwrite: bool = False, enable_formulas: bool = True) -> list[Path]:
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
            outputs.append(convert_pdf(pdf_path, converter=converter, overwrite=overwrite))
        except Exception as exc:
            print(f"    [FEHLER] {pdf_path.name}: {exc}")
    return outputs


if __name__ == "__main__":
    results = convert_all()
    print(f"\nFertig: {len(results)} Markdown-Datei(en) in {PROCESSED_DIR}")
