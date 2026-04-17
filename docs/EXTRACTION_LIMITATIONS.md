# PDF Extraction Limitations

This document is the honest accounting of where the ingestion pipeline breaks down. MAS regulatory documents are table-heavy and footnote-heavy. The current extractor, PyMuPDF with a font-size heuristic for headers, handles the body text cleanly and mis-handles structured content. This page says what breaks, shows what it looks like, and lists the mitigation options.

## What the current pipeline does

File: `ingest.py`. The pipeline is:

1. Open the PDF with `fitz.open` (PyMuPDF).
2. For each page, extract plain text via `page.get_text("text")`.
3. Detect section headers by font size: spans whose size is >1.15× page-median, or bold with size ≥ median, become candidate headers.
4. Attempt table extraction via `page.find_tables()` and format each row as pipe-delimited text.
5. Emit a `PageData` object with `text`, `section_headers`, and `tables`.

The chunker then builds chunks from the page text, prepending the nearest detected section header.

## Where it breaks

### 1. Tables mis-extracted as running text

PyMuPDF's `get_text("text")` flattens a table into whatever reading order the PDF's internal layout dictates. For a two-column table, this often means:

```
Header1
Header2
Cell1-1
Cell1-2
Cell2-1
Cell2-2
```

i.e. column-major interleaved instead of row-major. When this lands in a chunk, it reads as gibberish and embeds poorly. The retrieval score on questions that would logically hit this table is degraded, even though the content is technically present.

`_extract_tables` does try a separate table pass via `find_tables()`, and this works for clean grid tables. But it's stored in `PageData.tables` as a separate list — **the current chunker does not incorporate the structured `tables` field into chunks.** It only uses `text`. The well-extracted table sits unused while the badly-extracted version lands in the index.

### 2. Footnotes merged into body text

MAS notices often use footnotes to carry the enforcement bite — the definition of a term, the exception to a rule, the cross-reference to another notice. PyMuPDF extracts footnote text in reading order, which typically places it at the end of the page's text stream. When the chunker later carves fixed-size chunks, a footnote reference number (`²`, `3`) at the end of a sentence in the body gets separated from the footnote definition at the end of the page — possibly landing in different chunks entirely.

Concretely: a chunk may contain "...must comply with paragraph 4.2.³" with no visible explanation of `³`, while the footnote text "³ *Paragraph 4.2 applies only to deposit-taking institutions.*" lives in a different chunk and loses its numeric anchor during whitespace normalisation.

### 3. Multi-column layout on scanned pages

Some MAS PDFs have multi-column passages — particularly in annexes with tables beside commentary. `get_text("text")` doesn't know about columns; it emits text in the order it finds spans. The output mixes both columns and becomes semantically disordered.

### 4. Section header detection over-firing

The font-size heuristic (`size > 1.15 × median`) fires on any visually emphasised text. This over-collects: it flags bold in-line citations (e.g. a bolded "MAS Notice 644" inside a paragraph), tables-of-contents entries, and running headers. The chunker's "nearest header" logic can then prepend the wrong context to a chunk.

### 5. Section header detection under-firing

The inverse: MAS documents with stylistically uniform font sizes (all 10pt, headers slightly bolder rather than larger) can fool the heuristic. Chunks then lose their section anchor and inherit the last correctly-detected header, which may be several pages away.

## How to measure how badly this matters

Currently: **it is not measured.** That is the first fix. A measurement pass would:

1. Manually annotate a ~30-page sample of the corpus, page-by-page, labelling each page as `{clean | table_present | footnote_present | multi_column}`.
2. For each label, compute the "garbled rate": fraction of characters in the extracted text that do not appear in the same order as a human would read the page.
3. Run the pipeline over the five MAS PDFs, count chunks per category, and publish: "X% of chunks derive from table-heavy pages, Y% from footnote-heavy pages".

Output: `docs/EXTRACTION_QUALITY.md` with the per-category garbled rate and the top-10 worst-extracted chunks (by a heuristic — e.g. presence of isolated single-word lines) for eyeball review.

## Mitigation options

### Option A — Camelot for tables

Add a table extraction pass using [Camelot](https://camelot-py.readthedocs.io/). Camelot uses a different strategy (PDF vector primitives or image-based lattice detection) and produces structured DataFrames instead of flattened text. Integration points:

- Run Camelot per page before PyMuPDF's text pass.
- When Camelot detects a table, strip the table's bounding-box region from the PyMuPDF text output before it goes into `PageData.text`.
- Emit the Camelot table as a first-class chunk: its own chunk with a synthetic `section_header = "Table: <nearest heading>"` and a serialized text representation (Markdown table or row-per-line).

**Cost:** Camelot is heavier than PyMuPDF. Adds a Java-free but Ghostscript-dependent stack or requires OpenCV. Slower per page.

**Benefit:** Tables become retrievable. A question like "what notification timeline applies to material outsourcing" that maps to a table row actually hits the table chunk instead of a garbled body-text chunk.

### Option B — pdfplumber

[pdfplumber](https://github.com/jsvine/pdfplumber) is lighter than Camelot and has decent table support. Same integration pattern. Better for simple grid tables; worse for complex multi-cell layouts.

### Option C — Docling or Unstructured

Modern higher-level libraries ([Docling](https://github.com/DS4SD/docling), [Unstructured.io](https://unstructured.io/)) do layout-aware extraction: they return blocks tagged `text`, `title`, `table`, `list`, etc. These are closer to what MASquery needs but introduce a much larger dependency tree and non-trivial warmup cost.

### Option D — Keep PyMuPDF but use the structured table field

The simplest change, no new dependency: modify `chunker.py` to consume `PageData.tables` alongside `PageData.text`. Emit each detected table as its own chunk with a distinctive `section_header`. This alone would recover a lot of content currently lost to column-major flattening, because `page.find_tables().extract()` already returns row-major data.

This is the recommended first step — it ships value with zero new dependencies and measures how much of the pain is solvable without a heavier toolchain.

### Option E — Footnote handling

Orthogonal to the table fix. Footnotes need their own treatment:

1. Detect superscript numeric spans in PyMuPDF output (position + font size cues).
2. At end-of-page, find the matching footnote block (small font, numeric prefix).
3. Inline the footnote text into the chunk that contains the reference, in parentheses, rather than leaving it detached.

This is a non-trivial heuristic pass. Prioritise only after the table fix has been shipped and measured.

## Recommended order

1. **Ship Option D.** Consume the existing `PageData.tables` in chunking. One-file change in `chunker.py`. Measure the delta.
2. **Ship Option A or B (Camelot / pdfplumber) if D is insufficient.** Gate on the measurement: if tables are still under-retrieved, upgrade the extractor.
3. **Ship Option E (footnote inlining) last.** Only meaningful after (1) because footnote references in mis-extracted tables are noise.
4. Re-run the ablation in `docs/RETRIEVAL_ABLATION.md` after each step to quantify the gain.

## What the limitation means today

Until one of the above ships, the honest summary of "what happens when a MAS PDF has a complex table" is:

> The table is technically extracted into a separate structured field by PyMuPDF's `find_tables()`, but the chunker currently only consumes the flat-text field, which mis-orders table cells into running text. The chunk is still indexed, but its embedding quality is degraded and retrieval for table-based questions is unreliable. The fix is a one-file change to consume the already-extracted structured tables — scoped in this document.

That framing is more accurate than "it works fine".
