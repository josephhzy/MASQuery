# PDF Extraction Limitations

This document records where the ingestion pipeline breaks down. MAS regulatory documents are table-heavy and footnote-heavy. The current extractor, PyMuPDF with a font-size heuristic for headers, handles the body text cleanly and mis-handles structured content. This page says what breaks, shows what it looks like, and lists the mitigation options.

## What the current pipeline does

File: `ingest.py`. The pipeline is:

1. Open the PDF with `fitz.open` (PyMuPDF).
2. For each page, extract plain text via `page.get_text("text")`.
3. Detect section headers by font size: spans whose size is >1.15× page-median, or bold with size ≥ median, become candidate headers.
4. Attempt table extraction via `page.find_tables()` and format each row as pipe-delimited text.
5. Emit a `PageData` object with `text`, `section_headers`, and `tables`.

The chunker then builds chunks from the page text, folds any detected tables into the section text (wrapped in `[Table start]` / `[Table end]` markers), and prepends the nearest detected section header.

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

`_extract_tables` does a separate table pass via `find_tables()`, which works for clean grid tables. The chunker **does** fold this structured table text into the chunk: `chunker._build_sections` wraps each `PageData.tables` entry in explicit `[Table start]` / `[Table end]` markers and appends it to the page's section text (see `chunker.py` and the regression test `tests/test_chunker.py::test_tables_are_consumed_into_chunks`). So the cleanly-extracted table **is** indexed. Two real limitations remain: (a) `find_tables()` still misses borderless / complex / multi-cell tables, so not every table gets the clean treatment; and (b) the garbled column-major copy from `get_text("text")` is **not** stripped, so it still lands in the same chunk alongside the clean folded version.

### 2. Footnotes merged into body text

MAS notices often use footnotes to carry the enforcement bite — the definition of a term, the exception to a rule, the cross-reference to another notice. PyMuPDF extracts footnote text in reading order, which typically places it at the end of the page's text stream. When the chunker later carves fixed-size chunks, a footnote reference number (`²`, `3`) at the end of a sentence in the body gets separated from the footnote definition at the end of the page — possibly landing in different chunks entirely.

Concretely: a chunk may contain "...must comply with paragraph 4.2.³" with no visible explanation of `³`, while the footnote text "³ *Paragraph 4.2 applies only to deposit-taking institutions.*" lives in a different chunk and loses its numeric anchor during whitespace normalisation.

### 3. Multi-column layout extracted in the wrong reading order

Some MAS PDFs have multi-column passages — particularly in annexes with tables beside commentary. `get_text("text")` doesn't know about columns; it emits text in the order it finds spans. The output mixes both columns and becomes semantically disordered.

### 4. Section header detection over-firing

The font-size heuristic (`size > 1.15 × median`) fires on any visually emphasised text. This over-collects: it flags bold in-line citations (e.g. a bolded "MAS Notice 644" inside a paragraph), tables-of-contents entries, and running headers. The chunker's "nearest header" logic can then prepend the wrong context to a chunk.

#### 4a. Intra-page section boundary loss in `_build_sections`

A related but distinct failure: when two real section headers are detected on the same page (one section ends and the next begins mid-page), `_build_sections` flushes the *previous* section and assigns the **entire page's content** to the last detected header (`section_headers[-1]`). Any body text that belongs under the first (closing) header — everything above the second header on that page — is silently promoted into the new section and tagged with the wrong heading.

Concretely: if page 12 ends Section 3 and begins Section 4, all text on page 12 is labelled "Section 4". A retrieval query anchored to Section 3 content on that page will miss it, or receive it under a misleading header.

**Constraint:** pages that span a section transition should ideally be split at the header boundary. Until that is implemented, treat any chunk whose source page contains multiple detected headers as having uncertain section attribution.

### 5. Section header detection under-firing

The inverse: MAS documents with stylistically uniform font sizes (all 10pt, headers slightly bolder rather than larger) can fool the heuristic. Chunks then lose their section anchor and inherit the last correctly-detected header, which may be several pages away.

## Measuring extraction quality

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

### Option D — Use the structured table field (folding: SHIPPED; dedup: pending)

Consuming `PageData.tables` in the chunker is **already done**: `chunker._build_sections` folds each detected table (wrapped in `[Table start]` / `[Table end]` markers) into the page's section text, so `find_tables()`'s row-major output is what reaches the index. Regression tests: `tests/test_chunker.py::test_tables_are_consumed_into_chunks` and `::test_table_only_page_still_produces_chunk`.

The remaining, **not-yet-shipped** refinement is to promote tables to *first-class* chunks: rather than inlining a table into the surrounding body chunk, emit it as its own chunk with a synthetic `section_header = "Table: <nearest heading>"`, and strip the garbled column-major duplicate from `PageData.text` so it no longer co-exists with the clean version. That is the zero-new-dependency next step.

### Option E — Footnote handling

Orthogonal to the table fix. Footnotes need their own treatment:

1. Detect superscript numeric spans in PyMuPDF output (position + font size cues).
2. At end-of-page, find the matching footnote block (small font, numeric prefix).
3. Inline the footnote text into the chunk that contains the reference, in parentheses, rather than leaving it detached.

This is a non-trivial heuristic pass. Prioritise only after the table fix has been shipped and measured.

## Recommended order

1. **Option D consumption — DONE.** `PageData.tables` is already folded into chunks (with `[Table start]` / `[Table end]` markers) and covered by regression tests. The outstanding piece is the refinement above: promote tables to first-class chunks and strip the duplicated column-major body text.
2. **Ship Option A or B (Camelot / pdfplumber) if D is insufficient.** Gate on the measurement: if tables are still under-retrieved, upgrade the extractor.
3. **Ship Option E (footnote inlining) last.** Only meaningful after the table work, because footnote references in mis-extracted tables are noise.
4. Re-run the ablation in `docs/RETRIEVAL_ABLATION.md` after each step to quantify the gain.

## What the limitation means today

The honest summary of "what happens when a MAS PDF has a complex table" is:

> When `find_tables()` detects the table, its cleanly-extracted (row-major) text **is** folded into the chunk and indexed, wrapped in `[Table start]` / `[Table end]` markers — so table-based questions can retrieve it. Two gaps remain: (1) `find_tables()` misses borderless and complex multi-cell tables, which then exist only as PyMuPDF's column-interleaved running text; and (2) even when a table is detected, the garbled column-major copy from `get_text("text")` is not stripped, so it co-exists with the clean version in the same chunk and dilutes the embedding. The next step is to promote detected tables to first-class chunks and strip the duplicate body text (the Option D refinement above).

That framing is more accurate than either "it works fine" or "tables are dropped".
