"""
Download publicly available MAS regulatory documents for the RAG pipeline.
All documents are freely available from mas.gov.sg.
"""

import sys
from pathlib import Path

import httpx

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RAW_DIR

# MAS's CDN blocks requests without a browser-like User-Agent header; a plain
# Python/httpx string returns 403. The UA below mimics a current desktop browser
# to reliably fetch the publicly available PDFs.
_BROWSER_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/125.0.0.0 Safari/537.36"
)

# MAS documents — direct download URLs
# These are publicly available regulatory guidelines
MAS_DOCUMENTS = [
    {
        "name": "Technology Risk Management Guidelines (Jan 2021)",
        "url": "https://www.mas.gov.sg/-/media/MAS/Regulations-and-Financial-Stability/Regulatory-and-Supervisory-Framework/Risk-Management/TRM-Guidelines-18-January-2021.pdf",
        "filename": "TRM_Guidelines.pdf",
    },
    {
        "name": "Business Continuity Management Guidelines (Jun 2022)",
        "url": "https://www.mas.gov.sg/-/media/mas/regulations-and-financial-stability/regulatory-and-supervisory-framework/risk-management/bcm-guidelines/bcm-guidelines-june-2022.pdf",
        "filename": "BCM_Guidelines.pdf",
    },
    {
        "name": "Guidelines on Outsourcing (Banks)",
        "url": "https://www.mas.gov.sg/-/media/mas-media-library/regulation/guidelines/bd/guidelines-on-outsourcing/guidelines-on-outsourcing-banks.pdf",
        "filename": "Outsourcing_Guidelines.pdf",
    },
    {
        "name": "Guidelines on Fair Dealing (May 2024)",
        "url": "https://www.mas.gov.sg/-/media/mas-media-library/fair-dealing-guidelines-30-may-2024.pdf",
        "filename": "Fair_Dealing_Guidelines.pdf",
    },
    {
        "name": "E-Payments User Protection Guidelines (Dec 2024)",
        "url": "https://www.mas.gov.sg/-/media/mas-media-library/regulation/guidelines/pso/e-payments-user-protection-guidelines-with-effect-from-16-dec-2024/e-payments-user-protection-guidelines-with-effect-from-16-december-2024.pdf",
        "filename": "E_Payments_Guidelines.pdf",
    },
]


def download_documents():
    """Download all MAS documents to data/raw/."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Downloading MAS regulatory documents to {RAW_DIR}")
    print("=" * 60)

    downloaded = 0
    skipped = 0
    failed = 0

    for doc in MAS_DOCUMENTS:
        filepath = RAW_DIR / doc["filename"]

        if filepath.exists():
            print(f"  [SKIP] {doc['name']} (already exists)")
            skipped += 1
            continue

        print(f"  [DOWNLOADING] {doc['name']}...")
        try:
            response = httpx.get(
                doc["url"],
                follow_redirects=True,
                timeout=60,
                headers={
                    "User-Agent": _BROWSER_UA,
                    "Accept": "application/pdf,*/*",
                },
            )
            response.raise_for_status()

            # A 200 OK is not proof of a PDF — MAS sits behind a CDN that can
            # return an HTML interstitial or error page with status 200. Validate
            # the PDF magic bytes before writing, so a bogus payload isn't saved
            # as a .pdf (which the [SKIP] branch would then make permanent on the
            # next run, and which fitz.open() would later choke on).
            if not response.content.startswith(b"%PDF-"):
                ctype = response.headers.get("content-type", "unknown")
                print(f"    -> FAILED: response was not a PDF (content-type: {ctype})")
                failed += 1
                continue

            filepath.write_bytes(response.content)
            size_mb = len(response.content) / (1024 * 1024)
            print(f"    -> Saved: {filepath.name} ({size_mb:.1f} MB)")
            downloaded += 1

        except Exception as e:
            print(f"    -> FAILED: {e}")
            failed += 1

    print("=" * 60)
    print(f"Done: {downloaded} downloaded, {skipped} skipped, {failed} failed")
    return downloaded, skipped, failed


if __name__ == "__main__":
    download_documents()
