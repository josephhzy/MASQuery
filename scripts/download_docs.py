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
    failed = 0

    for doc in MAS_DOCUMENTS:
        filepath = RAW_DIR / doc["filename"]

        if filepath.exists():
            print(f"  [SKIP] {doc['name']} (already exists)")
            downloaded += 1
            continue

        print(f"  [DOWNLOADING] {doc['name']}...")
        try:
            response = httpx.get(
                doc["url"],
                follow_redirects=True,
                timeout=60,
                headers={
                    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                    "Accept": "application/pdf,*/*",
                },
            )
            response.raise_for_status()

            filepath.write_bytes(response.content)
            size_mb = len(response.content) / (1024 * 1024)
            print(f"    -> Saved: {filepath.name} ({size_mb:.1f} MB)")
            downloaded += 1

        except Exception as e:
            print(f"    -> FAILED: {e}")
            failed += 1

    print("=" * 60)
    print(f"Done: {downloaded} downloaded, {failed} failed")
    return downloaded, failed


if __name__ == "__main__":
    download_documents()
