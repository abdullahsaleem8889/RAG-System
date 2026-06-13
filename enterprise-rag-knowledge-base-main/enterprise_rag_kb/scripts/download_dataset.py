"""
Enterprise RAG - Kaggle Dataset Downloader
============================================
Downloads the Enterprise RAG challenge dataset from Kaggle.
Dataset: https://www.kaggle.com/datasets/rrr3try/enterprise-rag-markdown

Requirements:
    pip install kaggle
    Configure ~/.kaggle/kaggle.json with your API credentials

Usage:
    python scripts/download_dataset.py
    python scripts/download_dataset.py --dataset dkhundley/sample-rag-knowledge-item-dataset
"""

import sys
import os
import zipfile
import shutil
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Available RAG datasets
DATASETS = {
    "enterprise-rag": {
        "slug": "rrr3try/enterprise-rag-markdown",
        "description": "100 enterprise PDFs converted to Markdown from Enterprise RAG Challenge",
        "recommended": True,
    },
    "sample-rag": {
        "slug": "dkhundley/sample-rag-knowledge-item-dataset",
        "description": "Sample RAG knowledge items dataset",
        "recommended": False,
    },
    "datasets-for-rag": {
        "slug": "denniswang07/datasets-for-rag",
        "description": "Multi-domain documents for RAG",
        "recommended": False,
    },
    "rag-qa-synthetic": {
        "slug": "shivvamm/rag-synthetic-data-for-qa-systems-prompts",
        "description": "Synthetic Q&A data for RAG systems",
        "recommended": False,
    },
    "rag-financial": {
        "slug": "thedevastator/rag-financial-legal-evaluation-dataset",
        "description": "200 enterprise Q&A samples for financial & legal domains",
        "recommended": False,
    },
}


def list_datasets():
    """Print available datasets."""
    print("\n Available RAG Datasets:")
    print("=" * 70)
    for key, info in DATASETS.items():
        tag = "  RECOMMENDED" if info["recommended"] else ""
        print(f"\n  Key: {key}{tag}")
        print(f"  Slug: {info['slug']}")
        print(f"  Description: {info['description']}")
    print()


def download_dataset(dataset_key: str = "enterprise-rag", output_dir: str = "data/raw"):
    """Download a dataset from Kaggle."""

    if dataset_key not in DATASETS:
        print(f" Unknown dataset: {dataset_key}")
        print(f"Available: {list(DATASETS.keys())}")
        return False

    dataset_info = DATASETS[dataset_key]
    slug = dataset_info["slug"]

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f" Downloading: {slug}")
    print(f"   To: {output_path}")
    print(f"   Description: {dataset_info['description']}\n")

    # Method 1: Using kaggle CLI
    try:
        import subprocess
        result = subprocess.run(
            ["kaggle", "datasets", "download", "-d", slug,
             "-p", str(output_path), "--unzip"],
            capture_output=True, text=True, check=True
        )
        print(" Download complete via kaggle CLI!")
        print(f"Files: {list(output_path.iterdir())}")
        return True

    except subprocess.CalledProcessError as e:
        print(f"  kaggle CLI error: {e.stderr}")
    except FileNotFoundError:
        print("  kaggle CLI not found")

    # Method 2: Using kaggle Python API
    try:
        from kaggle.api.kaggle_api_extended import KaggleApiExtended
        api = KaggleApiExtended()
        api.authenticate()

        owner, dataset_name = slug.split("/")
        api.dataset_download_files(
            dataset=slug,
            path=str(output_path),
            unzip=True
        )
        print(" Download complete via kaggle API!")
        return True

    except ImportError:
        print("  kaggle package not installed")
    except Exception as e:
        print(f"  kaggle API error: {e}")

    # Manual instructions
    print("\n" + "=" * 60)
    print(" MANUAL DOWNLOAD INSTRUCTIONS")
    print("=" * 60)
    print(f"\n1. Go to: https://www.kaggle.com/datasets/{slug}")
    print(f"2. Click 'Download' button")
    print(f"3. Extract the ZIP file to: {output_path.absolute()}")
    print(f"\nOR use kaggle CLI:")
    print(f"   1. pip install kaggle")
    print(f"   2. Download API token from kaggle.com/settings/account")
    print(f"   3. Place kaggle.json in ~/.kaggle/")
    print(f"   4. Run: kaggle datasets download -d {slug} -p {output_path} --unzip")
    print()

    return False


def ingest_downloaded_data(data_dir: str = "data/raw"):
    """Ingest downloaded dataset into the RAG pipeline."""
    from src.pipeline.rag_pipeline import RAGPipeline

    data_path = Path(data_dir)
    files = list(data_path.rglob("*"))
    supported = {".pdf", ".txt", ".md", ".csv", ".json", ".docx"}
    doc_files = [f for f in files if f.suffix.lower() in supported]

    if not doc_files:
        print(f" No supported documents found in {data_dir}")
        return

    print(f"\n Found {len(doc_files)} documents to ingest...")

    pipeline = RAGPipeline(auto_load=True)
    result = pipeline.ingest_directory(data_dir)

    print("\n Ingestion Results:")
    for k, v in result.items():
        if k != "per_sample":
            print(f"  {k}: {v}")

    print("\n Testing with a sample query...")
    test_query = "What are the main topics in these documents?"
    result = pipeline.query(test_query)
    print(f"Q: {test_query}")
    print(f"A: {result['answer'][:400]}")


def main():
    parser = argparse.ArgumentParser(
        description="Download Kaggle datasets for the Enterprise RAG project"
    )
    parser.add_argument(
        "--dataset", type=str, default="enterprise-rag",
        help=f"Dataset key. Options: {list(DATASETS.keys())}"
    )
    parser.add_argument(
        "--output", type=str, default="data/raw",
        help="Output directory for downloaded data"
    )
    parser.add_argument(
        "--list", action="store_true",
        help="List available datasets"
    )
    parser.add_argument(
        "--ingest", action="store_true",
        help="Also ingest the downloaded data into the pipeline"
    )
    args = parser.parse_args()

    if args.list:
        list_datasets()
        return

    success = download_dataset(args.dataset, args.output)

    if success and args.ingest:
        ingest_downloaded_data(args.output)


if __name__ == "__main__":
    main()
