#!/usr/bin/env python3
"""Download GSM8K parquet files directly to a local folder."""

from __future__ import annotations

import argparse
from pathlib import Path

import requests


FILES = {
    "README.md": "https://huggingface.co/datasets/gsm8k/resolve/main/README.md",
    "eval.yaml": "https://huggingface.co/datasets/gsm8k/resolve/main/eval.yaml",
    "main/train-00000-of-00001.parquet": "https://huggingface.co/datasets/gsm8k/resolve/main/main/train-00000-of-00001.parquet",
    "main/test-00000-of-00001.parquet": "https://huggingface.co/datasets/gsm8k/resolve/main/main/test-00000-of-00001.parquet",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--chunk-size", type=int, default=1024 * 1024)
    return parser.parse_args()


def download_file(url: str, destination: Path, timeout: int, chunk_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=timeout) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=chunk_size):
                if chunk:
                    handle.write(chunk)


def main() -> None:
    args = parse_args()
    for rel_path, url in FILES.items():
        destination = args.output_dir / rel_path
        print(f"Downloading {rel_path} ...", flush=True)
        download_file(url, destination, args.timeout, args.chunk_size)
        print(f"Saved to {destination}", flush=True)


if __name__ == "__main__":
    main()
