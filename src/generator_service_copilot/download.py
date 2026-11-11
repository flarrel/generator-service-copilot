from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import snapshot_download

MVTEC_REPOSITORY = "foersben/mvtec-ad"
MVTEC_REVISION = "c75b39616f84db43677bcc8228caaafaf5096d7f"


def download_mvtec_cable(destination: Path) -> Path:
    local_path = snapshot_download(
        repo_id=MVTEC_REPOSITORY,
        repo_type="dataset",
        revision=MVTEC_REVISION,
        allow_patterns="cable/**",
        local_dir=destination,
    )
    return Path(local_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the pinned MVTec AD cable subset")
    parser.add_argument(
        "--destination",
        type=Path,
        default=Path("data/external/mvtec_ad"),
    )
    args = parser.parse_args()
    print(download_mvtec_cable(args.destination).resolve())


if __name__ == "__main__":
    main()
