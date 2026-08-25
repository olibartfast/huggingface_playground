"""Fetch HMDB-51 videos + official splits into the layout this harness expects.

    <root>/videos/<class_name>/<clip>.avi
    <root>/splits/<class_name>_test_split<N>.txt

The original Brown University host is dead (it 301s to the lab homepage), so
videos come from a Hub mirror of ``hmdb51_org`` and the official split archive
comes from a Wayback snapshot. Nested archives are unpacked with libarchive,
which reads RAR3 without needing a non-free ``unrar`` binary.

    python download_hmdb51.py /data/hmdb51
"""

from __future__ import annotations

import argparse
import os
import sys
import zipfile

VIDEO_REPO = "innat/HMDB51"
VIDEO_FILE = "hmdb51_org.zip"
SPLITS_URL = (
    "https://web.archive.org/web/20211104084122/"
    "https://serre-lab.clps.brown.edu/wp-content/uploads/2013/10/test_train_splits.rar"
)

EXPECTED_CLASSES = 51
EXPECTED_VIDEOS = 6766
EXPECTED_SPLITS = 153


def extract_archive(path: str, dest: str, suffixes: tuple[str, ...]) -> int:
    """Extract members ending in ``suffixes`` from any libarchive-readable file."""
    import libarchive

    os.makedirs(dest, exist_ok=True)
    count = 0
    with libarchive.file_reader(path) as archive:
        for entry in archive:
            name = os.path.basename(str(entry.pathname))
            if not name.endswith(suffixes):
                continue
            with open(os.path.join(dest, name), "wb") as fh:
                for block in entry.get_blocks():
                    fh.write(block)
            count += 1
    return count


def fetch_splits(root: str, archives: str) -> None:
    import urllib.request

    splits_dir = os.path.join(root, "splits")
    existing = (
        len([f for f in os.listdir(splits_dir) if f.endswith(".txt")])
        if os.path.isdir(splits_dir)
        else 0
    )
    if existing == EXPECTED_SPLITS:
        print(f"==> splits already present ({existing} files)")
        return

    rar = os.path.join(archives, "test_train_splits.rar")
    if not os.path.exists(rar):
        print(f"==> downloading official splits from the Wayback snapshot")
        urllib.request.urlretrieve(SPLITS_URL, rar)

    n = extract_archive(rar, splits_dir, ("_test_split1.txt", "_test_split2.txt", "_test_split3.txt"))
    print(f"==> extracted {n} split files")


def fetch_videos(root: str, archives: str) -> None:
    from huggingface_hub import hf_hub_download

    videos_dir = os.path.join(root, "videos")
    if os.path.isdir(videos_dir) and len(os.listdir(videos_dir)) == EXPECTED_CLASSES:
        print(f"==> videos already present ({EXPECTED_CLASSES} class directories)")
        return

    print(f"==> downloading {VIDEO_REPO}/{VIDEO_FILE} (~2.1 GB, resumable)")
    zip_path = hf_hub_download(
        VIDEO_REPO, VIDEO_FILE, repo_type="dataset", local_dir=archives
    )

    print("==> unpacking outer zip")
    inner_dir = os.path.join(archives, "class_archives")
    os.makedirs(inner_dir, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        members = zf.namelist()
        if any(m.endswith(".avi") for m in members):
            # Zip already holds the videos directly.
            for m in members:
                if not m.endswith(".avi"):
                    continue
                # .../<class>/<clip>.avi
                parts = m.strip("/").split("/")
                if len(parts) < 2:
                    continue
                class_name = parts[-2]
                target = os.path.join(videos_dir, class_name)
                os.makedirs(target, exist_ok=True)
                with zf.open(m) as src, open(os.path.join(target, parts[-1]), "wb") as dst:
                    dst.write(src.read())
        else:
            zf.extractall(inner_dir)

    # The canonical layout nests one RAR per class inside the outer archive.
    nested = []
    for dirpath, _, filenames in os.walk(inner_dir):
        nested.extend(
            os.path.join(dirpath, f) for f in filenames if f.endswith(".rar")
        )
    if nested:
        print(f"==> unpacking {len(nested)} per-class archives")
        for rar in nested:
            class_name = os.path.basename(rar)[: -len(".rar")]
            extract_archive(rar, os.path.join(videos_dir, class_name), (".avi",))


def verify(root: str) -> bool:
    videos_dir, splits_dir = os.path.join(root, "videos"), os.path.join(root, "splits")
    classes = (
        sorted(
            d for d in os.listdir(videos_dir)
            if os.path.isdir(os.path.join(videos_dir, d))
        )
        if os.path.isdir(videos_dir)
        else []
    )
    videos = sum(
        len([f for f in os.listdir(os.path.join(videos_dir, c)) if f.endswith(".avi")])
        for c in classes
    )
    splits = (
        len([f for f in os.listdir(splits_dir) if f.endswith(".txt")])
        if os.path.isdir(splits_dir)
        else 0
    )

    print(f"\n==> {root}")
    print(f"    classes: {len(classes):5d} (expected {EXPECTED_CLASSES})")
    print(f"    videos:  {videos:5d} (expected {EXPECTED_VIDEOS})")
    print(f"    splits:  {splits:5d} (expected {EXPECTED_SPLITS})")

    ok = len(classes) == EXPECTED_CLASSES and splits == EXPECTED_SPLITS
    if not ok:
        print("    [error] incomplete - re-run to resume")
    elif videos != EXPECTED_VIDEOS:
        print("    [warn] video count differs from the canonical release")
    return ok


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", help="destination directory")
    args = parser.parse_args()

    root = os.path.abspath(args.root)
    archives = os.path.join(root, "_archives")
    os.makedirs(archives, exist_ok=True)

    fetch_splits(root, archives)
    fetch_videos(root, archives)

    if not verify(root):
        sys.exit(1)
    print(f"\nArchives kept in {archives} (delete to reclaim ~2 GB).")


if __name__ == "__main__":
    main()
