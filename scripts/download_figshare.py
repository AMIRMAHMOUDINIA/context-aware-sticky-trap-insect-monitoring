#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import re
import tarfile
import zipfile
from pathlib import Path

import requests
from tqdm import tqdm


def _metadata(article_id: int, version: int | None) -> dict:
    endpoints = []
    if version is not None:
        endpoints.append(f"https://api.figshare.com/v2/articles/{article_id}/versions/{version}")
    endpoints.append(f"https://api.figshare.com/v2/articles/{article_id}")
    errors: list[str] = []
    for endpoint in endpoints:
        try:
            response = requests.get(endpoint, timeout=60)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as exc:
            errors.append(f"{endpoint}: {exc}")
    raise RuntimeError("Unable to query Figshare metadata. " + " | ".join(errors))


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5()  # noqa: S324 - required for source-file checksum verification
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _download(url: str, destination: Path, expected_md5: str | None) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    temporary.unlink(missing_ok=True)
    try:
        with requests.get(url, stream=True, timeout=(30, 120)) as response:
            response.raise_for_status()
            total = int(response.headers.get("content-length", 0))
            with (
                temporary.open("wb") as handle,
                tqdm(
                    total=total,
                    unit="B",
                    unit_scale=True,
                    desc=destination.name,
                ) as progress,
            ):
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)
                        progress.update(len(chunk))
        if expected_md5:
            observed = _md5(temporary)
            if observed.lower() != expected_md5.lower():
                raise RuntimeError(
                    f"Checksum mismatch for {destination.name}: expected {expected_md5}, observed {observed}"
                )
        temporary.replace(destination)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _safe_target(base: Path, member_name: str) -> Path:
    target = (base / member_name).resolve()
    if base.resolve() not in target.parents and target != base.resolve():
        raise RuntimeError(f"Archive member escapes extraction directory: {member_name}")
    return target


def _extract(path: Path, destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    suffixes = "".join(path.suffixes).lower()
    if path.suffix.lower() == ".zip":
        with zipfile.ZipFile(path) as archive:
            for member in archive.infolist():
                _safe_target(destination, member.filename)
            archive.extractall(destination)
    elif suffixes.endswith((".tar.gz", ".tgz", ".tar.bz2", ".tar.xz")) or path.suffix == ".tar":
        with tarfile.open(path) as archive:
            for member in archive.getmembers():
                _safe_target(destination, member.name)
                if member.issym() or member.islnk():
                    raise RuntimeError(f"Archive links are not extracted: {member.name}")
            archive.extractall(destination)
    else:
        return


def main() -> None:
    parser = argparse.ArgumentParser(description="Download public files from a Figshare article.")
    parser.add_argument("--article-id", type=int, required=True)
    parser.add_argument("--version", type=int, default=None)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--file-regex", default=None, help="Optional regex used to select file names."
    )
    parser.add_argument("--extract", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    metadata = _metadata(args.article_id, args.version)
    files = metadata.get("files", [])
    if args.file_regex:
        pattern = re.compile(args.file_regex, flags=re.IGNORECASE)
        files = [item for item in files if pattern.search(item.get("name", ""))]
    if not files:
        raise RuntimeError("No Figshare files matched the request.")

    print(f"Article: {metadata.get('title', args.article_id)}")
    for item in files:
        print(f"- {item.get('name')} ({item.get('size', 'unknown')} bytes)")
    if args.list_only:
        return

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for item in files:
        destination = output_dir / item["name"]
        if destination.exists() and not args.overwrite:
            expected = item.get("supplied_md5")
            if expected:
                observed = _md5(destination)
                if observed.lower() == expected.lower():
                    print(f"Skipping verified file: {destination}")
                    continue
                raise RuntimeError(
                    f"Existing file failed checksum verification: {destination}. "
                    "Remove it or rerun with --overwrite."
                )
            raise FileExistsError(
                f"File already exists and no source checksum is available: {destination}. "
                "Use --overwrite to replace it."
            )
        _download(item["download_url"], destination, item.get("supplied_md5"))
        if args.extract:
            extract_dir = output_dir / destination.stem
            _extract(destination, extract_dir)


if __name__ == "__main__":
    main()
