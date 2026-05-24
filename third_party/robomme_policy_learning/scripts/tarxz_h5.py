#!/usr/bin/env python3
"""
Parallel in-place per-file .tar.xz compression for HDF5 (.h5/.hdf5) datasets.

Behavior:
- Compress:
    a.h5 -> a.h5.tar.xz   (in the same directory)
- Decompress:
    a.h5.tar.xz -> a.h5   (in the same directory)

By default, the source file/archive is kept.
You can optionally delete the source after a successful operation:
- --remove_original  for compress
- --remove_archive   for decompress

Examples:
  # Compress in place
  uv run scripts/tarxz_h5.py compress \
    --input_dir data/robomme_data_h5 \
    --jobs 16

  # Compress in place and delete original .h5 files
  uv run scripts/tarxz_h5.py compress \
    --input_dir data/robomme_data_h5 \
    --jobs 16 \
    --remove_original

  # Decompress in place
  uv run scripts/tarxz_h5.py decompress \
    --input_dir data/robomme_data_h5 \
    --jobs 16

  # Decompress in place and delete .tar.xz archives
  uv run scripts/tarxz_h5.py decompress \
    --input_dir data/robomme_data_h5 \
    --jobs 16 \
    --remove_archive
"""

from __future__ import annotations

import argparse
import os
import tarfile
import time
from dataclasses import dataclass
from multiprocessing import get_context
from pathlib import Path
from typing import Iterable, Sequence


H5_SUFFIXES = {".h5", ".hdf5"}
ARCHIVE_SUFFIX = ".tar.xz"


@dataclass(frozen=True)
class Job:
    src: Path
    dst_archive: Path


def iter_h5_files(input_dir: Path) -> Iterable[Path]:
    for root, _dirs, files in os.walk(input_dir):
        for name in files:
            p = Path(root) / name
            if p.suffix.lower() in H5_SUFFIXES:
                yield p


def iter_archives(input_dir: Path) -> Iterable[Path]:
    for root, _dirs, files in os.walk(input_dir):
        for name in files:
            if name.endswith(ARCHIVE_SUFFIX):
                yield Path(root) / name


def make_compress_jobs(input_dir: Path) -> list[Job]:
    input_dir = input_dir.resolve()
    jobs: list[Job] = []

    for src in iter_h5_files(input_dir):
        dst = src.with_name(src.name + ARCHIVE_SUFFIX)
        jobs.append(Job(src=src, dst_archive=dst))

    return jobs


def compress_one(job: Job, overwrite: bool, remove_original: bool) -> tuple[str, bool, str]:
    if job.dst_archive.exists() and not overwrite:
        return (str(job.dst_archive), False, "archive exists")

    with tarfile.open(job.dst_archive, mode="w:xz") as tf:
        tf.add(job.src, arcname=job.src.name, recursive=False)

    if remove_original:
        try:
            job.src.unlink()
        except FileNotFoundError:
            pass

    return (str(job.dst_archive), True, "ok")


def compress_one_from_args(args: tuple[Job, bool, bool]) -> tuple[str, bool, str]:
    job, overwrite, remove_original = args
    return compress_one(job, overwrite=overwrite, remove_original=remove_original)


def is_within_directory(directory: Path, target: Path) -> bool:
    directory = directory.resolve()
    target = target.resolve()
    try:
        target.relative_to(directory)
        return True
    except ValueError:
        return False


def safe_extract_tar(tf: tarfile.TarFile, dest_dir: Path) -> None:
    dest_dir = dest_dir.resolve()

    for member in tf.getmembers():
        member_path = dest_dir / member.name
        if not is_within_directory(dest_dir, member_path):
            raise RuntimeError(f"Unsafe path in tar: {member.name}")

    tf.extractall(path=dest_dir)


def decompress_one(args: tuple[Path, Path, bool, bool]) -> tuple[str, bool, str]:
    archive_path, output_dir, overwrite, remove_archive = args

    with tarfile.open(archive_path, mode="r:xz") as tf:
        members = tf.getmembers()
        if not members:
            return (str(archive_path), False, "empty archive")

        if not overwrite:
            for m in members:
                target = output_dir / m.name
                if target.exists():
                    return (str(archive_path), False, f"target exists: {m.name}")

        safe_extract_tar(tf, output_dir)

    if remove_archive:
        try:
            archive_path.unlink()
        except FileNotFoundError:
            pass

    return (str(archive_path), True, "ok")


def format_duration(seconds: float) -> str:
    total_seconds = max(0, int(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, mins = divmod(minutes, 60)

    if hours:
        return f"{hours}h {mins:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def run_pool_iter(fn, items: Sequence, jobs: int):
    if jobs <= 1:
        for item in items:
            yield fn(item)
        return

    ctx = get_context("spawn")
    with ctx.Pool(processes=jobs) as pool:
        yield from pool.imap_unordered(fn, items)


def cmd_compress(ns: argparse.Namespace) -> int:
    input_dir = Path(ns.input_dir)
    jobs = make_compress_jobs(input_dir)

    if not jobs:
        print(f"No .h5/.hdf5 files found under: {input_dir}")
        return 0

    items = [(job, ns.overwrite, ns.remove_original) for job in jobs]
    total = len(items)
    wrote = 0
    results = []
    started_at = time.perf_counter()

    print(f"Found {total} files. Starting compression with {ns.jobs} worker(s)...", flush=True)
    for index, result in enumerate(run_pool_iter(compress_one_from_args, items, ns.jobs), start=1):
        path, did, msg = result
        results.append(result)
        wrote += int(did)
        elapsed = time.perf_counter() - started_at
        eta = (elapsed / index) * (total - index)
        status = "WROTE" if did else "SKIP "
        print(
            f"[{index}/{total}] [{status}] {path} ({msg}) | "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
            flush=True,
        )

    skipped = len(results) - wrote
    total_elapsed = time.perf_counter() - started_at
    avg_per_file = total_elapsed / total if total else 0.0
    print(
        f"Finished compression. Wrote {wrote} archives. Skipped {skipped}. "
        f"Total time: {format_duration(total_elapsed)}. "
        f"Avg/file: {avg_per_file:.2f}s."
    )

    return 0


def cmd_decompress(ns: argparse.Namespace) -> int:
    input_dir = Path(ns.input_dir)
    archives = sorted(iter_archives(input_dir))

    if not archives:
        print(f"No {ARCHIVE_SUFFIX} archives found under: {input_dir}")
        return 0

    items = [(p, p.parent, ns.overwrite, ns.remove_archive) for p in archives]
    total = len(items)
    extracted = 0
    results = []
    started_at = time.perf_counter()

    print(f"Found {total} archives. Starting decompression with {ns.jobs} worker(s)...", flush=True)
    for index, result in enumerate(run_pool_iter(decompress_one, items, ns.jobs), start=1):
        path, did, msg = result
        results.append(result)
        extracted += int(did)
        elapsed = time.perf_counter() - started_at
        eta = (elapsed / index) * (total - index)
        status = "DONE " if did else "SKIP "
        print(
            f"[{index}/{total}] [{status}] {path} ({msg}) | "
            f"elapsed={format_duration(elapsed)} eta={format_duration(eta)}",
            flush=True,
        )

    skipped = len(results) - extracted
    total_elapsed = time.perf_counter() - started_at
    avg_per_archive = total_elapsed / total if total else 0.0
    print(
        f"Finished decompression. Extracted {extracted}. Skipped {skipped}. "
        f"Total time: {format_duration(total_elapsed)}. "
        f"Avg/archive: {avg_per_archive:.2f}s."
    )

    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Parallel in-place .tar.xz compress/decompress for HDF5 files."
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    pc = sub.add_parser(
        "compress",
        help="Compress all .h5/.hdf5 files into per-file .tar.xz archives in the same directory.",
    )
    pc.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing .h5/.hdf5 files (recursively).",
    )
    pc.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of worker processes.",
    )
    pc.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing archives.",
    )
    pc.add_argument(
        "--remove_original",
        action="store_true",
        help="Delete original .h5/.hdf5 file after successful compression.",
    )
    pc.set_defaults(func=cmd_compress)

    pd = sub.add_parser(
        "decompress",
        help="Decompress per-file .tar.xz archives back into .h5/.hdf5 files in the same directory.",
    )
    pd.add_argument(
        "--input_dir",
        required=True,
        help="Directory containing .tar.xz archives (recursively).",
    )
    pd.add_argument(
        "--jobs",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="Number of worker processes.",
    )
    pd.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing extracted files.",
    )
    pd.add_argument(
        "--remove_archive",
        action="store_true",
        help="Delete .tar.xz archive after successful extraction.",
    )
    pd.set_defaults(func=cmd_decompress)

    return p


def main(argv: list[str] | None = None) -> int:
    ns = build_parser().parse_args(argv)
    return int(ns.func(ns))


if __name__ == "__main__":
    raise SystemExit(main())