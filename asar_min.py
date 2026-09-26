#!/usr/bin/env python3
"""Minimal asar reader/writer (stdlib only) for the Modly draw patch.

Electron asar layout (all UInt32 little-endian):
  [payload_size=4][header_buf_len]          <- 8-byte size pickle
  [payload_size][strlen][json][pad]          <- header pickle (header_buf_len bytes)
  [file data...]                             <- offsets in header are from here

Header JSON: {"files": {name: dir|file|link}}. File entries:
  {"size": N, "offset": "M"} (+ optional "executable"/"unpacked"/"integrity").
Unpacked files have no data in the archive (live in app.asar.unpacked/).
Only regular files are (re)packed; dirs/links/unpacked entries are preserved.
"""

import hashlib
import json
import os
import struct
from pathlib import Path

_UINT = struct.Struct("<I")


def _read_u32(buf, pos):
    return _UINT.unpack_from(buf, pos)[0]


def read_header(asar_path):
    """Return (header_dict, data_start_offset)."""
    with open(asar_path, "rb") as fh:
        head = fh.read(16)
    if _read_u32(head, 0) != 4:
        raise ValueError("not an asar file (bad size pickle)")
    header_buf_len = _read_u32(head, 4)
    with open(asar_path, "rb") as fh:
        fh.seek(8)
        header_buf = fh.read(header_buf_len)
    payload = _read_u32(header_buf, 0)
    strlen = _read_u32(header_buf, 4)
    raw = header_buf[8:8 + strlen]
    header = json.loads(raw.decode("utf-8"))
    if not isinstance(header, dict) or "files" not in header:
        raise ValueError("bad asar header json")
    _ = payload
    return header, 8 + header_buf_len


def _walk(header_files, prefix=""):
    """Yield (path, entry) for every file entry (not dirs)."""
    for name, entry in header_files.items():
        rel = f"{prefix}/{name}" if prefix else name
        if not isinstance(entry, dict):
            continue
        if "files" in entry:
            yield from _walk(entry["files"], rel)
        else:
            yield rel, entry


def list_files(header):
    return list(_walk(header["files"]))


def extract(asar_path, out_dir, only=None):
    """Extract files. only = set of archive-relative paths (or None = all
    regular packed files; unpacked/link entries are skipped)."""
    header, data_start = read_header(asar_path)
    out_dir = Path(out_dir)
    count = 0
    with open(asar_path, "rb") as fh:
        for rel, entry in _walk(header["files"]):
            if only is not None and rel not in only:
                continue
            if "link" in entry or entry.get("unpacked"):
                continue
            size = int(entry["size"])
            offset = int(entry["offset"])
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            fh.seek(data_start + offset)
            with open(dest, "wb") as out:
                remaining = size
                while remaining > 0:
                    chunk = fh.read(min(1024 * 1024, remaining))
                    if not chunk:
                        raise ValueError(f"truncated asar at {rel}")
                    out.write(chunk)
                    remaining -= len(chunk)
            if entry.get("executable"):
                try:
                    os.chmod(dest, 0o755)
                except OSError:
                    pass
            count += 1
    return count


def _build_pickles(header_json_bytes):
    """Return (size_pickle, header_pickle) for header json bytes."""
    strlen = len(header_json_bytes)
    pad = (4 - (strlen % 4)) % 4
    payload = 4 + strlen + pad
    header_pickle = (
        _UINT.pack(payload)
        + _UINT.pack(strlen)
        + header_json_bytes
        + b"\x00" * pad
    )
    size_pickle = _UINT.pack(4) + _UINT.pack(len(header_pickle))
    return size_pickle, header_pickle


def _integrity(data):
    block_size = 4194304
    blocks = [
        hashlib.sha256(data[i:i + block_size]).hexdigest()
        for i in range(0, max(len(data), 1), block_size)
    ]
    return {
        "algorithm": "SHA256",
        "hash": hashlib.sha256(data).hexdigest(),
        "blockSize": block_size,
        "blocks": blocks,
    }


def pack(src_dir, out_path, order=None):
    """Pack a directory tree into an asar. order = list of archive-relative
    paths for deterministic layout (defaults to sorted). Returns count."""
    src_dir = Path(src_dir)
    if order is None:
        order = sorted(
            p.relative_to(src_dir).as_posix()
            for p in src_dir.rglob("*")
            if p.is_file() and not p.is_symlink()
        )
    blobs = []
    files_tree = {}
    for rel in order:
        data = (src_dir / rel).read_bytes()
        blobs.append(data)
        node = files_tree
        parts = rel.split("/")
        for part in parts[:-1]:
            node = node.setdefault(part, {}).setdefault("files", {})
        leaf = {"size": len(data), "offset": "0"}
        blobs[-1] = (rel, data, leaf)
        node[parts[-1]] = leaf
    # Assign sequential offsets, rebuild header with integrity.
    offset = 0
    for _rel, data, leaf in blobs:
        leaf["offset"] = str(offset)
        leaf["integrity"] = _integrity(data)
        offset += len(data)
    header_json = json.dumps({"files": files_tree}, sort_keys=True).encode("utf-8")
    size_pickle, header_pickle = _build_pickles(header_json)
    with open(out_path, "wb") as fh:
        fh.write(size_pickle)
        fh.write(header_pickle)
        for _rel, data, _leaf in blobs:
            fh.write(data)
    return len(blobs)


def repack(original_header, tree_dir, out_path):
    """Rebuild an asar from an extracted tree, preserving the ORIGINAL
    header order and all link/unpacked entries verbatim. Packed regular
    files are re-read from tree_dir (modified or not) with fresh offsets
    and integrity. Returns packed file count."""
    import copy

    tree_dir = Path(tree_dir)
    header = copy.deepcopy(original_header)
    blobs = []
    order = []

    def collect(node, prefix=""):
        for name, entry in node.items():
            if not isinstance(entry, dict):
                continue
            rel = f"{prefix}/{name}" if prefix else name
            if "files" in entry:
                collect(entry["files"], rel)
            elif "link" in entry or entry.get("unpacked"):
                continue
            else:
                order.append((rel, entry))

    collect(header["files"])
    offset = 0
    for rel, entry in order:
        data = (tree_dir / rel).read_bytes()
        blobs.append(data)
        entry["offset"] = str(offset)
        entry["integrity"] = _integrity(data)
        entry["size"] = len(data)
        offset += len(data)
    header_json = json.dumps(header, sort_keys=True).encode("utf-8")
    size_pickle, header_pickle = _build_pickles(header_json)
    with open(out_path, "wb") as fh:
        fh.write(size_pickle)
        fh.write(header_pickle)
        for data in blobs:
            fh.write(data)
    return len(blobs)


def sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
