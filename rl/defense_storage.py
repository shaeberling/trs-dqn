"""Lossless APFS clone deduplication of immutable, own Defense artifacts.

Default is an inventory. --apply keeps every path and byte, excludes live
latest/best links and logs, and never deletes a historical artifact. APFS
copy-on-write clones, unlike hard links, remain independent when modified.
"""

import argparse
from collections import defaultdict
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import stat
import sys
import tempfile


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def duplicates(repo):
    roots = [repo/'results/defense/training', repo/'results/defense/learned/versions']
    roots += sorted((repo/'runs').glob('defense-*'))
    groups = defaultdict(list)
    for root in roots:
        if not root.is_dir() or root.is_symlink():
            continue
        for path in sorted(root.rglob('*')):
            if path.name not in ('model.safetensors', 'target.safetensors', 'optimizer.npz', 'trace.npz'):
                continue
            if (not path.is_file() or path.is_symlink() or path.resolve() != path.absolute()
                    or 'latest' in path.parts or 'best' in path.parts):
                continue
            if not any(p.startswith(('step-', 'replay')) or p in
                       ('checkpoint', 'final-checkpoint', 'versions', 'first-replay', 'best-effort')
                       for p in path.relative_to(repo).parts[:-1]):
                continue
            state = path.parent/'state.json'
            if not state.is_file() or state.is_symlink():
                continue
            try:
                config = json.loads(state.read_text())['config']
            except (ValueError, KeyError):
                continue
            if config.get('game') != 'defense':
                continue
            groups[(path.stat().st_size, digest(path))].append(path)
    return {key: paths for key, paths in groups.items() if len(paths) > 1}


def signature(path):
    s = path.stat(follow_symlinks=False)
    return s.st_dev, s.st_ino, s.st_size, s.st_mtime_ns, s.st_ctime_ns


def clone_duplicate(source, target, expected):
    if sys.platform != 'darwin':
        raise RuntimeError('APFS cloning requires macOS; no copying or hard-link fallback')
    if source == target or source.is_symlink() or target.is_symlink():
        raise ValueError('distinct regular immutable files required')
    if source.resolve() != source.absolute() or target.resolve() != target.absolute():
        raise ValueError('symlinked parent paths are not eligible')
    source_stat, target_stat = source.stat(), target.stat()
    if (not stat.S_ISREG(source_stat.st_mode) or not stat.S_ISREG(target_stat.st_mode)
            or source_stat.st_dev != target_stat.st_dev):
        raise ValueError('regular files on the same filesystem required')
    before_source, before_target = signature(source), signature(target)
    if digest(source) != expected or digest(target) != expected:
        raise ValueError('artifact changed since inventory; nothing replaced')
    folder = Path(tempfile.mkdtemp(prefix='.defense-cow-', dir=target.parent))
    clone = folder/'clone'
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        clonefile = libc.clonefile
        clonefile.argtypes = (ctypes.c_char_p, ctypes.c_char_p, ctypes.c_int)
        clonefile.restype = ctypes.c_int
        if clonefile(os.fsencode(source), os.fsencode(clone), 0):
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        # Keep destination metadata too; do not replace it with the source's.
        shutil.copystat(target, clone, follow_symlinks=False)
        os.chown(clone, target_stat.st_uid, target_stat.st_gid)
        if hasattr(os, 'listxattr'):
            for name in os.listxattr(clone):
                os.removexattr(clone, name)
            for name in os.listxattr(target):
                os.setxattr(clone, name, os.getxattr(target, name))
        if (digest(clone) != expected or signature(source) != before_source
                or signature(target) != before_target):
            raise RuntimeError('artifact changed during cloning; original destination preserved')
        os.replace(clone, target)
        if digest(target) != expected:
            raise RuntimeError('post-replacement checksum mismatch')
    finally:
        if clone.exists():
            clone.unlink()  # Only this newly created duplicate, never the source/target.
        folder.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    repo = Path(__file__).resolve().parents[1]
    before = shutil.disk_usage(repo).free
    groups = duplicates(repo)
    count = 0
    for (size, checksum), paths in groups.items():
        if args.apply:
            for target in paths[1:]:
                clone_duplicate(paths[0], target, checksum)
                count += 1
    print(json.dumps(dict(applied=args.apply, duplicate_groups=len(groups), replaced_files=count,
                         duplicate_logical_bytes=sum(size*(len(paths)-1) for (size,_),paths in groups.items()),
                         free_bytes_before=before, free_bytes_after=shutil.disk_usage(repo).free,
                         note='All paths and bytes retained. Logical duplicates may already share APFS extents; free space also changes while learners run.')))


if __name__ == '__main__':
    main()
