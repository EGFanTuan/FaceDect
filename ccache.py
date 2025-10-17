#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ccache.py - Clean .npy cache files under ./datasets/train and ./datasets/val

Features:
- Dry run by default (preview deletions)
- Age filter: delete only files older than N days
- Include/Exclude glob patterns
- Interactive confirmation (skipped with --yes)
- Verbose logging and summary

Safety:
- Only operates inside ./datasets/train and ./datasets/val
- Skips non-.npy files by default unless explicitly included
"""

from __future__ import annotations

import argparse
import fnmatch
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List


DATASETS_DIR = Path('./datasets').resolve()
TARGET_SUBDIRS = [DATASETS_DIR / 'train', DATASETS_DIR / 'val', DATASETS_DIR / 'test']


@dataclass
class CleanOptions:
	dry_run: bool = True
	older_than_days: float | None = None
	include: List[str] | None = None
	exclude: List[str] | None = None
	yes: bool = False
	verbose: bool = False


def within_targets(path: Path) -> bool:
	"""Ensure the file resides under one of the target subdirectories."""
	try:
		path = path.resolve()
		for sub in TARGET_SUBDIRS:
			try:
				path.relative_to(sub)
				return True
			except ValueError:
				continue
		return False
	except Exception:
		return False


def match_patterns(name: str, patterns: Iterable[str] | None, default: bool) -> bool:
	"""Return True if name matches patterns. If patterns is None, return default."""
	if not patterns:
		return default
	return any(fnmatch.fnmatch(name, p) for p in patterns)


def iter_cache_files(opts: CleanOptions) -> List[Path]:
	"""Collect candidate .npy files under targets according to filters."""
	candidates: List[Path] = []
	now = time.time()
	age_threshold = None
	if opts.older_than_days is not None and opts.older_than_days >= 0:
		age_threshold = now - (opts.older_than_days * 86400)

	for sub in TARGET_SUBDIRS:
		if not sub.exists():
			if opts.verbose:
				print(f"[info] Skip missing directory: {sub}")
			continue
		for root, _dirs, files in os.walk(sub):
			for fname in files:
				# Default include only .npy files
				if not fname.lower().endswith('.npy'):
					# Allow override via include patterns
					if not match_patterns(fname, opts.include, default=False):
						continue
				# Exclude patterns
				if match_patterns(fname, opts.exclude, default=False):
					continue

				fpath = Path(root) / fname
				# Safety: within targets
				if not within_targets(fpath):
					continue

				if age_threshold is not None:
					try:
						mtime = fpath.stat().st_mtime
						if mtime > age_threshold:
							continue
					except FileNotFoundError:
						continue

				candidates.append(fpath)

	return candidates


def prompt_confirm(count: int, dry_run: bool) -> bool:
	mode = 'preview' if dry_run else 'delete'
	print(f"About to {mode} {count} file(s). Continue? [y/N]: ", end='', flush=True)
	try:
		resp = input().strip().lower()
	except EOFError:
		return False
	return resp in ('y', 'yes')


def clean(opts: CleanOptions) -> int:
	candidates = iter_cache_files(opts)

	if not candidates:
		print("No matching cache files found.")
		return 0

	total_size = 0
	for p in candidates:
		try:
			total_size += p.stat().st_size
		except FileNotFoundError:
			pass

	print("=" * 60)
	print(f"Cache cleaner target directories:")
	for sub in TARGET_SUBDIRS:
		print(f"  - {sub}")
	print("Filters:")
	print(f"  include: {opts.include or ['*.npy (default)']}")
	print(f"  exclude: {opts.exclude or []}")
	if opts.older_than_days is not None:
		print(f"  older-than: {opts.older_than_days} day(s)")
	print(f"Mode: {'DRY-RUN (no deletion)' if opts.dry_run else 'DELETE'}")
	print("Candidates:")
	for p in candidates[:10]:
		print(f"  - {p}")
	if len(candidates) > 10:
		print(f"  ... and {len(candidates) - 10} more")
	print(f"Total files: {len(candidates)} | Total size: {total_size/1e6:.2f} MB")
	print("=" * 60)

	if not opts.yes:
		if not prompt_confirm(len(candidates), opts.dry_run):
			print("Cancelled.")
			return 0

	deleted = 0
	for p in candidates:
		if opts.dry_run:
			if opts.verbose:
				print(f"[dry-run] would delete: {p}")
			continue
		try:
			p.unlink()
			deleted += 1
			if opts.verbose:
				print(f"[del] {p}")
		except FileNotFoundError:
			if opts.verbose:
				print(f"[skip] not found: {p}")
		except PermissionError:
			print(f"[warn] permission denied: {p}")
		except Exception as e:
			print(f"[warn] failed to delete {p}: {e}")

	print(f"Done. {'Previewed' if opts.dry_run else 'Deleted'} {len(candidates) if opts.dry_run else deleted} file(s).")
	return 0


def parse_args(argv: List[str]) -> CleanOptions:
	parser = argparse.ArgumentParser(
		description='Clean .npy cache files under ./datasets/train and ./datasets/val'
	)
	parser.add_argument('--no-dry-run', action='store_true', help='Perform actual deletion (default is dry-run)')
	parser.add_argument('--older-than', type=float, default=None, help='Only delete files older than N days')
	parser.add_argument('--include', nargs='*', default=None, help='Additional include glob patterns (e.g., *.npy *.cache.npy)')
	parser.add_argument('--exclude', nargs='*', default=None, help='Exclude glob patterns (e.g., *keep*.npy)')
	parser.add_argument('-y', '--yes', action='store_true', help='Do not prompt for confirmation')
	parser.add_argument('-v', '--verbose', action='store_true', help='Verbose output')

	args = parser.parse_args(argv)

	return CleanOptions(
		dry_run=not args.no_dry_run,
		older_than_days=args.older_than,
		include=args.include,
		exclude=args.exclude,
		yes=args.yes,
		verbose=args.verbose,
	)


def main(argv: List[str] | None = None) -> int:
	opts = parse_args(sys.argv[1:] if argv is None else argv)
	return clean(opts)


if __name__ == '__main__':
	raise SystemExit(main())
