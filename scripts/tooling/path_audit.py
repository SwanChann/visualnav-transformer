#!/usr/bin/env python3
"""Scan the repository for hard-coded absolute paths and platform-coupled strings."""

from __future__ import annotations

import argparse
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple


# 中文注释：通过仓库根目录推导扫描范围，统一排查绝对路径问题
REPO_ROOT = Path(__file__).resolve().parents[2]

TEXT_SUFFIXES = {
    ".py",
    ".md",
    ".txt",
    ".yaml",
    ".yml",
    ".json",
    ".toml",
    ".ini",
    ".cfg",
    ".sh",
    ".launch",
}

PATTERNS: Sequence[Tuple[str, re.Pattern[str]]] = [
    ("linux_home", re.compile(r"/home/[A-Za-z0-9_.-]+/[^\"'\s]+")),
    ("workspace_linux", re.compile(r"/workspace/[^\"'\s]+")),
    ("windows_drive", re.compile(r"[A-Za-z]:\\\\[^\"'\s]+")),
]

SKIP_PARTS = {"__pycache__", ".git", ".mypy_cache", ".pytest_cache", ".ruff_cache"}
SKIP_PREFIXES = {
    Path("Lite3_rl_deploy") / "build",
    Path("Lite3_rl_deploy") / "third_party",
    Path("third_party") / "lite3" / "Lite3_rl_deploy" / "build",
    Path("third_party") / "lite3" / "Lite3_rl_deploy" / "third_party",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Find absolute paths and portability risks in text files."
    )
    parser.add_argument(
        "--include-results",
        action="store_true",
        help="Include generated results files in the scan output.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        help="Save the report under results/tools/.",
    )
    return parser.parse_args()


def should_skip_path(path: Path, include_results: bool) -> bool:
    rel_path = path.relative_to(REPO_ROOT)
    if any(part in SKIP_PARTS for part in rel_path.parts):
        return True
    if not include_results and rel_path.parts and rel_path.parts[0] == "results":
        return True
    return any(rel_path == prefix or prefix in rel_path.parents for prefix in SKIP_PREFIXES)


def iter_text_files(root: Path, include_results: bool) -> Iterable[Path]:
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        if should_skip_path(path, include_results):
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES:
            continue
        yield path


def scan_file(path: Path) -> List[Tuple[int, str, str]]:
    findings: List[Tuple[int, str, str]] = []
    if path.name == "path_audit.py":
        return findings
    try:
        content = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        content = path.read_text(encoding="utf-8", errors="ignore")

    for line_no, line in enumerate(content.splitlines(), start=1):
        for label, pattern in PATTERNS:
            match = pattern.search(line)
            if match:
                findings.append((line_no, label, match.group(0)))
    return findings


def build_report(include_results: bool) -> List[str]:
    lines = [
        "NoMaD Thesis Path Audit",
        f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"Repo root: {REPO_ROOT}",
        "",
    ]

    total = 0
    files = 0
    for path in iter_text_files(REPO_ROOT, include_results):
        findings = scan_file(path)
        if not findings:
            continue
        files += 1
        rel_path = path.relative_to(REPO_ROOT)
        lines.append(f"## {rel_path}")
        for line_no, label, snippet in findings:
            total += 1
            lines.append(f"- line {line_no} | {label} | {snippet}")
        lines.append("")

    if total == 0:
        lines.append("No absolute path findings were detected.")
    else:
        lines.append(f"Summary: {total} findings in {files} files.")

    return lines


def save_report(lines: Iterable[str]) -> Path:
    out_dir = REPO_ROOT / "results" / "tools"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"path_audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    out_file.write_text("\n".join(lines), encoding="utf-8")
    return out_file


def main() -> int:
    args = parse_args()
    lines = build_report(include_results=args.include_results)
    print("\n".join(lines))

    if args.save:
        out_file = save_report(lines)
        print(f"\nSaved report: {out_file.relative_to(REPO_ROOT)}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
