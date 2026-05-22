#!/usr/bin/env python3
"""compact_status.py — emit a readable digest of a long autoresearch STATUS.md.

The cnn-auto-cross-push campaign's STATUS.md grew to ~113 KB with most of the
bulk in the per-iteration table: each row is one massive line carrying the
iteration's calibration ledger, lesson tracking, and prose. This tool keeps the
last K substantive entries verbatim, collapses older rows to one-line headlines,
and passes through the structural sections (Headline / Protocol / Open
questions). Result: a STATUS_DIGEST.md that a human can actually read on
cold-read.

Usage:
    python compact_status.py <STATUS.md> [--keep-last N] [--out PATH]

Defaults: --keep-last 10, --out <input-stem>_DIGEST.md alongside the input.
"""

import argparse
import re
from pathlib import Path
from typing import List, Tuple


HEADLINE_RE = re.compile(r"^##\s+(.+?)\s*$")
TABLE_HEADER_RE = re.compile(r"^\|\s*iter\s*\|")
TABLE_SEP_RE = re.compile(r"^\|[\s\-:|]+\|\s*$")
TABLE_ROW_RE = re.compile(
    r"^\|\s*(?P<iter>\S+?)\s*\|"
    r"\s*(?P<commit>.*?)\s*\|"
    r"\s*(?P<metric>.*?)\s*\|"
    r"\s*(?P<guard>.*?)\s*\|"
    r"\s*(?P<status>.*?)\s*\|"
    r"\s*(?P<desc>.+?)\s*\|\s*$"
)


def _short_desc(desc: str, max_chars: int = 180) -> str:
    """Pick the first sentence-ish chunk of the description cell."""
    # Many rows lead with "(this iter, ...)" or "(prior-at-iter-N, ...)" — try
    # to keep that prefix and one short headline phrase.
    desc = desc.replace("**", "").strip()
    # First, prefer up to the first occurrence of "):" which usually closes the
    # classification parenthetical.
    end = desc.find("):")
    if 0 < end < max_chars * 2:
        return desc[: end + 2].strip()
    # Fallback: first sentence or first N chars.
    cut = re.split(r"(?<=[.!?])\s+", desc, maxsplit=1)
    head = cut[0]
    if len(head) > max_chars:
        head = head[: max_chars - 1].rstrip() + "…"
    return head


def _split_sections(lines: List[str]) -> List[Tuple[str, List[str]]]:
    """Split markdown into (heading, body_lines) sections at level-2 headings."""
    sections: List[Tuple[str, List[str]]] = []
    current_title = ""
    current_body: List[str] = []
    for line in lines:
        m = HEADLINE_RE.match(line)
        if m:
            if current_title or current_body:
                sections.append((current_title, current_body))
            current_title = m.group(1)
            current_body = []
        else:
            current_body.append(line)
    if current_title or current_body:
        sections.append((current_title, current_body))
    return sections


def _compact_iteration_table(
    body: List[str], keep_last: int,
) -> List[str]:
    """Find the iteration table in `body` and shorten old rows."""
    out: List[str] = []
    in_table = False
    rows: List[str] = []
    table_header_lines: List[str] = []
    table_idx: int = -1

    for i, line in enumerate(body):
        if not in_table and TABLE_HEADER_RE.match(line):
            in_table = True
            table_header_lines = [line]
            table_idx = i
            continue
        if in_table and TABLE_SEP_RE.match(line):
            table_header_lines.append(line)
            continue
        if in_table:
            if line.startswith("|") and TABLE_ROW_RE.match(line):
                rows.append(line)
            elif line.strip() == "" or not line.startswith("|"):
                # Table ended.
                in_table = False
                # Emit everything up to (but not including) the table header.
                out.extend(body[: table_idx])
                # Emit header.
                out.extend(table_header_lines)
                # Sort rows by iter (descending) to keep most-recent first.
                def _row_key(r: str) -> Tuple[int, int]:
                    m = TABLE_ROW_RE.match(r)
                    if not m:
                        return (-1, 0)
                    raw = m.group("iter")
                    digits = re.findall(r"\d+", raw)
                    n = int(digits[0]) if digits else -1
                    # Bias non-pure-int (e.g., "108-Q6ON") to be just under
                    # their numeric parent.
                    return (n, 0 if raw.isdigit() else -1)
                rows_sorted = sorted(rows, key=_row_key, reverse=True)
                keep = rows_sorted[:keep_last]
                older = rows_sorted[keep_last:]
                # Emit kept rows verbatim.
                out.extend(keep)
                # Collapse older rows to one-liners.
                if older:
                    out.append("")
                    out.append(
                        f"<!-- {len(older)} older rows collapsed by "
                        f"compact_status.py — one-liners follow -->"
                    )
                    out.append("")
                    out.append("| iter | status | summary |")
                    out.append("|-----:|--------|---------|")
                    for r in older:
                        m = TABLE_ROW_RE.match(r)
                        if not m:
                            continue
                        it = m.group("iter")
                        st = m.group("status").replace("|", "/")[:24]
                        short = _short_desc(m.group("desc"), max_chars=180)
                        short = short.replace("|", "/")
                        out.append(f"| {it} | {st} | {short} |")
                # Emit remainder of body after the table.
                out.append(line)
                # Track that we've handled everything through index `i`.
                # We'll just append the rest below.
                out.extend(body[i + 1 :])
                return out
    if in_table:
        # File ended inside the table — fall through and emit as-is.
        return body
    return body


def compact(text: str, keep_last: int = 10) -> str:
    lines = text.splitlines()
    sections = _split_sections(lines)
    out: List[str] = []
    for title, body in sections:
        if title:
            out.append(f"## {title}")
        if title.lower().startswith("per-iteration"):
            out.extend(_compact_iteration_table(body, keep_last))
        else:
            out.extend(body)
    return "\n".join(out).rstrip() + "\n"


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("path", type=Path, help="Path to STATUS.md to compact.")
    p.add_argument("--keep-last", type=int, default=10,
                   help="Keep the N most recent substantive rows verbatim "
                        "(default 10). Older rows collapse to one-liners.")
    p.add_argument("--out", type=Path, default=None,
                   help="Output path. Default: <stem>_DIGEST.md alongside input.")
    args = p.parse_args()

    src = args.path.read_text(encoding="utf-8")
    dst = args.out or args.path.with_name(args.path.stem + "_DIGEST.md")
    digest = compact(src, keep_last=args.keep_last)
    dst.write_text(digest, encoding="utf-8")
    src_kb = len(src) / 1024
    dst_kb = len(digest) / 1024
    print(f"wrote {dst}")
    print(f"  input : {src_kb:7.1f} KB ({len(src.splitlines())} lines)")
    print(f"  output: {dst_kb:7.1f} KB ({len(digest.splitlines())} lines)")
    print(f"  ratio : {dst_kb / src_kb:.1%} of input size")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
