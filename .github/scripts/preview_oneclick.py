# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Preview what a tutorial notebook looks like when launched in one click.

One-click launches on notebooks.amd.com serve each tutorial with the parts
marked by its one-click markers removed (see .github/ONECLICK_MARKERS.md).
This applies the markers to a notebook the same way the launch does and
writes the result as a notebook you can open in Jupyter:

- Cells tagged `oneclick-remove` and the lines between
  `<!-- oneclick:remove-start -->` and `<!-- oneclick:remove-end -->` are
  removed; `<!-- oneclick:insert ... -->` blocks are shown.
- Numbered headings ("## 3. Title", "### Step 4: Title") are renumbered,
  table-of-contents entries for removed sections are dropped, and links are
  repointed to the renumbered headings.

At launch, notebooks.amd.com also makes local web UIs (Gradio apps, servers
on 127.0.0.1) reachable through its proxy and prints a link to them. That
happens in the running notebook and isn't shown here.

Usage:
    python3 .github/scripts/preview_oneclick.py NOTEBOOK [-o OUTPUT] [--diff]

Writes <notebook name>.oneclick.ipynb in the current directory unless -o is
given. Standard library only.
"""

import argparse
import difflib
import json
import re
import sys
from collections import defaultdict, deque
from pathlib import Path

# The same fenced-code rules the marker check uses.
from check_oneclick_markers import track_fence as _track_fence

ONECLICK_REMOVE_TAG = "oneclick-remove"
_REMOVE_START_RE = re.compile(r"^\s*<!--\s*oneclick:remove-start\s*-->\s*$")
_REMOVE_END_RE = re.compile(r"^\s*<!--\s*oneclick:remove-end\s*-->\s*$")
_INSERT_START_RE = re.compile(r"^\s*<!--\s*oneclick:insert\s*$")
_INSERT_END = "-->"
_HTML_ANCHOR_RE = re.compile(r"""(?i)<a\b[^>]*?\b(?:id|name)\s*=\s*["']([^"']+)["']""")
_HEADING_RE = re.compile(r"^\s*(#{1,6})\s+(.*\S)\s*$")
_NUM_PREFIX_RE = re.compile(r"^\s*(\d+)\.\s+(.*)$")
_LABELED_ORDINAL_RE = re.compile(
    r"^\s*(?P<label>step|part|phase|section|chapter)\s+(?P<num>\d+)"
    r"(?P<sep>\s*[.:)\-]\s+)(?P<base>\S.*)$",
    re.I,
)
_LINK_RE = re.compile(r"\[([^\]]+)\]\(#([^)]+)\)")
_LINKLIST_ITEM_RE = re.compile(
    r"^(?P<indent>\s*)(?P<marker>(?:\d+[.)]|[-*+])\s+)?"
    r"\[(?P<text>[^\]]+)\]\(#(?P<anchor>[^)]+)\)(?P<trail>\s*)$"
)
_FENCE_RE = ("```", "~~~")
_CUT = object()


def cell_source(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def set_cell_source(cell, text):
    cell["source"] = text


def cell_tags(cell):
    return cell.get("metadata", {}).get("tags") or []


def parse_ordinal(text):
    """Parse a heading's leading ordinal into (number, template, base).

    `template` holds a single "{n}" placeholder for the number, with the
    original label and separator preserved, so renumbering can rewrite just the
    number ("Step 3: Install" -> "Step 1: Install"). Both the bare "2. Title"
    and labeled "Step 2: Title" forms are recognized. Headings without a
    recognized ordinal return (None, None, text)."""
    m = _LABELED_ORDINAL_RE.match(text)
    if m:
        template = f"{m.group('label')} {{n}}{m.group('sep')}"
        return int(m.group("num")), template, m.group("base").strip()
    m = _NUM_PREFIX_RE.match(text)
    if m:
        return int(m.group(1)), "{n}. ", m.group(2).strip()
    return None, None, text.strip()


def split_number(text):
    """Split an ordinal heading into ("2", "Title"), recognizing both the
    "2. Title" and "Step 2: Title" forms; unnumbered headings return
    (None, text). Used for anchor matching, which keys on the base title."""
    num, _, base = parse_ordinal(text)
    return (str(num) if num is not None else None), base


def github_slug(text):
    s = text.strip().lower()
    s = re.sub(r"[^\w\s-]", "", s)
    return s.replace(" ", "-")


def iter_heading_lines(source_text):
    """Yield (line_index, level, text) for headings outside fenced code blocks."""
    in_fence = False
    for i, line in enumerate(source_text.splitlines()):
        if line.lstrip().startswith(_FENCE_RE):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = _HEADING_RE.match(line)
        if m:
            yield i, len(m.group(1)), m.group(2).strip()


def first_heading(cell):
    if cell.get("cell_type") != "markdown":
        return None
    for _, level, text in iter_heading_lines(cell_source(cell)):
        return level, text
    return None


def collect_headings(cells):
    """List every heading with its GitHub-style anchor, in document order."""
    seen = {}
    out = []
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        for _, level, text in iter_heading_lines(cell_source(cell)):
            slug = github_slug(text)
            n = seen.get(slug, 0)
            seen[slug] = n + 1
            anchor = slug if n == 0 else f"{slug}-{n}"
            out.append({
                "level": level,
                "text": text,
                "base": split_number(text)[1],
                "anchor": anchor,
            })
    return out


def cell_label(cell, heading):
    """A short, human-readable identifier for a cell in report output."""
    if heading:
        return heading[1]
    cid = cell.get("id")
    if cid:
        return f"cell id={cid}"
    for line in cell_source(cell).splitlines():
        if line.strip():
            return line.strip()[:60]
    return "(untitled cell)"


def _close_gaps(out, source):
    """Join `out` (lines, with a _CUT where each block was removed) back into
    cell text. The blank lines on either side of a removed block become one,
    or none at the top or bottom of the cell; blank lines anywhere else, such
    as inside a code example, are left as written."""
    result = []
    i = 0
    while i < len(out):
        if out[i] is not _CUT:
            result.append(out[i])
            i += 1
            continue
        blank_before = False
        while result and not result[-1].strip():
            result.pop()
            blank_before = True
        j = i + 1
        blank_after = False
        while j < len(out) and (out[j] is _CUT or not out[j].strip()):
            blank_after = blank_after or out[j] is not _CUT
            j += 1
        if result and j < len(out) and (blank_before or blank_after):
            result.append("\n")
        i = j
    text = "".join(result)
    # A block removed from the end leaves the line before it ending in a
    # newline the cell itself did not end with.
    if text.endswith("\n") and not source.endswith("\n"):
        text = text[:-1]
    return text


def apply_comment_markers(source):
    """Drop remove-start/remove-end regions and unwrap insert blocks in one
    markdown cell's source. A marker inside a fenced code block is literal
    text, not a marker. Returns (new_source, removed_count, inserted_count);
    raises ValueError on an unbalanced or nested marker."""
    lines = source.splitlines(keepends=True)
    out = []
    state = None  # None, "remove", or "insert"
    opened_at = 0
    removed = inserted = 0
    fence = None
    for idx, line in enumerate(lines):
        bare = line.rstrip("\n")
        if state == "insert":
            # On the docs site an insert block is one HTML comment up to its
            # closing "-->", so nothing inside it, fences included, counts.
            if bare.strip() == _INSERT_END:
                state = None
                inserted += 1
            else:
                out.append(line)
            continue
        fence, fenced = _track_fence(fence, line)
        if not fenced:
            starts_remove = _REMOVE_START_RE.match(bare)
            if starts_remove or _INSERT_START_RE.match(bare):
                if state is not None:
                    raise ValueError(f"line {idx + 1}: oneclick marker opened inside "
                                     f"the oneclick:{state} block from line {opened_at}")
                state = "remove" if starts_remove else "insert"
                opened_at = idx + 1
                continue
            if _REMOVE_END_RE.match(bare):
                if state != "remove":
                    raise ValueError(f"line {idx + 1}: oneclick:remove-end without "
                                     f"a matching oneclick:remove-start")
                state = None
                removed += 1
                out.append(_CUT)
                continue
        if state != "remove":
            out.append(line)
    if state is not None:
        raise ValueError(f"line {opened_at}: oneclick:{state} block is never closed")

    if removed:
        return _close_gaps(out, source), removed, inserted
    return "".join(out), removed, inserted


def strip_with_markers(cells):
    """Remove exactly what the notebook's author marked (see the module
    docstring). Returns (kept_cells, report); raises ValueError naming the
    offending cell on an unbalanced marker."""
    report = {"cells_removed": 0, "blocks_removed": 0, "inserts": 0}
    kept = []
    for cell in cells:
        if ONECLICK_REMOVE_TAG in cell_tags(cell):
            report["cells_removed"] += 1
            continue
        if cell.get("cell_type") != "markdown":
            kept.append(cell)
            continue
        try:
            new_src, removed, inserted = apply_comment_markers(cell_source(cell))
        except ValueError as exc:
            raise ValueError(f"{cell_label(cell, first_heading(cell))}: {exc}") from None
        report["blocks_removed"] += removed
        report["inserts"] += inserted
        if not (removed or inserted):
            kept.append(cell)
        elif not new_src.strip():
            report["cells_removed"] += 1
        else:
            updated = dict(cell)
            set_cell_source(updated, new_src)
            kept.append(updated)
    return kept, report


def renumber_headings(cells):
    """Resequence ordinal headings ("2. Title" and "Step 2: Title") per level
    in document order, preserving each heading's label and separator. Returns a
    list of (old_text, new_text) for headings whose number changed."""
    counters = {}
    changes = []
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        lines = cell_source(cell).splitlines(keepends=True)
        in_fence = False
        modified = False
        for idx, line in enumerate(lines):
            if line.lstrip().startswith(_FENCE_RE):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            m = _HEADING_RE.match(line.rstrip("\n"))
            if not m:
                continue
            hashes, htext = m.group(1), m.group(2).strip()
            level = len(hashes)
            for k in [k for k in counters if k > level]:
                counters[k] = 0
            num, template, base = parse_ordinal(htext)
            if num is None:
                continue
            counters[level] = counters.get(level, 0) + 1
            new_num = counters[level]
            if num != new_num:
                prefix_ws = line[: line.index(hashes)]
                nl = "\n" if line.endswith("\n") else ""
                new_title = template.format(n=new_num) + base
                lines[idx] = f"{prefix_ws}{hashes} {new_title}{nl}"
                changes.append((htext, new_title))
                modified = True
        if modified:
            set_cell_source(cell, "".join(lines))
    return changes


def build_anchor_map(orig_headings, new_headings):
    """Map each original anchor to its new anchor, or None if the heading was
    removed. Headings are matched by (level, base-title), which is stable
    under renumbering."""
    buckets = defaultdict(deque)
    for h in new_headings:
        buckets[(h["level"], h["base"])].append(h["anchor"])
    amap = {}
    for h in orig_headings:
        key = (h["level"], h["base"])
        amap[h["anchor"]] = buckets[key].popleft() if buckets[key] else None
    return amap


def rebuild_link_lists(cells, amap):
    """Fix every navigation list of internal-anchor links -- numbered,
    bulleted, or bare, whether under a "Table of contents" heading or standing
    free in the intro. For each list: drop entries whose target section was
    removed, repoint entries to their renumbered anchors, and resequence any
    numbering. Returns the list of dropped entry titles.

    An entry is dropped when its anchor was a heading that got removed (amap
    maps it to None) or an explicit "<a id=...>" target that no longer exists
    in the kept cells."""
    valid = {h["anchor"] for h in collect_headings(cells)}
    valid |= {
        m.group(1)
        for cell in cells if cell.get("cell_type") == "markdown"
        for m in _HTML_ANCHOR_RE.finditer(cell_source(cell))
    }

    dropped = []
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        lines = cell_source(cell).splitlines(keepends=True)
        out = []
        in_fence = False
        changed = False
        i = 0
        while i < len(lines):
            line = lines[i]
            if line.lstrip().startswith(_FENCE_RE):
                in_fence = not in_fence
                out.append(line)
                i += 1
                continue
            if in_fence or not _LINKLIST_ITEM_RE.match(line.rstrip("\n")):
                out.append(line)
                i += 1
                continue

            # Extend the run over consecutive item lines, tolerating blank
            # lines between items so a spaced-out list stays one list.
            item_idxs = []
            j = i
            while j < len(lines):
                if _LINKLIST_ITEM_RE.match(lines[j].rstrip("\n")):
                    item_idxs.append(j)
                    j += 1
                elif lines[j].strip() == "":
                    k = j
                    while k < len(lines) and lines[k].strip() == "":
                        k += 1
                    if k < len(lines) and _LINKLIST_ITEM_RE.match(lines[k].rstrip("\n")):
                        j = k
                    else:
                        break
                else:
                    break

            # A lone internal link in prose is not a navigation list.
            if len(item_idxs) < 2:
                out.append(line)
                i += 1
                continue

            numbered = False
            for k in item_idxs:
                marker = _LINKLIST_ITEM_RE.match(lines[k].rstrip("\n")).group("marker")
                if marker and marker[0].isdigit():
                    numbered = True
                    break

            counter = 0
            last = item_idxs[-1]
            for k in range(i, last + 1):
                s = lines[k]
                m = _LINKLIST_ITEM_RE.match(s.rstrip("\n"))
                if not m:  # interior blank line, preserved as-is
                    out.append(s)
                    continue
                anchor = m.group("anchor")
                if anchor in amap:
                    new = amap[anchor]
                    if new is None:  # heading target was removed
                        dropped.append(m.group("text"))
                        changed = True
                        continue
                elif anchor not in valid:  # explicit anchor no longer exists
                    dropped.append(m.group("text"))
                    changed = True
                    continue
                else:
                    new = anchor
                counter += 1
                indent, trail, text = m.group("indent"), m.group("trail"), m.group("text")
                nl = "\n" if s.endswith("\n") else ""
                if numbered:
                    rebuilt = f"{indent}{counter}. [{text}](#{new}){trail}{nl}"
                else:
                    rebuilt = f"{indent}{m.group('marker') or ''}[{text}](#{new}){trail}{nl}"
                out.append(rebuilt)
                if rebuilt != s:
                    changed = True
            i = last + 1

        if changed:
            set_cell_source(cell, "".join(out))
    return dropped


def remap_links(cells, amap):
    """Repoint "[text](#anchor)" links to their new anchors. Returns
    (updated_count, dangling, text_flags)."""
    dangling = []
    text_flags = []

    def make_repl(counter):
        def repl(m):
            text, anchor = m.group(1), m.group(2)
            if anchor not in amap:
                return m.group(0)
            new = amap[anchor]
            if new is None:
                dangling.append((text, anchor))
                return m.group(0)
            if new != anchor:
                counter[0] += 1
                if re.search(r"\d", text) or re.match(
                    r"(?i)\s*(section|part|step|chapter)\b", text
                ):
                    text_flags.append((text, new))
                return f"[{text}](#{new})"
            return m.group(0)
        return repl

    counter = [0]
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        src = cell_source(cell)
        new_src = _LINK_RE.sub(make_repl(counter), src)
        if new_src != src:
            set_cell_source(cell, new_src)
    return counter[0], dangling, text_flags


def verify_internal_anchors(cells):
    """Confirm every internal "[text](#anchor)" link resolves to a heading in
    the given cells. Returns a list of (text, anchor) for broken links."""
    valid = {h["anchor"] for h in collect_headings(cells)}
    # An explicit "<a id="step1"></a>" is a link target in its own right,
    # alongside the anchors GitHub derives from heading text.
    valid |= {
        m.group(1)
        for cell in cells if cell.get("cell_type") == "markdown"
        for m in _HTML_ANCHOR_RE.finditer(cell_source(cell))
    }
    broken = []
    for cell in cells:
        if cell.get("cell_type") != "markdown":
            continue
        for m in _LINK_RE.finditer(cell_source(cell)):
            if m.group(2) not in valid:
                broken.append((m.group(1), m.group(2)))
    return broken


def cell_lines(cells):
    """Every cell's source as diffable lines, each cell under a header line."""
    lines = []
    for cell in cells:
        lines.append(f"──── {cell.get('cell_type', 'cell')} cell ────\n")
        lines += [line + "\n" for line in cell_source(cell).splitlines()]
    return lines


def parse_args(argv):
    parser = argparse.ArgumentParser(
        description="Write the one-click version of a tutorial notebook, with its "
        "one-click markers applied as notebooks.amd.com applies them."
    )
    parser.add_argument("notebook", type=Path, help="The tutorial notebook (.ipynb).")
    parser.add_argument("-o", "--output", type=Path,
                        help="Where to write the preview (default: "
                        "<notebook name>.oneclick.ipynb in the current directory).")
    parser.add_argument("--diff", action="store_true",
                        help="Also print what changed, cell by cell.")
    return parser.parse_args(argv)


def main(argv):
    args = parse_args(argv[1:])
    try:
        nb = json.loads(args.notebook.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        print(f"error: cannot read {args.notebook}: {exc}", file=sys.stderr)
        return 1
    cells = nb.get("cells", [])
    original = json.loads(json.dumps(cells))  # untouched copy for --diff
    orig_headings = collect_headings(cells)

    try:
        kept, report = strip_with_markers(cells)
    except ValueError as exc:
        print(f"error: {args.notebook}: {exc}", file=sys.stderr)
        print("Run .github/scripts/check_oneclick_markers.py on it for details.",
              file=sys.stderr)
        return 1
    renumbered = renumber_headings(kept)
    amap = build_anchor_map(orig_headings, collect_headings(kept))
    nav_dropped = rebuild_link_lists(kept, amap)
    remap_links(kept, amap)

    nb["cells"] = kept
    out = args.output or Path(args.notebook.stem + ".oneclick.ipynb")
    out.write_text(json.dumps(nb, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")

    if any(report.values()):
        print(f"Removed {report['cells_removed']} cell(s) and "
              f"{report['blocks_removed']} marked block(s); showed "
              f"{report['inserts']} insert block(s).")
    else:
        print("No one-click markers found; nothing is removed.")
    for old, new in renumbered:
        print(f"Renumbered heading: '{old}' -> '{new}'")
    for title in nav_dropped:
        print(f"Dropped table-of-contents entry: {title}")
    broken = verify_internal_anchors(kept)
    for text, anchor in broken:
        print(f"Warning: link '{text}' points at #{anchor}, which no longer exists")

    if args.diff:
        sys.stdout.writelines(difflib.unified_diff(
            cell_lines(original), cell_lines(kept),
            fromfile=str(args.notebook), tofile=str(out)))
    print(f"Wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
