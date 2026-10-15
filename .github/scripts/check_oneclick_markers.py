# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Validate the one-click markers in tutorial notebooks.

One-click launches on notebooks.amd.com serve each tutorial with its local
environment setup removed. Authors mark what to remove (or add) in the
notebook itself; the markers are invisible on the docs site and in Jupyter:

- Cell tag `oneclick-remove`: the cell is dropped.
- Markdown lines `<!-- oneclick:remove-start -->` ... `<!-- oneclick:remove-end -->`:
  the lines between them are dropped.
- Markdown lines `<!-- oneclick:insert` ... `-->`: the content is shown only
  in the one-click notebook.
- Cell tag `oneclick-only` (with `remove-cell`, so the site hides it): the
  cell is kept as-is and only runs in the one-click notebook.

This checks that every marker is well formed, so the one-click build can
apply it, and that it is placed where it does not change how the docs site
renders the notebook. It also requires a notebook carrying the "Open in AMD
Developer Cloud" badge to use markers at all.

Usage:
    python check_oneclick_markers.py [notebook.ipynb ...]

With no arguments, checks every tracked .ipynb file in the repository.
"""

import json
import re
import subprocess
import sys

REMOVE_TAG = "oneclick-remove"
ONLY_TAG = "oneclick-only"
KNOWN_TAGS = {REMOVE_TAG, ONLY_TAG}

REMOVE_START_RE = re.compile(r"^\s*<!--\s*oneclick:remove-start\s*-->\s*$")
REMOVE_END_RE = re.compile(r"^\s*<!--\s*oneclick:remove-end\s*-->\s*$")
INSERT_START_RE = re.compile(r"^\s*<!--\s*oneclick:insert\s*$")
INSERT_END_RE = re.compile(r"^\s*-->\s*$")
ANY_MARKER_RE = re.compile(r"oneclick:", re.I)

FENCE = ("```", "~~~")
LIST_ITEM_RE = re.compile(r"^(?:[-*+]|\d+[.)])\s+\S")
HEADING_RE = re.compile(r"^\s{0,3}#{1,6}\s")
BADGE_RE = re.compile(r"\]\(https?://notebooks\.amd\.com/tutorials/[^)]*\)")


def cell_source(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def cell_tags(cell):
    return cell.get("metadata", {}).get("tags") or []


def neighbor(lines, hidden, idx, step):
    """The nearest line from `idx` in direction `step` that the docs site
    renders (not blank, not in `hidden`), as (line, blank_between).
    blank_between is True if a blank line sits between `idx` and it. Returns
    (None, blank_between) at either end."""
    blank = False
    j = idx + step
    while 0 <= j < len(lines):
        if not lines[j].strip():
            blank = True
        elif j not in hidden:
            return lines[j], blank
        j += step
    return None, blank


def is_text(line):
    """A line that continues a paragraph: not a heading, list item, fence,
    HTML block, table row, or blockquote."""
    s = line.strip()
    return bool(s) and not (
        HEADING_RE.match(line)
        or LIST_ITEM_RE.match(s)
        or s.startswith(FENCE + ("<", "|", ">"))
    )


def check_placement(lines, hidden, idx):
    """Problems with where the marker at `idx` sits, from how CommonMark
    parses an HTML comment line: it interrupts a paragraph, and at column 0
    it ends an open list."""
    problems = []
    prev, blank_before = neighbor(lines, hidden, idx, -1)
    nxt, blank_after = neighbor(lines, hidden, idx, +1)
    if (prev is not None and nxt is not None and not blank_before and not blank_after
            and is_text(prev) and is_text(nxt)):
        problems.append("sits inside a paragraph, which splits it in two on the "
                        "docs site; move the marker to a paragraph boundary")
    indent = len(lines[idx]) - len(lines[idx].lstrip())
    if (indent < 2 and nxt is not None and LIST_ITEM_RE.match(nxt)
            and prev is not None and (LIST_ITEM_RE.match(prev) or prev[:1].isspace())):
        problems.append("splits a list in two on the docs site; indent the marker "
                        "to the list item's content (for example, two spaces for '* ')")
    return problems


def check_markdown(source, cell_removed=False):
    """Yield (line_number, message) for every marker problem in a markdown
    cell. `cell_removed` is True for a cell tagged for removal as a whole."""
    lines = source.splitlines()
    # Pass 1: pair up the markers, and collect the lines the docs site does
    # not render (the markers themselves and whole insert blocks).
    hidden = set()
    markers = []  # line indexes of the markers to check for placement
    state = None  # None, "remove", or "insert"
    opened_at = 0
    in_fence = False
    for idx, line in enumerate(lines):
        n = idx + 1
        if state == "insert":
            hidden.add(idx)
            if INSERT_END_RE.match(line):
                state = None
            continue
        if line.lstrip().startswith(FENCE):
            in_fence = not in_fence
            continue
        if in_fence:
            continue

        # In a launched notebook the badge would only relaunch itself, and the
        # one-click build removes only what is marked.
        if BADGE_RE.search(line) and state != "remove" and not cell_removed:
            yield n, ("the 'Open in AMD Developer Cloud' badge must be inside an "
                      "oneclick:remove block")
        if REMOVE_START_RE.match(line) or INSERT_START_RE.match(line):
            if state is not None:
                yield n, f"marker opened inside the oneclick:{state} block from line {opened_at}"
                continue
            state = "remove" if REMOVE_START_RE.match(line) else "insert"
            opened_at = n
        elif REMOVE_END_RE.match(line):
            if state != "remove":
                yield n, "oneclick:remove-end without a matching oneclick:remove-start"
                continue
            state = None
        elif ANY_MARKER_RE.search(line):
            yield n, ("unrecognized oneclick marker; each marker must be one of "
                      "'<!-- oneclick:remove-start -->', '<!-- oneclick:remove-end -->', "
                      "or '<!-- oneclick:insert', alone on its line")
            continue
        else:
            continue
        hidden.add(idx)
        markers.append(idx)

    if state is not None:
        yield opened_at, f"oneclick:{state} block is never closed"
        return

    # Pass 2: check each marker against the lines the site does render.
    for idx in markers:
        for problem in check_placement(lines, hidden, idx):
            yield idx + 1, problem


def check_notebook(path):
    """Return a list of "path: cell N: message" strings for one notebook."""
    try:
        with open(path, encoding="utf-8") as f:
            cells = json.load(f).get("cells", [])
    except (OSError, ValueError) as exc:
        return [f"{path}: cannot read notebook: {exc}"]

    errors = []
    uses_markers = False
    has_badge = False
    for i, cell in enumerate(cells):
        where = f"{path}: cell {i}"
        tags = cell_tags(cell)
        src = cell_source(cell)

        for tag in tags:
            if tag.startswith("oneclick") and tag not in KNOWN_TAGS:
                errors.append(f"{where}: unknown tag {tag!r} (expected one of "
                              f"{', '.join(sorted(KNOWN_TAGS))})")
        if KNOWN_TAGS & set(tags):
            uses_markers = True
        if REMOVE_TAG in tags and ONLY_TAG in tags:
            errors.append(f"{where}: tagged both {REMOVE_TAG!r} and {ONLY_TAG!r}")
        if ONLY_TAG in tags and "remove-cell" not in tags:
            errors.append(f"{where}: {ONLY_TAG!r} cell also needs the 'remove-cell' "
                          f"tag, or it shows on the docs site")

        if cell.get("cell_type") == "markdown":
            if ANY_MARKER_RE.search(src):
                uses_markers = True
            if BADGE_RE.search(src):
                has_badge = True
            for line, message in check_markdown(src, cell_removed=REMOVE_TAG in tags):
                errors.append(f"{where}, line {line}: {message}")
        elif re.search(r"<!--\s*oneclick:", src):
            errors.append(f"{where}: oneclick comment markers only work in markdown "
                          f"cells; tag the whole {cell.get('cell_type')} cell instead")

    if has_badge and not uses_markers:
        errors.append(f"{path}: has an 'Open in AMD Developer Cloud' badge but no "
                      f"oneclick markers; mark its environment setup for removal")
    return errors


def tracked_notebooks():
    out = subprocess.run(["git", "ls-files", "*.ipynb"], capture_output=True,
                         text=True, check=True).stdout
    return [p for p in out.splitlines() if p]


def main(argv):
    paths = argv[1:] or tracked_notebooks()
    errors = [e for p in paths for e in check_notebook(p)]
    for e in errors:
        print(e)
    print(f"Checked {len(paths)} notebook(s): "
          f"{'no problems' if not errors else f'{len(errors)} problem(s)'}")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
