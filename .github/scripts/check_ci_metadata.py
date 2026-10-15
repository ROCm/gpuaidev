# Copyright Advanced Micro Devices, Inc.
# SPDX-License-Identifier: MIT
"""Validate the CI requirements declared in tutorial notebooks.

The notebook regression runner executes every tutorial on AMD GPUs. Each
notebook declares what it needs, and the runner picks a machine that has it:

- Notebook metadata `ci`: hardware, GPU count and memory, host memory, disk,
  capabilities, environment, Docker image, CI parameter values, and notebooks
  that must run first.
- Cell tag `ci-skip`: a code cell that cannot run unattended (a browser UI,
  an interactive loop, a foreground GUI server).
- Cell tag `parameters`: the cell whose variables `ci.parameters` overrides.

Every tutorial under docs/notebooks must declare at least `hardware` and
`gpus`, so what it runs on is stated rather than assumed. This checks that,
and that the metadata is well formed and refers to things that exist.

Usage:
    python check_ci_metadata.py [notebook.ipynb ...]

With no arguments, checks every tracked .ipynb file in the repository.
"""

import ast
import json
import os
import re
import subprocess
import sys

GUIDE = ".github/CI_METADATA.md"
NOTEBOOK_ROOT = "docs/notebooks"

SKIP_TAG = "ci-skip"
PARAMETERS_TAG = "parameters"

# Products, architectures and families the runner can match against a machine.
HARDWARE = {
    "mi300x", "mi308x", "mi325x", "mi350x", "mi355x",
    "rx7900xtx", "w7900", "rx9070xt",
    "gfx942", "gfx950", "gfx1100", "gfx1201",
    "instinct", "radeon", "cdna3", "cdna4", "rdna3", "rdna4",
}
POSITIVE_INTS = ("gpus", "gpu_memory_gb", "memory_gb", "disk_gb")
REQUIRED_KEYS = ("hardware", "gpus")
KNOWN_KEYS = {"hardware", "capabilities", "env", "image", "parameters", "depends_on",
              *POSITIVE_INTS}
NAME_RE = re.compile(r"^[a-z0-9][a-z0-9-]*$")
IMAGE_RE = re.compile(r"^[a-z0-9._-]+/[a-z0-9._-]+$")


def cell_source(cell):
    src = cell.get("source", "")
    return "".join(src) if isinstance(src, list) else src


def cell_tags(cell):
    return cell.get("metadata", {}).get("tags", []) or []


def assigned_names(source):
    """Names assigned at the top level of a parameters cell."""
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    names = set()
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else (
            [node.target] if isinstance(node, (ast.AnnAssign, ast.AugAssign)) else [])
        names |= {t.id for t in targets if isinstance(t, ast.Name)}
    return names


def check_metadata(path, ci, parameter_cells):
    errors = []
    if not isinstance(ci, dict):
        return [f"{path}: metadata 'ci' must be an object"]
    for key in sorted(set(ci) - KNOWN_KEYS):
        errors.append(f"{path}: unknown key ci.{key} (expected one of "
                      f"{', '.join(sorted(KNOWN_KEYS))})")
    for key in POSITIVE_INTS:
        if key in ci and not (isinstance(ci[key], int) and not isinstance(ci[key], bool)
                              and ci[key] > 0):
            errors.append(f"{path}: ci.{key} must be a positive integer")

    hardware = ci.get("hardware", [])
    if not (isinstance(hardware, list) and hardware) and "hardware" in ci:
        errors.append(f"{path}: ci.hardware must be a non-empty list")
    for name in hardware if isinstance(hardware, list) else []:
        if name not in HARDWARE:
            errors.append(f"{path}: unknown hardware {name!r} in ci.hardware")

    capabilities = ci.get("capabilities", [])
    if not isinstance(capabilities, list) or not all(
            isinstance(c, str) and NAME_RE.match(c) for c in capabilities):
        errors.append(f"{path}: ci.capabilities must be a list of lowercase names")

    env = ci.get("env", {})
    if not isinstance(env, dict) or not all(
            isinstance(k, str) and isinstance(v, str) for k, v in env.items()):
        errors.append(f"{path}: ci.env must map names to string values")

    if "image" in ci and not (isinstance(ci["image"], str) and IMAGE_RE.match(ci["image"])):
        errors.append(f"{path}: ci.image must be a Docker repository like "
                      f"'rocm/pytorch', without a tag")

    parameters = ci.get("parameters", {})
    if not isinstance(parameters, dict):
        errors.append(f"{path}: ci.parameters must be an object")
    elif parameters:
        if len(parameter_cells) != 1:
            errors.append(f"{path}: ci.parameters needs exactly one cell tagged "
                          f"{PARAMETERS_TAG!r} (found {len(parameter_cells)})")
        else:
            names = assigned_names(parameter_cells[0])
            if names is None:
                errors.append(f"{path}: the {PARAMETERS_TAG!r} cell is not valid Python")
            for key in sorted(set(parameters) - (names or set())):
                errors.append(f"{path}: ci.parameters.{key} is not assigned in the "
                              f"{PARAMETERS_TAG!r} cell")

    depends_on = ci.get("depends_on", [])
    if not isinstance(depends_on, list):
        errors.append(f"{path}: ci.depends_on must be a list")
    for dep in depends_on if isinstance(depends_on, list) else []:
        if not os.path.isfile(os.path.join(NOTEBOOK_ROOT, str(dep))):
            errors.append(f"{path}: ci.depends_on {dep!r} is not a notebook under "
                          f"{NOTEBOOK_ROOT}")
    return errors


def check_notebook(path):
    """Return a list of "path: message" strings for one notebook."""
    try:
        with open(path, encoding="utf-8") as f:
            nb = json.load(f)
    except (OSError, ValueError) as exc:
        return [f"{path}: cannot read notebook: {exc}"]

    errors = []
    parameter_cells = []
    for i, cell in enumerate(nb.get("cells", [])):
        tags = cell_tags(cell)
        for tag in tags:
            if tag.startswith("ci-") and tag != SKIP_TAG:
                errors.append(f"{path}: cell {i}: unknown tag {tag!r} (expected {SKIP_TAG!r})")
        if SKIP_TAG in tags and cell.get("cell_type") != "code":
            errors.append(f"{path}: cell {i}: {SKIP_TAG!r} only applies to code cells")
        if PARAMETERS_TAG in tags:
            parameter_cells.append(cell_source(cell))

    ci = nb.get("metadata", {}).get("ci")
    if ci is None:
        if is_tutorial(path):
            errors.append(f"{path}: missing notebook metadata 'ci' (declare at least "
                          f"{' and '.join(REQUIRED_KEYS)})")
        return errors
    errors += check_metadata(path, ci, parameter_cells)
    if is_tutorial(path) and isinstance(ci, dict):
        for key in REQUIRED_KEYS:
            if key not in ci:
                errors.append(f"{path}: ci.{key} is required for tutorials under "
                              f"{NOTEBOOK_ROOT}")
    return errors


def is_tutorial(path):
    """A notebook the regression runner tests (under docs/notebooks)."""
    return os.path.normpath(path).startswith(os.path.normpath(NOTEBOOK_ROOT) + os.sep)


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
    if errors:
        print(f"See {GUIDE} for how to declare CI requirements.")
    return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
