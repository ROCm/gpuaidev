# One-click markers

Some tutorials can be launched in one click on
[AMD Developer Cloud](https://notebooks.amd.com) from the "Open in AMD
Developer Cloud" badge. The launched notebook runs inside a ready-made
container, so the steps that set up that environment (pulling and running a
Docker image, installing and starting Jupyter, and so on) are removed from it
first.

You tell the one-click launch what to remove by adding **markers** to the
notebook. The markers are invisible on the documentation site and in Jupyter,
so a single notebook serves the site, GitHub, local Jupyter, and the
one-click launch alike.

Every notebook with the badge must use markers. The
[One-click markers](workflows/oneclick-markers.yml) check validates them on
every pull request that changes a notebook.

## Markers

| To do this in the one-click notebook | Use |
|---|---|
| Remove a whole cell (markdown or code) | The cell tag `oneclick-remove` |
| Remove some lines of a markdown cell | `<!-- oneclick:remove-start -->` and `<!-- oneclick:remove-end -->` around them |
| Add text to a markdown cell | `<!-- oneclick:insert` and `-->` around it |
| Add a code cell | The cell tags `oneclick-only` and `remove-cell` |

To add a cell tag in JupyterLab, select the cell and open the **Property
Inspector** (the gear icon in the right sidebar). In VS Code, use **Add Cell
Tag** from the cell's "..." menu.

### Remove a whole cell

Tag every cell that exists only for environment setup with `oneclick-remove`:
the "Prepare the training environment" section, a `!pip uninstall` of a
package the image already provides, an `!amd-smi` sanity check, and so on.

```json
"metadata": {
  "tags": ["oneclick-remove"]
}
```

Code cells can only be removed whole. If a code cell mixes setup with
tutorial steps, split it.

### Remove part of a markdown cell

```markdown
This tutorial explores how to use vision-language models for OCR.

<!-- oneclick:remove-start -->
## 1. Environment setup with Docker and ROCm

Launch the Docker container. From your host machine, run this command:
...
<!-- oneclick:remove-end -->

## 3. Installing dependencies
```

### Add text

The text inside an insert block is hidden on the site and shown only in the
one-click notebook. Combine it with a remove block to reword a sentence:

```markdown
<!-- oneclick:remove-start -->
Run the following commands inside the Jupyter notebook running within the Docker container:
<!-- oneclick:remove-end -->
<!-- oneclick:insert
Run the following commands in this notebook:
-->
```

### Add a code cell

Tag the cell with both `oneclick-only` and `remove-cell`. `remove-cell` hides
it on the documentation site, but it **still shows on GitHub and runs for
anyone using the notebook locally**, so make sure it is harmless there.

## Rules for placing comment markers

Each rule keeps the documentation site rendering exactly as it does without
the markers. The check enforces all of them.

1. **Put each marker on its own line.** Text after `-->` on the same line is
   rendered as raw HTML on the site instead of as markdown.
2. **Don't put a marker inside a paragraph.** A marker line between two lines
   of the same paragraph splits it into two paragraphs on the site. Place
   markers at paragraph boundaries.
3. **Inside a bullet list, indent the marker** to the list item's content
   (two spaces for `* `, three for `1. `). A marker at column 0 ends the list
   and starts a new one, which changes the spacing on the site:

   ```markdown
   * **ROCm 6.3**: Install and verify ROCm.

     <!-- oneclick:remove-start -->

   * **Docker**: Ensure Docker is installed and configured correctly.

     <!-- oneclick:remove-end -->
   ```

4. **Wrap the "Open in AMD Developer Cloud" badge** in a remove block. In a
   launched notebook it would only relaunch itself.
5. **Markers inside fenced code blocks are ignored**, so you can show them in
   examples like the ones on this page.

## What you don't need to mark

The one-click launch takes care of these itself:

- **Navigation links** to removed sections are dropped from the table of
  contents, so there's no need to mark them (marking links that sit in one
  paragraph would break rule 2).
- **Numbered headings** ("## 3. Installing dependencies", "### Step 4: ...")
  are renumbered after removal, and links to them are updated.
- **Gradio apps and local servers** are made reachable through the
  notebooks.amd.com proxy, and a link to them is printed.

## Check your markers

```bash
python3 .github/scripts/check_oneclick_markers.py docs/notebooks/inference/ocr_vllm.ipynb
```

With no arguments, the script checks every notebook in the repository.
