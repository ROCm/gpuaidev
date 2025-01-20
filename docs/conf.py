"""Configuration file for the Sphinx documentation builder."""
import os
import sys

os.system("mkdir -p ../_readthedocs/html/downloads")

latex_engine = "xelatex"
latex_elements = {
    "fontpkg": r"""
\usepackage{tgtermes}
\usepackage{tgheros}
\renewcommand\ttdefault{txtt}
"""
}

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "rocm.docs.amd.com")
html_context = {}
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

# configurations for PDF output by Read the Docs
project = "AI Developer Hub Documentation"
version = "0.0.1"
release = version
setting_all_article_info = True
all_article_info_os = ["linux", "windows"]
all_article_info_author = ""
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',            # For integrating Jupyter notebooks
    'sphinx.ext.mathjax', # For rendering math expressions
    'sphinx_rtd_theme',   # For the Read the Docs theme
]

# Paths for templates and static files
templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store', 'sphinx/venv']
nb_execution_mode = "off"
nb_execution_excludepatterns = [
    "notebooks/pretrain/*.ipynb",  # Adjust the path to match notebooks you want to exclude
]
nb_execution_allow_errors = True
nb_render_text_lexer = "none"  # Disables syntax highlighting errors


nb_execution_mode = "off"
nb_execution_excludepatterns = [
    "notebooks/pretrain/*.ipynb",
    "notebooks/fine-tune/*.ipynb",
    "notebooks/inference/*.ipynb",  # Adjust the path to match notebooks you want to exclude
]
nb_execution_allow_errors = True
nb_render_text_lexer = "none"  # Disables syntax highlighting errors

source_suffix = ['.md', '.rst', '.ipynb']  

external_toc_path = "./sphinx/_toc.yml"

extensions = ["rocm_docs", "sphinx_reredirects", "sphinx_sitemap"]

external_projects_current_project = "rocm"

html_baseurl = os.environ.get("READTHEDOCS_CANONICAL_URL", "https://rocm-stg.amd.com/")
html_context = {}
if os.environ.get("READTHEDOCS", "") == "True":
    html_context["READTHEDOCS"] = True

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm-docs-home"}

html_static_path = ["sphinx/static/css"]
html_css_files = ["rocm_custom.css", "rocm_rn.css"]

html_title = "ROCm Documentation"

html_theme_options = {"link_main_doc": False}

redirects = {"reference/openmp/openmp": "../../about/compatibility/openmp.html"}

numfig = False

# Use Read the Docs theme
# html_theme = 'sphinx_rtd_theme'

# # Paths for custom static files (e.g., CSS, JS)
# html_static_path = ['_static']
