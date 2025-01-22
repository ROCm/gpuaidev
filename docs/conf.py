"""Configuration file for the Sphinx documentation builder."""
import os
import sys

project = 'gpuaidev-docs'
version = "0.0.1"
release = version
html_title = ""
author = "Advanced Micro Devices, Inc."
copyright = "Copyright (c) 2025 Advanced Micro Devices, Inc. All rights reserved."

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'myst_nb',            # For integrating Jupyter notebooks
    'sphinx.ext.mathjax', # For rendering math expressions
    'rocm_docs'           # For ROCm Docs theme
]
external_toc_path = "./sphinx/_toc.yml"
external_projects_current_project = "gpuaidev-docs"

html_theme = "rocm_docs_theme"
html_theme_options = {"flavor": "rocm"}


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



source_suffix = ['.md', '.ipynb', ".rst"]
suppress_warnings = ['app.add_source_parser']

# # Paths for custom static files (e.g., CSS, JS)
# html_static_path = ['_static']
