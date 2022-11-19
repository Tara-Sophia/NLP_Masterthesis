# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join("..", "..")))
# sys.path.insert(1, os.path.abspath(os.path.join("..", "..", "src")))
# sys.path.insert(
#     2, os.path.abspath(os.path.join("..", "..", "src", "data"))
# )
# sys.path.insert(
#     3, os.path.abspath(os.path.join("..", "..", "src", "clf"))
# )
# sys.path.insert(
#     4, os.path.abspath(os.path.join("..", "..", "src", "stt"))
# )

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Masterthesis"
copyright = "2022, Hannha Petry, Tara-Sophia Tumbraegel, Florentin von Haugwitz"
author = (
    "Hannha Petry, Tara-Sophia Tumbraegel, Florentin von Haugwitz"
)
release = "1.0.0"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    "sphinx.ext.napoleon",
]

templates_path = ["_templates"]
exclude_patterns = [
    "_build",
    "Thumbs.db",
    ".DS_Store",
    "src/data/streamlit/",
]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
