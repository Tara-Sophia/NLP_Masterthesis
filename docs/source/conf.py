# -*- coding: utf-8 -*-
# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html
"""
Configuration file for the Sphinx documentation builder.
"""

import os
import sys

sys.path.insert(
    0, os.path.abspath(os.path.join("..", "..", "src", "data"))
)
sys.path.insert(
    1, os.path.abspath(os.path.join("..", "..", "src", "clf"))
)


# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "Masterthesis"
copyright = "2022, Hannha Petry, Tara-Sophia Tumbraegel, Florentin von Haugwitz"
author = (
    "Hannha Petry, Tara-Sophia Tumbraegel, Florentin von Haugwitz"
)
release = "0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["sphinx.ext.napoleon", "sphinx.ext.autodoc"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]
