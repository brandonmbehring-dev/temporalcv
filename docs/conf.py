"""
Sphinx configuration for temporalcv documentation.

This configuration follows best practices from NumPy, SciPy, and scikit-learn.
"""

import os
import sys

# Add source to path for autodoc
sys.path.insert(0, os.path.abspath("../src"))

# -- Project information -----------------------------------------------------

project = "temporalcv"
copyright = "2025, Brandon Behring"
author = "Brandon Behring"
release = "1.0.0"

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.autodoc",  # Auto-generate from docstrings
    "sphinx.ext.napoleon",  # NumPy docstring parsing (MUST come before typehints)
    "sphinx_autodoc_typehints",  # Better type hint rendering
    "sphinx.ext.intersphinx",  # Link to numpy/scipy/sklearn docs
    "sphinx.ext.viewcode",  # Source code links
    "sphinx.ext.mathjax",  # Math rendering
    "myst_parser",  # Keep existing .md files
    "sphinx_copybutton",  # Copy button for code blocks
    "sphinxcontrib.mermaid",  # Flowchart diagrams
    "sphinx_design",  # Cards, grids, tabs for better layout
    "sphinx_gallery.gen_gallery",  # Auto-generate example gallery
]

# Napoleon settings (enables auto-linking in See Also)
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = True
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = True
napoleon_use_admonition_for_examples = True
napoleon_use_admonition_for_notes = True
napoleon_use_admonition_for_references = False
napoleon_use_ivar = False
napoleon_use_param = True
napoleon_use_rtype = True
napoleon_preprocess_types = True  # Auto-link types in Params/Returns
napoleon_type_aliases = {
    "ArrayLike": "numpy.typing.ArrayLike",
    "NDArray": "numpy.ndarray",
}

# Autodoc settings
autodoc_typehints = "description"
autodoc_member_order = "bysource"
autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": False,
}

# Type hints settings
typehints_fully_qualified = False
always_document_param_types = True
typehints_document_rtype = True

# Intersphinx (enables external package linking)
intersphinx_mapping = {
    "python": ("https://docs.python.org/3", None),
    "numpy": ("https://numpy.org/doc/stable/", None),
    "scipy": ("https://docs.scipy.org/doc/scipy/", None),
    "sklearn": ("https://scikit-learn.org/stable/", None),
    "pandas": ("https://pandas.pydata.org/docs/", None),
}

# MyST parser settings (for .md files)
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "dollarmath",
    "fieldlist",
    "substitution",
    "tasklist",
]
myst_heading_anchors = 3

# Source settings
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store", "audits"]

# -- Options for HTML output -------------------------------------------------

html_theme = "sphinx_rtd_theme"
html_theme_options = {
    "navigation_depth": 4,
    "collapse_navigation": False,
    "sticky_navigation": True,
    "includehidden": True,
    "titles_only": False,
}
html_static_path = ["_static"]
html_css_files = []

# -- Options for LaTeX output ------------------------------------------------

latex_elements = {
    "papersize": "letterpaper",
    "pointsize": "10pt",
}

# -- Copybutton settings -----------------------------------------------------

copybutton_prompt_text = r">>> |\.\.\. |\$ |In \[\d*\]: | {2,5}\.\.\.: | {5,8}: "
copybutton_prompt_is_regexp = True

# -- Mermaid settings --------------------------------------------------------

mermaid_version = "10.6.1"  # Stable version with good flowchart support
mermaid_init_js = """
mermaid.initialize({
    startOnLoad: true,
    theme: 'default',
    flowchart: {
        useMaxWidth: true,
        htmlLabels: true,
        curve: 'basis'
    }
});
"""

# -- Suppress warnings -------------------------------------------------------

suppress_warnings = [
    "myst.header",
    "toc.not_included",  # Internal planning/knowledge docs not in toctree
    "docutils",  # Transition issues in internal docs
    "ref.python",  # Duplicate object warnings from autodoc
]

# -- Sphinx Gallery settings -------------------------------------------------

sphinx_gallery_conf = {
    # Path to example scripts
    "examples_dirs": ["../examples"],
    # Where to save gallery generated output
    "gallery_dirs": ["auto_examples"],
    # Pattern to identify example files
    "filename_pattern": r"/(0[0-9]|1[0-9]|2[0-9])_",  # Match 00-29
    # Ignore pattern (failure cases still show but with warning thumbnails)
    "ignore_pattern": r"__init__\.py",
    # Directory for example thumbnails
    "backreferences_dir": "gen_modules/backreferences",
    # Generate mini-galleries in API docs
    "doc_module": ("temporalcv",),
    # Show memory usage
    "show_memory": False,
    # Show download links
    "download_all_examples": True,
    # Thumbnail size
    "thumbnail_size": (400, 280),
    # Default thumbnail for examples without plots
    "default_thumb_file": None,  # Uses sphinx-gallery default
    # Sort examples by file name
    "within_subsection_order": "FileNameSortKey",
    # Capture stderr
    "capture_repr": ("_repr_html_", "__repr__"),
    # Show signature in mini-galleries
    "show_signature": True,
    # Reference label format
    "reference_url": {
        "temporalcv": None,
    },
    # Remove config comments from example scripts
    "remove_config_comments": True,
    # Minimum reported time
    "min_reported_time": 1,
    # Subsection order (categories)
    "subsection_order": [
        "../examples",
    ],
}
