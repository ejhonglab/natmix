# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
# import os
# import sys
# sys.path.insert(0, os.path.abspath('.'))


# -- Project information -----------------------------------------------------

project = 'natmix'
copyright = "2022, Tom O'Connell"
author = "Tom O'Connell"

# The full version, including alpha/beta/rc tags
release = '0.0.0'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinxcontrib.apidoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.autodoc',

    # TODO actually use
    'sphinx.ext.viewcode',

    # TODO actually use
    'sphinx-prompt',

    # It is important that this comes after napoleon, see:
    # https://github.com/agronholm/sphinx-autodoc-typehints/issues/15
    # TODO actually use
    'sphinx_autodoc_typehints',
]


# TODO TODO any way to exclude specific stuff (e.g. functions in a module) from apidoc
# -> autodoc generation? would i need to manually change the outputs of apidoc?

# TODO also give sphinx-autoapi a try
# (suite2p seems like they tried it briefly but they don't install the packaged in the
# docs section of setup.py where they install the other stuff)

apidoc_module_dir = '../../natmix'
apidoc_output_dir = 'apidoc'
apidoc_separate_modules = True
apidoc_excluded_paths = []

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
