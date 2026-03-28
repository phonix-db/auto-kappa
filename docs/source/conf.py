#
# Configuration file for the Sphinx documentation builder.
#
import os
import re
import sys
sys.path.insert(0, os.path.abspath('..'))
sys.path.insert(0, os.path.abspath('../..'))

mathjax_path = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"

# get version info
#from auto_kappa import __version__ as version

init_path = os.path.abspath('../../auto_kappa/__init__.py')
with open(init_path, encoding='utf-8') as f:
    content = f.read()

path = os.path.join(os.path.dirname(__file__), "auto_kappa", "version.py")
with open(path) as f:
    content = f.read()

match = re.search(r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", content, re.MULTILINE)
version = match.group(1) if match else "unknown"
release = version

# -- Project information 
project = 'auto-kappa'
copyright = '2025, Masato Ohnishi'
author = 'Masato Ohnishi'

# The full version, including alpha/beta/rc tags
# release = 'Sept. 1st, 2025'

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.autosummary",
    "sphinx_rtd_theme",
    "sphinxarg.ext",
    'sphinx_copybutton',
]
autosummary_generate = True

autosummary_filename_map = {
    "auto_kappa.io": "auto_kappa.io",
    "auto_kappa.plot": "auto_kappa.plot",
}

autodoc_default_options = {
    "members": True,
    "undoc-members": True,
    "show-inheritance": True,
    "inherited-members": True,
    # "imported-members": True,
}
autoclass_content = "both"
autodoc_typehints = "description"

templates_path = ['_templates']
exclude_patterns = []

# -- Options for HTML output 
html_theme = 'sphinx_rtd_theme'
html_theme_options = {
    'logo_only': False,
}

html_static_path = ['_static']
html_css_files = ['css/custom.css']
html_favicon = '_static/favicon.ico'
html_logo = 'img/ak_logo.png'
pygments_style = "sphinx"
html_show_sphinx = False

## numbering
numfig = True
math_numfig = True
numfig_secnum_depth = 2
math_eqref_format = "Eq.{number}"
html_use_smartypants = False

## Latex
latex_engine = 'xelatex'
latex_elements = {
    'fontpkg': r'''
''',
    'preamble': r'''
\usepackage{array}
\renewcommand{\arraystretch}{1.2}
'''
}
latex_show_urls = 'footnote'

### open links with a new tab
#window.addEventListener("load", func=()=>{
#    const external_links = document.querySelectorAll("a.external");
#    for(let i=0; i<external_links.length; i++){
#        external_links[i].setAttribute("target", "_blank");
#        external_links[i].setAttribute("rel", "noopener noreferrer");
#    }
#});

def setup(app):
    app.add_css_file('custom.css')

