project = "evaluma"
author = "Nils Lehmann"
copyright = "2024, Nils Lehmann"

extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "autoapi.extension",
    "myst_nb",
    "sphinx_design",
]

html_theme = "sphinx_book_theme"
html_logo = "_static/logo.png"
html_static_path = ["_static"]
html_theme_options = {
    "repository_url": "https://github.com/nilsleh/evaluma",
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_edit_page_button": True,
    "use_issues_button": True,
}

autoapi_dirs = ["../evaluma"]
autoapi_ignore = []

exclude_patterns = ["_build", "**.ipynb_checkpoints"]

suppress_warnings = ["ref.python"]

nb_execution_mode = "force"
nb_execution_timeout = 120
myst_enable_extensions = ["amsmath", "dollarmath", "colon_fence"]
