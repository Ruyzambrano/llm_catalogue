"""Single source of truth for the package version.

``pyproject.toml`` reads it from here (``[tool.hatch.version]``), ``__init__``
re-exports it, and :mod:`llm_catalogue.catalogue` puts it in the refresh
User-Agent -- so there's one place to bump and nothing to keep in sync.
"""

__version__ = "0.3.0"
