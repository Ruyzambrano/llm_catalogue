"""llm_catalogue: model names, pricing, and free-tier metadata for OpenAI,
Anthropic, Google Gemini, and any other provider the registry grows.

    from llm_catalogue import Catalog

    catalog = Catalog()
    catalog.get_free_models("google")

The data format and loader are built to accept registries written by newer
releases than the one you have installed -- unknown vendors, statuses and
fields are preserved rather than rejected. See :mod:`llm_catalogue.models`
and :mod:`llm_catalogue.catalogue` for the specifics.
"""
from ._version import __version__
from .catalogue import SCHEMA_VERSION, Catalog, IncompatibleRegistryError, RegistryData, parse_registry
from .models import AIModel, FreeTierPolicy, ModelStatus, TieredPricing, TokenPricing, Vendor

__all__ = [
    "Catalog",
    "AIModel",
    "TokenPricing",
    "TieredPricing",
    "FreeTierPolicy",
    "ModelStatus",
    "Vendor",
    "RegistryData",
    "parse_registry",
    "IncompatibleRegistryError",
    "SCHEMA_VERSION",
    "__version__",
]
