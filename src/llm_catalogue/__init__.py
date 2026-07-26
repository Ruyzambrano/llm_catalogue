"""llm_catalogue: model names, pricing, and free-tier metadata for OpenAI,
Anthropic, and Google Gemini.

    from llm_catalogue import Catalog

    catalog = Catalog()
    catalog.get_free_models("google")
"""
from .catalogue import Catalog
from .models import AIModel, FreeTierPolicy, ModelStatus, TieredPricing, TokenPricing, Vendor

__version__ = "0.1.0"
__all__ = [
    "Catalog",
    "AIModel",
    "TokenPricing",
    "TieredPricing",
    "FreeTierPolicy",
    "ModelStatus",
    "Vendor",
]
