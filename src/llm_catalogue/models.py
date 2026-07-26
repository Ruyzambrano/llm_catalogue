"""Data model for AI model pricing and free-tier metadata.

Every class here is a plain, JSON-serializable dataclass: call ``to_dict()``
to get a ``dict`` safe for ``json.dumps``, and ``from_dict()`` to rebuild the
object from parsed JSON. This is what ``registry.json`` is made of and what
:class:`llm_catalogue.catalogue.Catalog` loads and returns.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, Optional


class Vendor(str, Enum):
    """The AI providers this package tracks.

    A str subclass, so ``Vendor.OPENAI == "openai"`` and it serializes as a
    plain string in JSON without needing a custom encoder.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ModelStatus(str, Enum):
    """Lifecycle state of a model, as stated by the provider's own docs.

    Attributes:
        ACTIVE: Generally available and recommended for new use.
        DEPRECATED: Still usable but the provider is steering users away.
        RETIRED: Shut down, or only available on a narrow legacy channel
            (e.g. "except on Bedrock and Google Cloud").
        LIMITED: Gated behind restricted/limited availability.
    """

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"
    LIMITED = "limited_availability"


@dataclass
class TokenPricing:
    """USD price per 1 million tokens for one pricing tier of one model.

    Attributes:
        standard_input: Price per 1M input tokens at standard (non-batch) rates.
        output: Price per 1M output tokens at standard (non-batch) rates.
        cached_input: Price per 1M *cache-read* input tokens, if the provider
            offers prompt caching. ``None`` if unknown/not applicable. Note
            this is the cache-*read* price -- separate cache-*write*
            premiums (e.g. Anthropic's 5m/1h writes) aren't modelled.
        batch_input: Price per 1M input tokens via the provider's batch/async
            API, if any.
        batch_output: Price per 1M output tokens via the provider's batch/async
            API, if any.
    """

    standard_input: float
    output: float
    cached_input: Optional[float] = None
    batch_input: Optional[float] = None
    batch_output: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain, JSON-serializable dict of this pricing."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenPricing":
        """Rebuilds a TokenPricing from a dict produced by :meth:`to_dict`."""
        return cls(**data)


@dataclass
class TieredPricing:
    """Context-length-dependent pricing.

    Some models (e.g. Gemini 2.5 Pro past 200k tokens) charge a higher rate
    once the prompt crosses a token threshold. :meth:`AIModel.calculate_cost`
    picks ``over_threshold_rate`` automatically when ``input_tokens`` exceeds
    ``threshold_tokens``.

    Attributes:
        threshold_tokens: The input-token count at which the higher rate kicks in.
        base_rate: Pricing used at or below ``threshold_tokens``.
        over_threshold_rate: Pricing used above ``threshold_tokens``.
    """

    threshold_tokens: int
    base_rate: TokenPricing
    over_threshold_rate: TokenPricing

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain, JSON-serializable dict of this tiered pricing."""
        return {
            "threshold_tokens": self.threshold_tokens,
            "base_rate": self.base_rate.to_dict(),
            "over_threshold_rate": self.over_threshold_rate.to_dict(),
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TieredPricing":
        """Rebuilds a TieredPricing from a dict produced by :meth:`to_dict`."""
        return cls(
            threshold_tokens=data["threshold_tokens"],
            base_rate=TokenPricing.from_dict(data["base_rate"]),
            over_threshold_rate=TokenPricing.from_dict(data["over_threshold_rate"]),
        )


@dataclass
class FreeTierPolicy:
    """Whether, and how, a model can be used free of charge.

    Attributes:
        has_free_tier: Whether the provider lets you call this model without
            paying. This is what :attr:`AIModel.is_free` and
            :meth:`Catalog.get_free_models` key off of.
        rate_limit_rpm: Free-tier requests-per-minute limit, if known. Not
            currently populated for Gemini -- see README.
        data_used_for_training: Whether using the free tier means the
            provider may use your prompts/outputs to improve their models.
            ``None`` when not applicable (e.g. no free tier exists at all).
    """

    has_free_tier: bool
    rate_limit_rpm: Optional[int] = None
    data_used_for_training: Optional[bool] = None

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain, JSON-serializable dict of this policy."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FreeTierPolicy":
        """Rebuilds a FreeTierPolicy from a dict produced by :meth:`to_dict`."""
        return cls(**data)


@dataclass
class AIModel:
    """A single model from a single provider, with its pricing and metadata.

    This is the unit of data returned by every :class:`~llm_catalogue.catalogue.Catalog`
    query method (``get_models``, ``get_free_models``, ``find_free_models``, ``get_model``).

    Attributes:
        id: The provider's own model identifier, e.g. ``"gemini-2.5-flash"``.
            Use this to look models up with ``Catalog.get_model()``.
        name: Human-readable display name, e.g. ``"Gemini 2.5 Flash"``.
        vendor: Which provider this model belongs to.
        pricing: Standard-tier USD pricing for this model.
        context_window: Max input tokens, if known. ``None`` when not
            documented on the pricing page itself.
        tiered_pricing: Alternate pricing that applies past a token
            threshold (see :class:`TieredPricing`), or ``None`` if this
            model's price doesn't change with context length.
        free_tier: Free-tier eligibility and terms, or ``None`` if the
            provider doesn't document one at all.
        status: Lifecycle state (active/deprecated/retired/limited).
        tool_costs: Reserved for per-tool costs (e.g. web search, code
            execution) keyed by tool name. Not populated by the bundled
            scraper in v1.
    """

    id: str
    name: str
    vendor: Vendor
    pricing: TokenPricing
    context_window: Optional[int] = None
    tiered_pricing: Optional[TieredPricing] = None
    free_tier: Optional[FreeTierPolicy] = None
    status: ModelStatus = ModelStatus.ACTIVE
    tool_costs: Dict[str, float] = field(default_factory=dict)

    @property
    def is_free(self) -> bool:
        """Whether this model has any free tier at all.

        Shorthand for ``bool(model.free_tier and model.free_tier.has_free_tier)``.
        This is what :class:`~llm_catalogue.catalogue.Catalog`'s
        ``get_free_models``/``find_free_models``/``has_free_tier`` filter on.
        """
        return bool(self.free_tier and self.free_tier.has_free_tier)

    def calculate_cost(
        self,
        input_tokens: int,
        output_tokens: int,
        cached_tokens: int = 0,
        is_batch: bool = False,
    ) -> float:
        """Estimates the USD cost of a single request against this model's pricing.

        Args:
            input_tokens: Total input/prompt tokens for the request,
                including any cached tokens (``cached_tokens`` must be a
                subset of this count, not additional to it).
            output_tokens: Total output/completion tokens for the request.
            cached_tokens: How many of ``input_tokens`` were served from a
                prompt cache and billed at ``pricing.cached_input`` instead
                of the standard input rate. Defaults to 0 (no caching).
            is_batch: If True and the model has batch pricing configured,
                bills at ``pricing.batch_input``/``pricing.batch_output``
                instead of standard rates, and caching is ignored (batch
                APIs generally don't support prompt caching).

        Returns:
            Estimated cost in USD, rounded to 6 decimal places. If
            ``tiered_pricing`` is set and ``input_tokens`` exceeds its
            ``threshold_tokens``, the over-threshold rate is used instead
            of ``pricing``.
        """
        rates = self.pricing
        if self.tiered_pricing and input_tokens > self.tiered_pricing.threshold_tokens:
            rates = self.tiered_pricing.over_threshold_rate

        if is_batch and rates.batch_input is not None and rates.batch_output is not None:
            in_cost = (input_tokens / 1_000_000) * rates.batch_input
            out_cost = (output_tokens / 1_000_000) * rates.batch_output
            return round(in_cost + out_cost, 6)

        fresh_tokens = max(0, input_tokens - cached_tokens)
        cached_rate = rates.cached_input if rates.cached_input is not None else rates.standard_input

        in_cost = (fresh_tokens / 1_000_000) * rates.standard_input
        cache_cost = (cached_tokens / 1_000_000) * cached_rate
        out_cost = (output_tokens / 1_000_000) * rates.output

        return round(in_cost + cache_cost + out_cost, 6)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain, JSON-serializable dict of this model.

        This is the exact shape stored per-entry in ``registry.json``'s
        ``"models"`` list.
        """
        return {
            "id": self.id,
            "name": self.name,
            "vendor": self.vendor.value,
            "pricing": self.pricing.to_dict(),
            "context_window": self.context_window,
            "tiered_pricing": self.tiered_pricing.to_dict() if self.tiered_pricing else None,
            "free_tier": self.free_tier.to_dict() if self.free_tier else None,
            "status": self.status.value,
            "tool_costs": self.tool_costs,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIModel":
        """Rebuilds an AIModel from a dict produced by :meth:`to_dict`."""
        tiered = data.get("tiered_pricing")
        free_tier = data.get("free_tier")
        return cls(
            id=data["id"],
            name=data["name"],
            vendor=Vendor(data["vendor"]),
            pricing=TokenPricing.from_dict(data["pricing"]),
            context_window=data.get("context_window"),
            tiered_pricing=TieredPricing.from_dict(tiered) if tiered else None,
            free_tier=FreeTierPolicy.from_dict(free_tier) if free_tier else None,
            status=ModelStatus(data.get("status", "active")),
            tool_costs=data.get("tool_costs", {}),
        )
