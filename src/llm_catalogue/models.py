"""Data model for AI model pricing and free-tier metadata.

Every class here is a plain, JSON-serializable dataclass: call ``to_dict()``
to get a ``dict`` safe for ``json.dumps``, and ``from_dict()`` to rebuild the
object from parsed JSON. This is what ``registry.json`` is made of and what
:class:`llm_catalogue.catalogue.Catalog` loads and returns.

**Forward compatibility.** Providers add models, statuses, pricing dimensions
and whole new APIs faster than anyone re-releases a package, and
:meth:`llm_catalogue.catalogue.Catalog.refresh` deliberately feeds *newer*
registry data into *older* installed code. Everything here is therefore built
to accept data it has never seen:

* :class:`Vendor` and :class:`ModelStatus` are **open enums** -- an unknown
  string (``"mistral"``, ``"sunset"``) becomes a usable member instead of
  raising ``ValueError``.
* Every ``from_dict`` ignores nothing: unrecognized keys are kept in
  :attr:`~AIModel.extra` and written back out by ``to_dict``, so a field added
  to the registry next year still round-trips through this year's code.
* Missing or non-numeric values degrade to ``None`` rather than raising, so a
  model published before its price is announced still loads.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field, fields
from enum import Enum
from typing import Any, Dict, List, Optional


class _OpenStrEnum(str, Enum):
    """A string enum that accepts values it has never heard of.

    Calling it with an unknown string mints a new member on the fly rather
    than raising ``ValueError``. That member behaves like any other -- it is
    a ``str``, compares equal to its value, and has ``.value`` and ``.name``
    -- but it is deliberately kept out of ``__members__`` and iteration, so
    "every vendor this package knows about" stays a meaningful question
    (:meth:`known`) separately from "every vendor this data contains".

    This is what lets a registry naming a provider that didn't exist when
    you installed the package load cleanly instead of blowing up.
    """

    @classmethod
    def _missing_(cls, value: Any) -> Optional["_OpenStrEnum"]:
        if not isinstance(value, str):
            return None
        normalized = value.strip().lower()
        if not normalized:
            return None
        existing = cls._value2member_map_.get(normalized)
        if existing is not None:
            return existing
        pseudo = str.__new__(cls, normalized)
        pseudo._name_ = re.sub(r"[^A-Z0-9]+", "_", normalized.upper()).strip("_") or "UNKNOWN"
        pseudo._value_ = normalized
        # Memoize so repeated lookups return the same object (``is`` keeps
        # working), while staying out of _member_map_ so iteration and
        # __members__ only ever list the canonical, package-known members.
        cls._value2member_map_[normalized] = pseudo
        return pseudo

    @classmethod
    def known(cls) -> List["_OpenStrEnum"]:
        """Returns the canonical members declared by this package.

        Excludes members minted at runtime from unrecognized registry data;
        use :attr:`is_known` to test a single value.
        """
        return list(cls)

    @property
    def is_known(self) -> bool:
        """Whether this member was declared by the package (vs. minted from
        data this version of the package doesn't recognize)."""
        return self._name_ in type(self).__members__

    def __str__(self) -> str:
        return self._value_


class Vendor(_OpenStrEnum):
    """The AI providers this package tracks.

    A str subclass, so ``Vendor.OPENAI == "openai"`` and it serializes as a
    plain string in JSON without needing a custom encoder.

    Open by design: ``Vendor("mistral")`` returns a working vendor rather
    than raising, so a registry listing a provider newer than your installed
    copy of llm-catalogue still loads. ``Vendor.known()`` lists the ones
    shipped with this version; ``vendor.is_known`` tests a single value.
    """

    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class ModelStatus(_OpenStrEnum):
    """Lifecycle state of a model, as stated by the provider's own docs.

    Open by design (see :class:`Vendor`): a status string this version
    doesn't know is preserved verbatim instead of raising.

    Attributes:
        ACTIVE: Generally available and recommended for new use.
        DEPRECATED: Still usable but the provider is steering users away.
        RETIRED: Shut down, or only available on a narrow legacy channel
            (e.g. "except on Bedrock and Google Cloud").
        LIMITED: Gated behind restricted/limited availability.
        UNKNOWN: The source data didn't say.
    """

    ACTIVE = "active"
    DEPRECATED = "deprecated"
    RETIRED = "retired"
    LIMITED = "limited_availability"
    UNKNOWN = "unknown"


def _as_float(value: Any) -> Optional[float]:
    """Coerces a JSON value to a float, or ``None`` if it isn't numeric.

    Tolerates the string forms providers' docs (and hand-edited registries)
    drift into -- ``"2.50"``, ``"$2.50"``, ``"1,250"`` -- rather than
    failing the whole load over one odd cell.
    """
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = re.search(r"-?\d+(?:\.\d+)?", value.replace(",", ""))
        if match:
            return float(match.group(0))
    return None


def _as_int(value: Any) -> Optional[int]:
    """Coerces a JSON value to an int, or ``None`` if it isn't numeric."""
    number = _as_float(value)
    return int(number) if number is not None else None


def _as_bool(value: Any) -> Optional[bool]:
    """Coerces a JSON value to a bool, or ``None`` if it isn't boolean-ish."""
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "yes", "1"}:
            return True
        if lowered in {"false", "no", "0"}:
            return False
    return None


def _split_known(cls: type, data: Dict[str, Any]) -> tuple:
    """Splits a raw dict into (fields this dataclass declares, everything else).

    The "everything else" half is what gets stashed in ``extra`` so that
    fields added to the registry format after this release still survive a
    ``from_dict``/``to_dict`` round trip instead of being silently dropped.
    """
    if not isinstance(data, dict):
        return {}, {}
    declared = {f.name for f in fields(cls)} - {"extra"}
    known = {k: v for k, v in data.items() if k in declared}
    extra = {k: v for k, v in data.items() if k not in declared}
    return known, extra


@dataclass
class TokenPricing:
    """USD price per 1 million tokens for one pricing tier of one model.

    Every rate is optional: providers routinely announce a model before its
    price, retire a dimension, or add one this release has no field for
    (those land in :attr:`extra`).

    Attributes:
        standard_input: Price per 1M input tokens at standard (non-batch)
            rates, or ``None`` if the provider hasn't published one.
        output: Price per 1M output tokens at standard (non-batch) rates.
        cached_input: Price per 1M *cache-read* input tokens, if the provider
            offers prompt caching. ``None`` if unknown/not applicable. Note
            this is the cache-*read* price -- separate cache-*write*
            premiums (e.g. Anthropic's 5m/1h writes) aren't modelled.
        batch_input: Price per 1M input tokens via the provider's batch/async
            API, if any.
        batch_output: Price per 1M output tokens via the provider's batch/async
            API, if any.
        extra: Any pricing keys present in the source data that this version
            doesn't model (e.g. a newly introduced rate). Preserved verbatim
            and re-emitted by :meth:`to_dict`.
    """

    standard_input: Optional[float] = None
    output: Optional[float] = None
    cached_input: Optional[float] = None
    batch_input: Optional[float] = None
    batch_output: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_priced(self) -> bool:
        """Whether both headline rates are known, i.e. a cost can be computed."""
        return self.standard_input is not None and self.output is not None

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain, JSON-serializable dict of this pricing.

        Unmodelled keys captured in :attr:`extra` are written back out
        alongside the known ones, so round-tripping never loses data.
        """
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TokenPricing":
        """Rebuilds a TokenPricing from a dict produced by :meth:`to_dict`.

        Unknown keys are kept in :attr:`extra`; non-numeric values become
        ``None`` instead of raising.
        """
        known, extra = _split_known(cls, data or {})
        return cls(
            standard_input=_as_float(known.get("standard_input")),
            output=_as_float(known.get("output")),
            cached_input=_as_float(known.get("cached_input")),
            batch_input=_as_float(known.get("batch_input")),
            batch_output=_as_float(known.get("batch_output")),
            extra=extra,
        )


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
        extra: Unmodelled keys from the source data, preserved on round trip.
    """

    threshold_tokens: Optional[int] = None
    base_rate: TokenPricing = field(default_factory=TokenPricing)
    over_threshold_rate: TokenPricing = field(default_factory=TokenPricing)
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain, JSON-serializable dict of this tiered pricing."""
        out = {
            "threshold_tokens": self.threshold_tokens,
            "base_rate": self.base_rate.to_dict(),
            "over_threshold_rate": self.over_threshold_rate.to_dict(),
        }
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TieredPricing":
        """Rebuilds a TieredPricing from a dict produced by :meth:`to_dict`."""
        known, extra = _split_known(cls, data or {})
        return cls(
            threshold_tokens=_as_int(known.get("threshold_tokens")),
            base_rate=TokenPricing.from_dict(known.get("base_rate") or {}),
            over_threshold_rate=TokenPricing.from_dict(known.get("over_threshold_rate") or {}),
            extra=extra,
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
        extra: Unmodelled keys from the source data, preserved on round trip.
    """

    has_free_tier: bool = False
    rate_limit_rpm: Optional[int] = None
    data_used_for_training: Optional[bool] = None
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain, JSON-serializable dict of this policy."""
        out = {f.name: getattr(self, f.name) for f in fields(self) if f.name != "extra"}
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "FreeTierPolicy":
        """Rebuilds a FreeTierPolicy from a dict produced by :meth:`to_dict`."""
        known, extra = _split_known(cls, data or {})
        return cls(
            has_free_tier=bool(_as_bool(known.get("has_free_tier"))),
            rate_limit_rpm=_as_int(known.get("rate_limit_rpm")),
            data_used_for_training=_as_bool(known.get("data_used_for_training")),
            extra=extra,
        )


@dataclass
class AIModel:
    """A single model from a single provider, with its pricing and metadata.

    This is the unit of data returned by every :class:`~llm_catalogue.catalogue.Catalog`
    query method (``get_models``, ``get_free_models``, ``find_free_models``, ``get_model``).

    Attributes:
        id: The provider's own model identifier, e.g. ``"gemini-2.5-flash"``.
            Use this to look models up with ``Catalog.get_model()``.
        name: Human-readable display name, e.g. ``"Gemini 2.5 Flash"``.
        vendor: Which provider this model belongs to. May be a vendor this
            package version doesn't know about (see :class:`Vendor`).
        pricing: Standard-tier USD pricing for this model.
        context_window: Max input tokens, if known. ``None`` when not
            documented on the pricing page itself.
        tiered_pricing: Alternate pricing that applies past a token
            threshold (see :class:`TieredPricing`), or ``None`` if this
            model's price doesn't change with context length.
        free_tier: Free-tier eligibility and terms, or ``None`` if the
            provider doesn't document one at all.
        status: Lifecycle state (active/deprecated/retired/limited), or any
            other string the provider's docs start using.
        tool_costs: Per-tool costs (e.g. web search, code execution) keyed
            by tool name. Not populated by the bundled scraper in v1.
        extra: Any top-level model keys present in the source data that this
            version doesn't model -- e.g. a ``modalities`` or ``knowledge_cutoff``
            field added to the registry after this release. Preserved
            verbatim, re-emitted by :meth:`to_dict`, and readable directly
            (``model.extra["modalities"]``) so new registry data is usable
            without upgrading.
    """

    id: str
    name: str = ""
    vendor: Vendor = Vendor.OPENAI
    pricing: TokenPricing = field(default_factory=TokenPricing)
    context_window: Optional[int] = None
    tiered_pricing: Optional[TieredPricing] = None
    free_tier: Optional[FreeTierPolicy] = None
    status: ModelStatus = ModelStatus.ACTIVE
    tool_costs: Dict[str, float] = field(default_factory=dict)
    extra: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_free(self) -> bool:
        """Whether this model has any free tier at all.

        Shorthand for ``bool(model.free_tier and model.free_tier.has_free_tier)``.
        This is what :class:`~llm_catalogue.catalogue.Catalog`'s
        ``get_free_models``/``find_free_models``/``has_free_tier`` filter on.
        """
        return bool(self.free_tier and self.free_tier.has_free_tier)

    @property
    def is_priced(self) -> bool:
        """Whether this model has enough pricing data for :meth:`calculate_cost`.

        ``False`` for a model the provider has announced but not yet priced.
        Check this before costing models you didn't hand-pick.
        """
        return self.pricing.is_priced

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
                APIs generally don't support prompt caching). Falls back to
                standard rates if the model has no batch pricing.

        Returns:
            Estimated cost in USD, rounded to 6 decimal places. If
            ``tiered_pricing`` is set and ``input_tokens`` exceeds its
            ``threshold_tokens``, the over-threshold rate is used instead
            of ``pricing``.

        Raises:
            ValueError: If the registry has no published input/output rate
                for this model (check :attr:`is_priced` first).
        """
        rates = self.pricing
        tiered = self.tiered_pricing
        if tiered and tiered.threshold_tokens is not None and input_tokens > tiered.threshold_tokens:
            rates = tiered.over_threshold_rate

        if is_batch and rates.batch_input is not None and rates.batch_output is not None:
            in_cost = (input_tokens / 1_000_000) * rates.batch_input
            out_cost = (output_tokens / 1_000_000) * rates.batch_output
            return round(in_cost + out_cost, 6)

        if not rates.is_priced:
            raise ValueError(
                f"No published pricing for model {self.id!r}"
                f"{' above its tier threshold' if rates is not self.pricing else ''} -- "
                "check AIModel.is_priced before calling calculate_cost()."
            )

        fresh_tokens = max(0, input_tokens - cached_tokens)
        cached_rate = rates.cached_input if rates.cached_input is not None else rates.standard_input

        in_cost = (fresh_tokens / 1_000_000) * rates.standard_input
        cache_cost = (cached_tokens / 1_000_000) * cached_rate
        out_cost = (output_tokens / 1_000_000) * rates.output

        return round(in_cost + cache_cost + out_cost, 6)

    def to_dict(self) -> Dict[str, Any]:
        """Returns a plain, JSON-serializable dict of this model.

        This is the exact shape stored per-entry in ``registry.json``'s
        ``"models"`` list, plus any unmodelled keys held in :attr:`extra`.
        """
        out = {
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
        out.update(self.extra)
        return out

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AIModel":
        """Rebuilds an AIModel from a dict produced by :meth:`to_dict`.

        Deliberately forgiving, because this parses data that may have been
        written by a *newer* version of the package: unknown vendors and
        statuses are accepted as-is, unknown keys are kept in :attr:`extra`,
        and missing optional data degrades to ``None`` rather than raising.

        Raises:
            ValueError: Only if ``data`` isn't a dict or has no usable
                ``id`` -- an entry with no identity can't be looked up, so
                it's rejected (and skipped by
                :class:`~llm_catalogue.catalogue.Catalog`) rather than
                silently kept.
        """
        if not isinstance(data, dict):
            raise ValueError(f"Model entry must be an object, got {type(data).__name__}")
        known, extra = _split_known(cls, data)

        model_id = known.get("id")
        if not isinstance(model_id, str) or not model_id.strip():
            raise ValueError("Model entry is missing a usable 'id'")

        name = known.get("name")
        tiered = known.get("tiered_pricing")
        free_tier = known.get("free_tier")
        tool_costs = known.get("tool_costs")

        return cls(
            id=model_id.strip(),
            name=name if isinstance(name, str) and name else model_id.strip(),
            vendor=Vendor(known.get("vendor") or "unknown"),
            pricing=TokenPricing.from_dict(known.get("pricing") or {}),
            context_window=_as_int(known.get("context_window")),
            tiered_pricing=TieredPricing.from_dict(tiered) if isinstance(tiered, dict) else None,
            free_tier=FreeTierPolicy.from_dict(free_tier) if isinstance(free_tier, dict) else None,
            status=ModelStatus(known.get("status") or "active"),
            tool_costs=tool_costs if isinstance(tool_costs, dict) else {},
            extra=extra,
        )
