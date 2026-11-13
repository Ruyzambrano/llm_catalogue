"""Dev-only tool that rebuilds ``data/registry.json`` from each provider's
public pricing docs.

Not a runtime dependency of llm_catalogue -- the published package only ships
the bundled ``registry.json``, it never scrapes at import time. Install the
extra scraping deps and run this module directly to refresh the bundled data::

    pip install -e ".[scraper]"
    python -m llm_catalogue.scraper            # rebuild the registry
    python -m llm_catalogue.scraper --check    # CI: did any provider's page change?

Providers publish pricing as prose for humans, so every parser here is a
guess about a page that will eventually be redesigned without warning. Two
things follow, and they shape this module:

**Adding a provider is one function.** Decorate a parser with
:func:`register_provider` and it joins the run -- no editing of
:func:`main`, no list to keep in sync::

    @register_provider("mistral", Vendor("mistral"), "https://example.com/pricing.md")
    def parse_mistral(text: str) -> List[AIModel]:
        ...

**A broken parser fails loudly, never quietly.** A parser that finds nothing
raises :class:`ParseError`, and :func:`main` additionally rejects a run whose
model count collapses against the registry on disk (see
:data:`REGRESSION_THRESHOLD`). On failure the previous data for that provider
is kept and the process exits non-zero, so a page redesign shows up as a red
CI run rather than as a provider that silently vanished from the registry.

Only "Standard" tier, text-in/text-out pricing is captured. Batch pricing is
captured where the source table has it. Multimodal/audio/image/video/embedding
models and cache-write premiums are intentionally out of scope for v1 -- see
README.md.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from llm_catalogue.catalogue import SCHEMA_VERSION
from llm_catalogue.models import AIModel, FreeTierPolicy, ModelStatus, TieredPricing, TokenPricing, Vendor

GEMINI_URL = "https://ai.google.dev/gemini-api/docs/pricing.md.txt"
ANTHROPIC_URL = "https://platform.claude.com/docs/en/about-claude/pricing.md"
OPENAI_URL = "https://developers.openai.com/api/docs/pricing.md"

REGISTRY_PATH = Path(__file__).parent / "data" / "registry.json"

#: A provider run that returns fewer than this fraction of the models already
#: in the registry is treated as a parse failure rather than a real shrink.
#: Providers retire models a few at a time; they don't lose half their
#: lineup overnight, so a collapse means the page changed shape.
REGRESSION_THRESHOLD = 0.5


class ParseError(RuntimeError):
    """Raised when a provider's pricing page can't be parsed.

    Always a signal that the page changed and the parser needs updating --
    which is exactly why parsers raise this instead of returning ``[]``: an
    empty list would quietly erase that provider from the registry.
    """


# ---------------------------------------------------------------------------
# Provider registry
# ---------------------------------------------------------------------------


@dataclass
class Provider:
    """One provider's pricing source and the parser that reads it.

    Attributes:
        key: Short CLI-friendly identifier, e.g. ``"openai"``.
        vendor: The :class:`~llm_catalogue.models.Vendor` its models get.
        url: The pricing doc to fetch.
        parse: Callable taking the fetched text and returning models.
        label: Human-readable name used in log output.
    """

    key: str
    vendor: Vendor
    url: str
    parse: Callable[[str], List[AIModel]]
    label: str = ""

    def __post_init__(self) -> None:
        self.label = self.label or self.key.title()


#: Every registered provider, keyed by :attr:`Provider.key`, in registration
#: order. :func:`main` iterates this -- nothing else enumerates providers.
PROVIDERS: Dict[str, Provider] = {}


def register_provider(
    key: str,
    vendor: Vendor,
    url: str,
    label: str = "",
) -> Callable[[Callable[[str], List[AIModel]]], Callable[[str], List[AIModel]]]:
    """Registers a parser function as a provider for :func:`main` to run.

    This is the whole extension point: adding a provider means writing a
    parser and decorating it, with no other file to touch.

    Args:
        key: Short identifier, also what ``--only`` matches on.
        vendor: Vendor to tag the parsed models with. Pass
            ``Vendor("something-new")`` for a provider this package doesn't
            declare -- :class:`~llm_catalogue.models.Vendor` is an open enum.
        url: The pricing doc URL to fetch.
        label: Display name for log output; defaults to a title-cased key.

    Returns:
        The decorator, which registers the function and returns it unchanged
        so it stays directly importable and unit-testable.
    """

    def decorator(func: Callable[[str], List[AIModel]]) -> Callable[[str], List[AIModel]]:
        PROVIDERS[key] = Provider(key=key, vendor=vendor, url=url, parse=func, label=label)
        return func

    return decorator


# ---------------------------------------------------------------------------
# Shared parsing helpers
# ---------------------------------------------------------------------------

_PRICE_RE = re.compile(r"\$\s*([\d,]+(?:\.\d+)?)")


def money(cell: str) -> Optional[float]:
    """Extracts a dollar amount from a table cell, or ``None`` if there isn't one.

    Deliberately format-agnostic -- it handles ``$3.50``, ``$3.50 / MTok``,
    ``$1,250``, and returns ``None`` for the ``-`` and ``N/A`` cells
    providers use for "not offered" -- so a provider restyling its price
    column doesn't need a new regex.
    """
    match = _PRICE_RE.search(cell or "")
    return float(match.group(1).replace(",", "")) if match else None


def table_rows(table: str) -> List[List[str]]:
    """Splits a markdown table into rows of stripped cells.

    Separator rows (``| --- |``) and blank lines are dropped; the header
    row is kept as ``rows[0]``.
    """
    rows = []
    for line in table.splitlines():
        line = line.strip()
        if not line.startswith("|") or re.fullmatch(r"[|\s:\-]+", line):
            continue
        rows.append([cell.strip() for cell in line.strip("|").split("|")])
    return rows


_MARKDOC_TABLE_RE = re.compile(r"\{%\s*table\s*%\}(?P<body>.*?)\{%\s*/\s*table\s*%\}", re.DOTALL)


def markdoc_tables(text: str) -> List[List[List[str]]]:
    """Parses Markdoc ``{% table %}`` blocks into the same rows-of-cells shape
    as :func:`table_rows`.

    Google's pricing page swapped markdown tables for this format, where
    cells are ``*`` bullets and rows are separated by ``---``::

        {% table %}
        * Feature
        * Free Tier
        ---
        * Input price
        * Free of charge
        {% /table %}

    Supporting both formats means a provider switching between them costs
    nothing here.
    """
    tables = []
    for block in _MARKDOC_TABLE_RE.finditer(text):
        rows: List[List[str]] = [[]]
        cells = rows[0]
        for line in block.group("body").splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            if re.fullmatch(r"-{3,}", stripped):
                cells = []
                rows.append(cells)
            elif stripped.startswith("* "):
                cells.append(stripped[2:].strip())
            elif cells:
                # A wrapped continuation of the previous cell.
                cells[-1] = f"{cells[-1]} {stripped}".strip()
        tables.append([[re.sub(r"\s+", " ", c).strip() for c in row] for row in rows if row])
    return tables


def find_table(text: str, heading: str) -> Optional[str]:
    """Returns the first markdown table following ``heading``, if any.

    Args:
        text: The document to search.
        heading: A regex matched against the heading line that precedes the
            table (e.g. ``r"### Standard pricing data"``).
    """
    match = re.search(heading + r"[^\n]*\n+(?P<table>\|.+?)(?:\n\s*\n|\Z)", text, re.DOTALL)
    return match.group("table") if match else None


def _status_and_name(raw_name: str) -> tuple:
    """Splits a raw provider-doc name into a clean display name and status.

    Handles the "Name (annotation)" convention both Anthropic and OpenAI use
    for lifecycle notes, e.g. "Claude Opus 4.1 ([deprecated](...))" or
    "gpt-5.5 (<272K context length)". Everything from the first "(" onward
    is dropped from the name; ``status`` is only set from keywords, so a
    purely descriptive parenthetical (like the context-length one) leaves
    status as ACTIVE.
    """
    lowered = raw_name.lower()
    if "retired" in lowered or "shut down" in lowered or "sunset" in lowered:
        status = ModelStatus.RETIRED
    elif "deprecated" in lowered or "legacy" in lowered:
        status = ModelStatus.DEPRECATED
    elif "limited availability" in lowered or "preview" in lowered:
        status = ModelStatus.LIMITED
    else:
        status = ModelStatus.ACTIVE
    name = raw_name.split("(")[0].strip()
    return name, status


_CONTEXT_TIER_RE = re.compile(r"(?P<tier>up to|over|above|below|under)\s+(?P<count>[\d,]+)\s*(?P<unit>[KkMm])?\s*tokens")


def _context_tier(raw_name: str):
    """Reads a "(for prompts up to 100,000 tokens)" annotation off a model name.

    Returns ``("base"|"over", threshold_tokens)``, or ``(None, None)`` when
    the name carries no such annotation. Providers increasingly split one
    model across two priced rows this way; recognizing it keeps the two
    from colliding under one id.
    """
    match = _CONTEXT_TIER_RE.search(raw_name.lower())
    if not match:
        return None, None
    count = float(match.group("count").replace(",", ""))
    unit = (match.group("unit") or "").lower()
    threshold = int(count * {"k": 1_000, "m": 1_000_000}.get(unit, 1))
    tier = "base" if match.group("tier") in {"up to", "below", "under"} else "over"
    return tier, threshold


def _slugify(name: str) -> str:
    """Turns a display name like "Claude Opus 4.1" into "claude-opus-4-1".

    Anthropic's pricing page doesn't publish machine-readable model ids the
    way Google and OpenAI's docs do, so this derives one from the display name.
    """
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------


@register_provider("anthropic", Vendor.ANTHROPIC, ANTHROPIC_URL, label="Anthropic")
def parse_anthropic(text: str) -> List[AIModel]:
    """Parses the standard-context pricing table from platform.claude.com.

    Columns are located by header name rather than position, so Anthropic
    adding or reordering a column (another cache tier, a batch column)
    changes nothing here.

    Args:
        text: Raw markdown body fetched from :data:`ANTHROPIC_URL`.

    Returns:
        One AIModel per model row in the pricing table. ``cached_input`` is
        taken from the cache-hits column; the 5m/1h cache-*write* columns
        aren't captured (see README's "Known limitations").

    Raises:
        ParseError: If no table, or no model rows, can be found -- i.e. the
            page changed and this parser needs updating.
    """
    match = re.search(r"(?:^\|[^\n]+\|\n?)+", text, re.MULTILINE)
    if not match:
        raise ParseError("Could not find a pricing table in the Anthropic pricing doc")

    rows = table_rows(match.group(0))
    if not rows:
        raise ParseError("Anthropic pricing table contained no rows")

    header = [cell.lower() for cell in rows[0]]

    def column(*keywords: str, exclude: Iterable[str] = ()) -> Optional[int]:
        """Index of the first header cell containing every keyword and no
        excluded word."""
        for index, cell in enumerate(header):
            if all(word in cell for word in keywords) and not any(word in cell for word in exclude):
                return index
        return None

    # Exclusions matter: a "Batch input tokens" or "5m cache writes" column
    # must never be mistaken for the base input or output rate if Anthropic
    # adds one ahead of the existing columns.
    base_only = ("batch", "cache", "write", "hit")
    input_col = column("input", exclude=base_only)
    if input_col is None:
        input_col = 1
    output_col = column("output", exclude=base_only)
    cache_hit_col = column("cache", "hit")
    batch_in_col = column("batch", "input")
    batch_out_col = column("batch", "output")

    def cell(row: List[str], index: Optional[int]) -> Optional[str]:
        return row[index] if index is not None and index < len(row) else None

    models: Dict[str, AIModel] = {}
    for row in rows[1:]:
        raw_name = row[0]
        if not raw_name or not re.match(r"^[A-Za-z]", raw_name):
            continue
        name, status = _status_and_name(raw_name)
        standard_input = money(cell(row, input_col) or "")
        output = money(cell(row, output_col) or "") if output_col is not None else None
        if standard_input is None and output is None:
            continue

        pricing = TokenPricing(
            standard_input=standard_input,
            output=output,
            cached_input=money(cell(row, cache_hit_col) or ""),
            batch_input=money(cell(row, batch_in_col) or ""),
            batch_output=money(cell(row, batch_out_col) or ""),
        )
        model_id = _slugify(name)
        tier, threshold = _context_tier(raw_name)

        existing = models.get(model_id)
        if existing is None:
            models[model_id] = AIModel(
                id=model_id,
                name=name,
                vendor=Vendor.ANTHROPIC,
                pricing=pricing,
                free_tier=FreeTierPolicy(has_free_tier=False),
                status=status,
            )
        elif tier == "over":
            # Anthropic splits a context-tiered model across two rows that
            # share a name ("... up to 100,000 tokens" / "... over 100,000
            # tokens"). Fold the second into the first as tiered pricing
            # rather than letting one row silently shadow the other.
            existing.tiered_pricing = TieredPricing(
                threshold_tokens=threshold,
                base_rate=existing.pricing,
                over_threshold_rate=pricing,
            )
        elif tier == "base":
            existing.tiered_pricing = TieredPricing(
                threshold_tokens=threshold,
                base_rate=pricing,
                over_threshold_rate=existing.pricing,
            )
            existing.pricing = pricing

    if not models:
        raise ParseError("Found an Anthropic pricing table but no model rows in it")
    return list(models.values())


_GEMINI_HEADING_RE = re.compile(r"^##\s+(?P<heading>[^\n]+)$", re.MULTILINE)
_GEMINI_ID_ATTR_RE = re.compile(r"\{%\s*id=\"(?P<id>[^\"]+)\"\s*%\}")
_GEMINI_ID_FALLBACK_RE = re.compile(r"`(?P<id>gemini[a-z0-9.\-]*)`")
_GEMINI_TIERED_RE = re.compile(
    r"\$([\d.]+),\s*prompts\s*\\?<=\s*(\d+)k tokens\s*\$([\d.]+),\s*prompts\s*\\?>\s*\d+k"
)


def _gemini_cell_prices(cell: str):
    """Returns (base_rate, over_threshold_rate, threshold_tokens).

    The last two are ``None`` when the price isn't tiered by context length.
    The threshold is read from the cell rather than assumed to be 200k, so
    Google moving the boundary doesn't silently mis-price long prompts.
    """
    tiered = _GEMINI_TIERED_RE.search(cell)
    if tiered:
        return float(tiered.group(1)), float(tiered.group(3)), int(tiered.group(2)) * 1_000
    return money(cell), None, None


def _gemini_sections(text: str):
    """Yields ``(model_id, heading, section_text)`` for each '## <Model>' section.

    A section runs up to the next heading, so a model with no pricing table
    of its own can never borrow its neighbour's. The id is taken from the
    heading's Markdoc ``{% id="..." %}`` attribute when present (the current
    format) and otherwise from the first backticked ``gemini-*`` token in the
    section (the older ``*`gemini-2.5-flash`*`` convention) -- either layout
    parses.
    """
    headings = list(_GEMINI_HEADING_RE.finditer(text))
    for index, heading in enumerate(headings):
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        section = text[heading.start():end]
        raw_heading = heading.group("heading")

        attr = _GEMINI_ID_ATTR_RE.search(raw_heading)
        if attr:
            model_id = attr.group("id")
        else:
            fallback = _GEMINI_ID_FALLBACK_RE.search(section[:400])
            if not fallback:
                continue
            model_id = fallback.group("id")

        yield model_id, _GEMINI_ID_ATTR_RE.sub("", raw_heading).strip(), section


def _gemini_tab_rows(section: str, title: str = "Standard") -> Dict[str, List[str]]:
    """Returns one pricing tab's rows for a model, keyed by row label.

    Reads whichever shape the page is currently in: a Markdoc
    ``{% tab title="..." %}`` block (current) or a ``### <title>`` markdown
    table (previous). Returns ``{}`` when the section has neither -- e.g. a
    Live model billed only per audio minute, or a model with no Batch tier.
    """
    tab = re.search(
        r"\{%\s*tab\s+title=\"" + re.escape(title) + r"\"\s*%\}(?P<body>.*?)"
        r"(?=\{%\s*tab\s+title=|\{%\s*/\s*tabs\s*%\})",
        section,
        re.DOTALL,
    )
    if tab:
        tables = markdoc_tables(tab.group("body"))
        rows = tables[0] if tables else []
    else:
        table = find_table(section, r"### " + re.escape(title))
        rows = table_rows(table) if table else []

    return {row[0]: row for row in rows if len(row) >= 3}


@register_provider("google", Vendor.GOOGLE, GEMINI_URL, label="Google")
def parse_gemini(text: str) -> List[AIModel]:
    """Parses each '## <Model>' section's Standard-tier pricing.

    Sections without a Standard text-pricing table (image/video/audio
    specialty models billed per asset or per minute) are skipped -- those
    are out of scope for v1. A model's ``free_tier.has_free_tier`` comes
    directly from whether the doc marks its Input price row "Free of
    charge" vs "Not available", not a guess.

    Args:
        text: Raw markdown body fetched from :data:`GEMINI_URL`.

    Returns:
        One AIModel per parseable model section. Models priced past a
        context-length threshold get a populated ``tiered_pricing``, with
        the threshold read from the page rather than hardcoded.

    Raises:
        ParseError: If the document yields no models at all.
    """
    models = []
    for model_id, heading, section in _gemini_sections(text):
        if "gemini" not in model_id:
            continue
        name, status = _status_and_name(heading)

        rows = _gemini_tab_rows(section, "Standard")
        input_row = next((v for k, v in rows.items() if k.startswith("Input price")), None)
        output_row = next((v for k, v in rows.items() if k.startswith("Output price")), None)
        if not input_row or not output_row:
            continue

        body = section.lower()
        if "shut down" in body or "retired" in body:
            status = ModelStatus.RETIRED
        elif "deprecated" in body:
            status = ModelStatus.DEPRECATED

        cache_row = next((v for k, v in rows.items() if k.startswith("Context caching price")), None)

        # Batch pricing lives in its own tab, not a column of the Standard one.
        batch_rows = _gemini_tab_rows(section, "Batch")
        batch_input_row = next((v for k, v in batch_rows.items() if k.startswith("Input price")), None)
        batch_output_row = next((v for k, v in batch_rows.items() if k.startswith("Output price")), None)

        is_free = "free of charge" in input_row[1].lower()
        input_base, input_over, threshold = _gemini_cell_prices(input_row[2])
        output_base, output_over, _ = _gemini_cell_prices(output_row[2])

        pricing = TokenPricing(
            standard_input=input_base,
            output=output_base,
            cached_input=_gemini_cell_prices(cache_row[2])[0] if cache_row else None,
            batch_input=_gemini_cell_prices(batch_input_row[2])[0] if batch_input_row else None,
            batch_output=_gemini_cell_prices(batch_output_row[2])[0] if batch_output_row else None,
        )
        tiered = None
        if input_over is not None:
            tiered = TieredPricing(
                threshold_tokens=threshold or 200_000,
                base_rate=pricing,
                over_threshold_rate=TokenPricing(
                    standard_input=input_over, output=output_over or output_base
                ),
            )

        models.append(
            AIModel(
                id=model_id,
                name=name,
                vendor=Vendor.GOOGLE,
                pricing=pricing,
                tiered_pricing=tiered,
                free_tier=FreeTierPolicy(has_free_tier=is_free, data_used_for_training=is_free or None),
                status=status,
            )
        )

    if not models:
        raise ParseError(
            "Parsed no models from the Gemini pricing doc -- the per-model section / "
            "Standard pricing table structure this parser relies on has probably changed"
        )
    return models


def _section(text: str, heading: str) -> str:
    """Returns the text between ``heading`` and the next markdown heading.

    Lets a parser scope itself to one pricing tier ("Standard", "Batch")
    without caring how many tables that tier happens to be split across --
    a provider breaking one table into two, as OpenAI has, needs no change.
    """
    match = re.search(
        heading + r"[^\n]*\n(?P<body>.*?)(?=\n#{1,6}\s|\Z)", text, re.DOTALL | re.MULTILINE
    )
    return match.group("body") if match else ""


def _openai_price_rows(section: str) -> Dict[str, Dict[str, Optional[float]]]:
    """Reads every pricing table in ``section`` into ``{model: rates}``.

    Columns are located by header text, never position: OpenAI has already
    reordered these columns once (output used to follow cached-input, it
    now precedes it) and split one table into two. Both tables' layouts,
    and the next reshuffle, parse the same way.
    """
    found: Dict[str, Dict[str, Optional[float]]] = {}
    for table in re.findall(r"(?:^\|[^\n]*\|[ \t]*\n?)+", section, re.MULTILINE):
        rows = table_rows(table)
        if len(rows) < 2:
            continue
        header = [cell.lower() for cell in rows[0]]
        if not any("input" in cell for cell in header):
            continue

        def column(*keywords: str, exclude: Iterable[str] = ()) -> Optional[int]:
            """First header index containing every keyword and no excluded word.

            Prefers an explicitly short-context column, falling back to a
            plain one for tables that don't split by context length.
            """
            for wanted in (("short",) + keywords, keywords):
                for index, cell in enumerate(header):
                    if all(word in cell for word in wanted) and not any(word in cell for word in exclude):
                        return index
            return None

        input_col = column("input", exclude=("cached", "write", "long"))
        output_col = column("output", exclude=("long",))
        cached_col = column("cached", exclude=("long",))

        for row in rows[1:]:
            def cell(index: Optional[int]) -> str:
                return row[index] if index is not None and index < len(row) else ""

            name = row[0].strip()
            if not name or name in found:
                continue
            found[name] = {
                "input": money(cell(input_col)),
                "output": money(cell(output_col)),
                "cached": money(cell(cached_col)),
            }
    return found


@register_provider("openai", Vendor.OPENAI, OPENAI_URL, label="OpenAI")
def parse_openai(text: str) -> List[AIModel]:
    """Parses the Standard-tier pricing tables from the OpenAI pricing doc.

    Reads every table under the ``### Standard pricing data`` heading (the
    page currently splits its models across two with different column
    layouts), and merges in per-model batch rates from the ``#### Batch``
    table under ``### Batch pricing data`` where they exist.

    Only short-context columns are captured: the page doesn't state the
    token threshold at which long-context pricing applies, so it can't be
    turned into a ``tiered_pricing`` the way Gemini's is. Cache-*write*
    columns remain out of scope (see README's "Known limitations").

    Args:
        text: Raw doc body fetched from :data:`OPENAI_URL`.

    Returns:
        One AIModel per model row in the Standard pricing tables. The API
        has no free tier, so every model's ``free_tier.has_free_tier`` is False.

    Raises:
        ParseError: If the Standard pricing section can't be found, or
            contains no usable rows.
    """
    standard_section = _section(text, r"###\s*Standard pricing data")
    if not standard_section:
        raise ParseError("Could not find the standard-tier pricing rows in the OpenAI pricing doc")

    standard = _openai_price_rows(standard_section)
    batch = _openai_price_rows(_section(text, r"^####\s+Batch\s*$"))

    models = []
    for raw_name, rates in standard.items():
        if rates["input"] is None or rates["output"] is None:
            continue
        name, status = _status_and_name(raw_name)
        if not name:
            continue

        context_window = None
        context_match = re.search(r"<\s*([\d.]+)\s*([KM])\s*context", raw_name, re.IGNORECASE)
        if context_match:
            scale = 1_000 if context_match.group(2).upper() == "K" else 1_000_000
            context_window = int(float(context_match.group(1)) * scale)

        batch_rates = batch.get(raw_name, {})
        models.append(
            AIModel(
                id=name,
                name=name,
                vendor=Vendor.OPENAI,
                pricing=TokenPricing(
                    standard_input=rates["input"],
                    output=rates["output"],
                    cached_input=rates["cached"],
                    batch_input=batch_rates.get("input"),
                    batch_output=batch_rates.get("output"),
                ),
                context_window=context_window,
                free_tier=FreeTierPolicy(has_free_tier=False),
                status=status,
            )
        )

    if not models:
        raise ParseError("Found the OpenAI standard pricing section but no usable model rows in it")
    return models


# ---------------------------------------------------------------------------
# Build / CLI
# ---------------------------------------------------------------------------


def build_registry(models: List[AIModel], updated_at: Optional[str] = None) -> dict:
    """Wraps a flat list of models into the registry.json document shape.

    The document carries a ``schema_version`` so that a future format
    change can be rejected by older installs instead of half-read -- see
    :func:`llm_catalogue.catalogue.parse_registry`.
    """
    return {
        "schema_version": SCHEMA_VERSION,
        "updated_at": updated_at or date.today().isoformat(),
        "models": [m.to_dict() for m in models],
    }


def load_existing(path: Path = REGISTRY_PATH) -> List[AIModel]:
    """Loads the registry currently on disk, or [] if there isn't a readable one.

    Used as the fallback when a provider's parser breaks: its previous
    models are carried forward rather than dropped.
    """
    try:
        data = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    models = []
    for entry in data.get("models", []):
        try:
            models.append(AIModel.from_dict(entry))
        except ValueError:
            continue
    return models


def fetch(url: str, timeout: float = 15.0) -> str:
    """Fetches a pricing doc.

    Imports ``requests`` lazily so that merely importing this module (as the
    docs build and the test suite do) doesn't require the scraper extra.
    """
    try:
        import requests
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise RuntimeError(
            "The scraper needs the 'requests' package: pip install -e '.[scraper]'"
        ) from exc

    response = requests.get(url, timeout=timeout, headers={"User-Agent": "llm-catalogue-scraper"})
    response.raise_for_status()
    return response.text


def scrape_provider(provider: Provider, previous: List[AIModel], timeout: float = 15.0) -> tuple:
    """Fetches and parses one provider, with a guard against silent collapse.

    Args:
        provider: The provider to run.
        previous: Models already in the registry for this provider's vendor,
            used both as the regression baseline and as the fallback.
        timeout: HTTP timeout in seconds.

    Returns:
        ``(models, error)``. On success ``error`` is ``None``. On failure --
        a network error, a :class:`ParseError`, or a model count that
        collapsed below :data:`REGRESSION_THRESHOLD` of ``previous`` --
        ``models`` is ``previous`` (carried forward, never silently
        dropped) and ``error`` describes what went wrong.
    """
    try:
        models = provider.parse(fetch(provider.url, timeout=timeout))
    except Exception as exc:  # noqa: BLE001 -- one provider's outage isn't the run's
        return previous, f"{type(exc).__name__}: {exc}"

    if previous and len(models) < len(previous) * REGRESSION_THRESHOLD:
        return previous, (
            f"parsed only {len(models)} models, down from {len(previous)} in the current "
            f"registry -- treating as a parse failure, not a real change. Re-run with "
            f"--allow-partial if {provider.label} really did retire that many models."
        )
    return models, None


def main(argv: Optional[List[str]] = None) -> int:
    """Fetches every registered provider's pricing doc and rebuilds the registry.

    Requires the ``scraper`` extra (``pip install -e ".[scraper]"``) and
    network access to each provider's docs.

    Returns:
        ``0`` if every provider parsed cleanly, ``1`` otherwise -- so CI can
        catch a provider's page redesign before users get a stale registry.
    """
    parser = argparse.ArgumentParser(
        prog="python -m llm_catalogue.scraper",
        description="Rebuild registry.json from each provider's public pricing docs.",
    )
    parser.add_argument(
        "--only",
        action="append",
        choices=sorted(PROVIDERS),
        help="Only scrape this provider (repeatable). Others keep their current entries.",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Parse every provider and report, but don't write the registry. Exits 1 if any parser broke.",
    )
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="Write the registry even if a provider failed, keeping that provider's existing entries.",
    )
    parser.add_argument("--output", type=Path, default=REGISTRY_PATH, help="Where to write the registry.")
    parser.add_argument("--timeout", type=float, default=15.0, help="HTTP timeout in seconds.")
    args = parser.parse_args(argv)

    existing = load_existing(args.output)
    selected = [p for key, p in PROVIDERS.items() if not args.only or key in args.only]

    models: List[AIModel] = []
    failures: Dict[str, str] = {}
    scraped_vendors = set()

    for provider in selected:
        previous = [m for m in existing if m.vendor == provider.vendor]
        parsed, error = scrape_provider(provider, previous, timeout=args.timeout)
        scraped_vendors.add(provider.vendor)
        models.extend(parsed)
        if error:
            failures[provider.label] = error
            print(f"{provider.label}: FAILED -- {error}", file=sys.stderr)
            if previous:
                print(f"{provider.label}: keeping {len(previous)} existing models", file=sys.stderr)
        else:
            delta = len(parsed) - len(previous)
            change = f" ({delta:+d})" if previous and delta else ""
            print(f"{provider.label}: parsed {len(parsed)} models{change}")

    # Vendors not scraped in this run (--only, or a provider removed from the
    # code) keep whatever the registry already had for them.
    models.extend(m for m in existing if m.vendor not in scraped_vendors)

    if args.check:
        print(f"Checked {len(selected)} providers, {len(failures)} failing.")
        return 1 if failures else 0

    if failures and not args.allow_partial:
        print(
            f"\nRefusing to write {args.output}: {len(failures)} provider(s) failed to parse. "
            "Fix the parser, or re-run with --allow-partial to keep their existing entries.",
            file=sys.stderr,
        )
        return 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(build_registry(models), indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(models)} models to {args.output}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
