"""Dev-only tool that rebuilds ``data/registry.json`` from each provider's
public pricing docs.

Not a runtime dependency of llm_catalogue -- the published package only ships
the bundled ``registry.json``, it never scrapes at import time. Install the
extra scraping deps and run this module directly to refresh the bundled data:

    pip install -e ".[scraper]"
    python -m llm_catalogue.scraper

Only "Standard" tier, text-in/text-out pricing is captured. Batch pricing is
captured where the source table has it. Multimodal/audio/image/video/embedding
models and cache-write premiums are intentionally out of scope for v1 -- see
README.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import List, Optional

import requests

from llm_catalogue.models import AIModel, FreeTierPolicy, ModelStatus, TieredPricing, TokenPricing, Vendor

GEMINI_URL = "https://ai.google.dev/gemini-api/docs/pricing.md.txt"
ANTHROPIC_URL = "https://platform.claude.com/docs/en/about-claude/pricing.md"
OPENAI_URL = "https://developers.openai.com/api/docs/pricing.md"

REGISTRY_PATH = Path(__file__).parent / "data" / "registry.json"

_PRICE_RE = re.compile(r"\$([\d.]+)\s*/\s*MTok")


def _price(cell: str) -> Optional[float]:
    """Extracts a "$X / MTok"-style dollar amount from an Anthropic table cell."""
    match = _PRICE_RE.search(cell or "")
    return float(match.group(1)) if match else None


def _status_and_name(raw_name: str) -> tuple[str, ModelStatus]:
    """Splits a raw provider-doc name into a clean display name and status.

    Handles the "Name (annotation)" convention both Anthropic and OpenAI use
    for lifecycle notes, e.g. "Claude Opus 4.1 ([deprecated](...))" or
    "gpt-5.5 (<272K context length)". Everything from the first "(" onward
    is dropped from the name; ``status`` is only set from keywords, so a
    purely descriptive parenthetical (like the context-length one) leaves
    status as ACTIVE.
    """
    lowered = raw_name.lower()
    if "retired" in lowered:
        status = ModelStatus.RETIRED
    elif "deprecated" in lowered:
        status = ModelStatus.DEPRECATED
    elif "limited availability" in lowered:
        status = ModelStatus.LIMITED
    else:
        status = ModelStatus.ACTIVE
    name = raw_name.split("(")[0].strip()
    return name, status


def _slugify(name: str) -> str:
    """Turns a display name like "Claude Opus 4.1" into "claude-opus-4-1".

    Anthropic's pricing page doesn't publish machine-readable model ids the
    way Google and OpenAI's docs do, so this derives one from the display name.
    """
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def parse_anthropic(text: str) -> List[AIModel]:
    """Parses the standard-context pricing table from platform.claude.com.

    Args:
        text: Raw markdown body fetched from :data:`ANTHROPIC_URL`.

    Returns:
        One AIModel per row in the pricing table. ``cached_input`` is taken
        from the "Cache Hits & Refreshes" column; the 5m/1h cache-*write*
        columns and the separate Batch API discount aren't captured (see
        README's "Known limitations").

    Raises:
        ValueError: If no markdown table can be found in ``text`` at all
            (e.g. the page structure changed).
    """
    match = re.search(r"(?:^\|[^\n]+\|\n?)+", text, re.MULTILINE)
    if not match:
        raise ValueError("Could not find a pricing table in the Anthropic pricing doc")

    rows = [r for r in match.group(0).splitlines() if r.startswith("| Claude")]
    models = []
    for row in rows:
        raw_name, base_input, _cache_5m, _cache_1h, cache_hit, output = (
            c.strip() for c in row.split("|")[1:-1]
        )
        name, status = _status_and_name(raw_name)
        models.append(
            AIModel(
                id=_slugify(name),
                name=name,
                vendor=Vendor.ANTHROPIC,
                pricing=TokenPricing(
                    standard_input=_price(base_input),
                    output=_price(output),
                    cached_input=_price(cache_hit),
                ),
                free_tier=FreeTierPolicy(has_free_tier=False),
                status=status,
            )
        )
    return models


_GEMINI_HEADER_RE = re.compile(r"^## (?P<name>[^\n]+)\n\n\*`(?P<id>[a-z0-9.\-]+)`", re.MULTILINE)
_GEMINI_STANDARD_TABLE_RE = re.compile(r"### Standard\n\n(?P<table>\|.+?)(?:\n\n|\Z)", re.DOTALL)
_GEMINI_TIERED_RE = re.compile(
    r"\$([\d.]+),\s*prompts\s*\\?<=\s*200k tokens\s*\$([\d.]+),\s*prompts\s*\\?>\s*200k"
)


def _gemini_cell_prices(cell: str) -> tuple[Optional[float], Optional[float]]:
    """Returns (base_rate, over_200k_rate); over_200k_rate is None when the
    price isn't tiered by context length."""
    tiered = _GEMINI_TIERED_RE.search(cell)
    if tiered:
        return float(tiered.group(1)), float(tiered.group(2))
    plain = re.search(r"\$([\d.]+)", cell)
    return (float(plain.group(1)) if plain else None), None


def parse_gemini(text: str) -> List[AIModel]:
    """Parses each '## <Model>' section's '### Standard' pricing table.

    Each section spans from its '## <Model>' heading up to (but not past)
    the next one, so a section with no Standard table of its own -- e.g. a
    Live/Translate model billed only per audio minute -- is skipped rather
    than accidentally matching its neighbour's table. Skips sections without
    a Standard text-pricing table (image/video/audio specialty models) --
    those are out of scope for v1. A model's ``free_tier.has_free_tier``
    comes directly from whether the doc marks its Input price row "Free of
    charge" vs "Not available", not a guess.

    Args:
        text: Raw markdown body fetched from :data:`GEMINI_URL`.

    Returns:
        One AIModel per parseable "## <Model>" section. Models priced past
        a 200k-token threshold get a populated ``tiered_pricing``.
    """
    headers = list(_GEMINI_HEADER_RE.finditer(text))
    models = []
    for i, header in enumerate(headers):
        model_id = header.group("id")
        if "gemini-" not in model_id:
            continue
        name, status = _status_and_name(header.group("name"))

        section_end = headers[i + 1].start() if i + 1 < len(headers) else len(text)
        section = text[header.start():section_end]

        table_match = _GEMINI_STANDARD_TABLE_RE.search(section)
        if not table_match:
            continue
        table = table_match.group("table")

        body = section.lower()
        if "shut down" in body or "retired" in body:
            status = ModelStatus.RETIRED
        elif "deprecated" in body:
            status = ModelStatus.DEPRECATED

        rows = {}
        for line in table.splitlines():
            cells = [c.strip() for c in line.strip().strip("|").split("|")]
            if len(cells) >= 3:
                rows[cells[0]] = cells

        input_row = rows.get("Input price") or rows.get("Input price (text, image, video)")
        output_row = next((v for k, v in rows.items() if k.startswith("Output price")), None)
        cache_row = rows.get("Context caching price")
        if not input_row or not output_row:
            continue

        is_free = "free of charge" in input_row[1].lower()
        input_base, input_over = _gemini_cell_prices(input_row[2])
        output_base, output_over = _gemini_cell_prices(output_row[2])
        cache_base = _gemini_cell_prices(cache_row[2])[0] if cache_row else None

        pricing = TokenPricing(standard_input=input_base, output=output_base, cached_input=cache_base)
        tiered = None
        if input_over is not None:
            tiered = TieredPricing(
                threshold_tokens=200_000,
                base_rate=pricing,
                over_threshold_rate=TokenPricing(standard_input=input_over, output=output_over or output_base),
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
    return models


_OPENAI_STANDARD_TABLE_RE = re.compile(r"### Standard pricing data\n\n(?P<table>\|.+?)(?:\n\n|\Z)", re.DOTALL)
_OPENAI_PRICE_RE = re.compile(r"\$([\d.]+)")


def _openai_price(cell: str) -> Optional[float]:
    """Extracts a plain "$X"-style dollar amount from an OpenAI table cell,
    or ``None`` for a "-" (not offered) cell."""
    match = _OPENAI_PRICE_RE.search(cell or "")
    return float(match.group(1)) if match else None


def parse_openai(text: str) -> List[AIModel]:
    """Parses the 'Flagship models' Standard-tier markdown table (heading
    ``### Standard pricing data``) from the OpenAI pricing doc.

    As of the 2026-07 page redesign this is a plain markdown table with nine
    columns -- model, then Short-context input/cached-input/cache-write/output,
    then the same four for Long-context -- replacing the JS ``rows={[[...]]}``
    literal the page used to embed instead. Only the Short-context input,
    cached-input, and output columns are captured: the page doesn't state the
    token threshold at which Long-context pricing applies, so it can't be
    turned into a ``tiered_pricing`` the way Gemini's is; cache-write columns
    remain out of scope too (see README's "Known limitations").

    Args:
        text: Raw doc body fetched from :data:`OPENAI_URL`.

    Returns:
        One AIModel per row in the Standard pricing table. The API has no
        free tier, so every model's ``free_tier.has_free_tier`` is False.

    Raises:
        ValueError: If the ``### Standard pricing data`` table can't be
            found (e.g. the page structure changed again).
    """
    match = _OPENAI_STANDARD_TABLE_RE.search(text)
    if not match:
        raise ValueError("Could not find the standard-tier pricing rows in the OpenAI pricing doc")

    rows = [r for r in match.group("table").splitlines() if r.startswith("|") and "---" not in r]

    models = []
    for row in rows[1:]:  # rows[0] is the header row
        cells = [c.strip() for c in row.strip().strip("|").split("|")]
        if len(cells) < 5:
            continue

        raw_name = cells[0]
        name, status = _status_and_name(raw_name)
        context_window = 272_000 if "272K context length" in raw_name else None

        standard_input = _openai_price(cells[1])
        cached_input = _openai_price(cells[2])
        output = _openai_price(cells[4])
        if standard_input is None or output is None:
            continue

        models.append(
            AIModel(
                id=name,
                name=name,
                vendor=Vendor.OPENAI,
                pricing=TokenPricing(standard_input=standard_input, output=output, cached_input=cached_input),
                context_window=context_window,
                free_tier=FreeTierPolicy(has_free_tier=False),
                status=status,
            )
        )
    return models


def build_registry(models: List[AIModel]) -> dict:
    """Wraps a flat list of models into the registry.json document shape:
    ``{"updated_at": <today's date>, "models": [...]}``.
    """
    from datetime import date

    return {
        "updated_at": date.today().isoformat(),
        "models": [m.to_dict() for m in models],
    }


def main() -> None:
    """Fetches all three providers' pricing docs, parses them, and overwrites
    ``src/llm_catalogue/data/registry.json`` with the result.

    Requires the ``scraper`` extra (``pip install -e ".[scraper]"``) and
    network access to ``ai.google.dev``, ``platform.claude.com``, and
    ``developers.openai.com``.
    """
    models: List[AIModel] = []
    for label, url, parser in (
        ("Anthropic", ANTHROPIC_URL, parse_anthropic),
        ("Google", GEMINI_URL, parse_gemini),
        ("OpenAI", OPENAI_URL, parse_openai),
    ):
        response = requests.get(url, timeout=15)
        response.raise_for_status()
        parsed = parser(response.text)
        print(f"{label}: parsed {len(parsed)} models")
        models.extend(parsed)

    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(build_registry(models), indent=2))
    print(f"Wrote {len(models)} models to {REGISTRY_PATH}")


if __name__ == "__main__":
    main()
