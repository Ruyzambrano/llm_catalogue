# LLM Catalogue

One of the greatest issues of today is deciding which AI model to use, and navigating a constant, ever changing tier system. I became frustrated trying to hard code which Gemini models were a part of the free tier, and getting stuck in an argument with ChatGPT lying to me about how it actually does have a free tier. So I went and figured it out myself.

Model names, pricing, and free-tier metadata for OpenAI, Anthropic, and Google
Gemini, in one small zero-dependency package.

```bash
pip install llm-catalogue
```

Full API reference (generated from docstrings): see [Documentation](#documentation) below.

## Usage

```python
from llm_catalogue import Catalog

catalog = Catalog()

# All free-tier-eligible models for a provider ([] if none)
free_gemini = catalog.get_free_models("google")
free_openai = catalog.get_free_models("openai")  # -> []

# All models for a provider
claude_models = catalog.get_models("anthropic")

# UI toggle helper
if catalog.has_free_tier("google"):
    ...

# Free models across every provider
for model in catalog.find_free_models():
    print(model.id, model.vendor.value)

# Look up one model directly
model = catalog.get_model("gemini-2.5-flash")
```

`get_models`/`get_free_models`/`has_free_tier` accept `"openai"`, `"anthropic"`
(or `"claude"`), and `"google"` (or `"gemini"`).

## Estimating request cost

Every `AIModel` can price a request via `calculate_cost()`, which accounts for
prompt caching, batch pricing, and context-length tiering automatically:

```python
model = catalog.get_model("gemini-2.5-pro")

# Standard request
model.calculate_cost(input_tokens=50_000, output_tokens=2_000)

# Half the input tokens were served from a prompt cache
model.calculate_cost(input_tokens=50_000, output_tokens=2_000, cached_tokens=25_000)

# Via the batch API (uses pricing.batch_input/batch_output instead)
model.calculate_cost(input_tokens=50_000, output_tokens=2_000, is_batch=True)

# Over the model's context-length threshold -- automatically picks up
# tiered_pricing.over_threshold_rate instead of the base rate
model.calculate_cost(input_tokens=250_000, output_tokens=2_000)
```

Cost is returned in USD, rounded to 6 decimal places. See the `AIModel` API
reference for exactly how each argument affects the rate used.

## Data freshness

`Catalog()` never makes a network call — it reads the `registry.json` bundled
with the package (or a previously cached one under `~/.cache/llm_catalogue/`),
so imports stay fast and offline-safe. To pull the latest data from GitHub:

```python
catalog = Catalog(auto_update=True)   # fetch on construction
catalog.refresh()                     # or fetch explicitly, any time
catalog.refresh(force=True)           # bypass the 24h cache TTL
```

`refresh()` never raises — on failure (offline, timeout, bad response) it
leaves the currently loaded data untouched and returns `False`.

## registry.json schema

`Catalog` loads this file at `src/llm_catalogue/data/registry.json`. It's a
plain JSON document, so you can also read it directly without the package:

| Field | Type | Notes |
|---|---|---|
| `updated_at` | string | ISO date the registry was last rebuilt. |
| `models` | array | List of model objects, described below. |

Each entry in `models` matches `AIModel.to_dict()`:

| Field | Type | Notes |
|---|---|---|
| `id` | string | Provider-native model id, e.g. `"gpt-4o"`. |
| `name` | string | Human-readable display name. |
| `vendor` | string | One of `"openai"`, `"anthropic"`, `"google"`. |
| `pricing` | object | `TokenPricing`: `standard_input`, `output`, `cached_input`, `batch_input`, `batch_output` (USD per 1M tokens; nulls where unknown/not applicable). |
| `context_window` | int or null | Max input tokens, where documented. |
| `tiered_pricing` | object or null | `{threshold_tokens, base_rate, over_threshold_rate}` for models with context-length-dependent pricing. |
| `free_tier` | object or null | `{has_free_tier, rate_limit_rpm, data_used_for_training}`. |
| `status` | string | `"active"`, `"deprecated"`, `"retired"`, or `"limited_availability"`. |
| `tool_costs` | object | Reserved for per-tool pricing; empty in v1. |

## Documentation

Full API docs are generated from the docstrings on `Catalog`, `AIModel`,
`TokenPricing`, `TieredPricing`, and `FreeTierPolicy` via
[mkdocstrings](https://mkdocstrings.github.io/). To browse them locally:

```bash
pip install -e ".[docs]"
mkdocs serve
```

then open http://127.0.0.1:8000. `mkdocs build` produces a static site under
`site/` you can host anywhere (e.g. GitHub Pages).

## Contributing / keeping the registry up to date

Project layout:

```text
src/llm_catalogue/
  models.py     # AIModel, TokenPricing, TieredPricing, FreeTierPolicy, Vendor, ModelStatus
  catalogue.py  # Catalog -- the main entry point
  scraper.py    # dev-only tool that rebuilds data/registry.json
  data/registry.json
tests/          # pytest
docs/           # mkdocs source
```

Run the test suite:

```bash
pip install -e ".[dev]"
pytest
```

Refresh the bundled pricing data from each provider's live docs:

```bash
pip install -e ".[scraper]"
python -m llm_catalogue.scraper
```

This overwrites `src/llm_catalogue/data/registry.json` from:

- [Gemini pricing](https://ai.google.dev/gemini-api/docs/pricing.md.txt)
- [Claude pricing](https://platform.claude.com/docs/en/about-claude/pricing.md)
- [OpenAI pricing](https://developers.openai.com/api/docs/pricing.md)

Review the diff, commit it, and cut a new release so it ships in the next
`pip install`.

## Scope and known limitations (v1)

- Only "Standard" tier, text-in/text-out pricing is captured. Batch pricing is
  included where the source table has it; Flex/Priority tiers are not.
- Multimodal, audio, image, video, and embedding-specialist models are out of
  scope — this tracks general-purpose chat/text LLMs.
- `cached_input` is the cache-*read* price. Separate cache-*write* premiums
  (e.g. Anthropic's 5m/1h cache writes, OpenAI's gpt-5.6-family write cost)
  aren't modelled yet.
- `free_tier.rate_limit_rpm` isn't populated — Gemini's free-tier RPM limits
  live on a separate rate-limits doc this scraper doesn't fetch yet.
- Gemini's tiered (>200k token) pricing is captured via `tiered_pricing`;
  OpenAI's `<272K context length` models are recorded with `context_window`
  but don't have a documented over-the-limit rate, so they aren't tiered.

Data last refreshed: 2026-07-26.
