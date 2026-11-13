# LLM Catalogue

One of the greatest issues of today is deciding which AI model to use, and navigating a constant, ever changing tier system. I became frustrated trying to hard code which Gemini models were a part of the free tier, and getting stuck in an argument with ChatGPT lying to me about how it actually does have a free tier. So I went and figured it out myself.

Model names, pricing, and free-tier metadata for OpenAI, Anthropic, and Google
Gemini, in one small zero-dependency package -- built so that a provider
changing its lineup, its pricing page, or its whole schema doesn't break your
install. See [Handling provider changes](#handling-provider-changes).

```bash
pip install llm-catalogue
```

Full API reference (generated from docstrings): see [Documentation](#documentation) below.

## Usage

```python
from llm_catalogue import Catalog

# auto_update=True keeps a long-running app (a server, a Streamlit app, a
# notebook you leave open) from silently going stale -- it checks for fresh
# data on construction and is cheap to call again later (see "Data
# freshness" below). Leave it off (the default) for tests, scripts, and
# anywhere you want zero network calls.
catalog = Catalog(auto_update=True)

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
(or `"claude"`), and `"google"` (or `"gemini"`, `"vertex"`) -- plus the name of
any other provider present in the loaded registry, so data listing a provider
newer than your installed version is still queryable. Use `catalog.vendors()`
to see what the loaded data actually contains instead of hardcoding a list.

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

`Catalog()` on its own never makes a network call — it reads the
`registry.json` bundled with the package (or a previously cached one under
`~/.cache/llm_catalogue/`), so imports stay fast and offline-safe. That's the
right default for tests and one-off scripts, but it also means a plain
`Catalog()` in a long-running process will happily serve data as old as
whatever was bundled when it was installed. For anything long-running, use
`auto_update=True` (see Usage above) or call `refresh()` yourself:

```python
catalog = Catalog(auto_update=True)   # fetch on construction
catalog.refresh()                     # or fetch explicitly, any time
catalog.refresh(force=True)           # bypass the 1h cache TTL
```

`refresh()` checks a local cache file's age before touching the network, so
calling it liberally (e.g. once per request) is cheap -- it's a file-stat
check that only becomes an actual HTTP request once the 1-hour TTL
(`cache_ttl_seconds`) has elapsed. It never raises: on failure (offline,
timeout, bad response) it leaves the currently loaded data untouched and
returns `False`.

## Handling provider changes

Providers add models, retire them, rename tiers, reorder pricing tables and
occasionally rebuild their docs from scratch. Two separate mechanisms keep
that from reaching you as a crash or as silently wrong data.

### Reading data newer than your install

`Catalog.refresh()` feeds a registry published *today* into whatever version
of this package you installed *whenever*, so the data model accepts things it
has never seen:

| If the registry contains... | ...this is what happens |
|---|---|
| A provider this release doesn't know (`"mistral"`) | Loads normally. `Vendor("mistral")` works, `catalog.get_models("mistral")` works, `vendor.is_known` is `False`. |
| A status this release doesn't know (`"sunset"`) | Preserved verbatim on `model.status`. |
| A pricing field this release doesn't model | Kept in `model.pricing.extra` / `model.extra`, and written back out by `to_dict()` -- nothing is dropped on round trip. |
| A model with no published price yet | Loads with `is_priced == False`; `calculate_cost()` raises a clear error rather than inventing a number. |
| One malformed entry | Skipped, listed in `catalog.load_errors`; the other models still load. |
| A corrupt or truncated cache file | Ignored in favour of the bundled registry. |
| A `schema_version` newer than this release understands | Refused over the network (your current data stays in place, `refresh()` returns `False`); read best-effort with a warning from local disk. |
| An empty or broken registry published upstream | `refresh()` returns `False` and the bad response is **not** cached, so one bad deploy can't poison every later start. |

Two smaller things in the same spirit: after `pip install -U`, a stale cache
no longer shadows the fresher registry that shipped with the upgrade (the
newer `updated_at` wins), and `get_model()` tolerates gateway-style ids, so
`"anthropic/claude-opus-5"` and `"Claude-Opus-5"` resolve like `"claude-opus-5"`.

Point `refresh()` at a mirror or an internal registry with the
`LLM_CATALOGUE_REGISTRY_URL` environment variable (or `Catalog(registry_url=...)`)
-- each URL gets its own cache file.

### When a provider redesigns its pricing page

The scraper reads pages written for humans, so every parser is a standing bet
that will eventually be wrong. The design makes that loud instead of quiet:

- **A parser that finds nothing raises `ParseError`.** Returning `[]` would
  have erased that provider from the registry on the next run.
- **A collapsed model count is treated as a parse failure.** If a provider
  returns under half the models already in the registry, the run fails rather
  than deleting the rest (override with `--allow-partial`).
- **A failed provider keeps its existing entries** and the registry is not
  overwritten; the process exits non-zero.
- **Columns are matched by header text, never position**, so adding or
  reordering a pricing column changes nothing.
- **Both layouts parse** where a provider has switched formats -- Google's
  Markdoc `{% table %}` blocks and the markdown tables they replaced.
- **A weekly CI job** (`.github/workflows/registry-check.yml`) runs
  `--check` against the live pages, so a redesign surfaces as a failed run
  rather than as stale data in the next release.

```bash
python -m llm_catalogue.scraper --check          # parse everything, write nothing, exit 1 on breakage
python -m llm_catalogue.scraper --only google    # rebuild one provider, leave the others' entries alone
python -m llm_catalogue.scraper --allow-partial  # write even if a provider failed, keeping its old entries
```

### Adding a provider

One decorated function, in `scraper.py`. Nothing else to edit -- `main()`
iterates the registry of providers, and `Vendor` accepts names it doesn't
declare:

```python
@register_provider("mistral", Vendor("mistral"), "https://docs.mistral.ai/pricing.md")
def parse_mistral(text: str) -> list[AIModel]:
    rows = table_rows(find_table(text, r"## Pricing"))
    return [
        AIModel(
            id=row[0],
            name=row[0],
            vendor=Vendor("mistral"),
            pricing=TokenPricing(standard_input=money(row[1]), output=money(row[2])),
        )
        for row in rows[1:]
    ]
```

`table_rows`, `markdoc_tables`, `find_table` and `money` are shared helpers;
`money` handles `$3.50`, `$3.50 / MTok` and `$1,250`, and returns `None` for
"not offered" cells.

## registry.json schema

`Catalog` loads this file at `src/llm_catalogue/data/registry.json`. It's a
plain JSON document, so you can also read it directly without the package:

| Field | Type | Notes |
|---|---|---|
| `schema_version` | int | Format version (currently `1`). A registry declaring a higher one is refused by older installs rather than half-read. Absent means `1`. |
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
| `status` | string | `"active"`, `"deprecated"`, `"retired"`, `"limited_availability"`, or any other string a provider's docs start using. |
| `tool_costs` | object | Reserved for per-tool pricing; empty in v1. |

Any key not listed here is preserved rather than dropped: unrecognized
model-level keys land in `AIModel.extra` and unrecognized pricing keys in
`TokenPricing.extra`, and both are written back out by `to_dict()`. That is
what lets a field added to this schema next year survive a round trip through
this year's code.

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
  _version.py   # the one place the version number lives
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

It exits non-zero if any provider failed to parse, and in that case does not
overwrite the registry -- see [When a provider redesigns its pricing
page](#when-a-provider-redesigns-its-pricing-page) for the flags.

Review the diff, commit it, and cut a new release so it ships in the next
`pip install`. Bump the version in `src/llm_catalogue/_version.py`; the build
and the refresh User-Agent both read it from there.

## Scope and known limitations (v1)

- Only "Standard" tier, text-in/text-out pricing is captured, plus Batch
  pricing where the provider publishes it (currently OpenAI and Gemini).
  Flex/Priority tiers are not.
- Multimodal, audio, image, video, and embedding-specialist models are out of
  scope — this tracks general-purpose chat/text LLMs.
- `cached_input` is the cache-*read* price. Separate cache-*write* premiums
  (e.g. Anthropic's 5m/1h cache writes, OpenAI's gpt-5.6-family write cost)
  aren't modelled yet.
- `free_tier.rate_limit_rpm` isn't populated — Gemini's free-tier RPM limits
  live on a separate rate-limits doc this scraper doesn't fetch yet.
- Context-length tiering is captured via `tiered_pricing` for Gemini (whose
  threshold is read off the page, not assumed to be 200k) and for Anthropic
  models the docs split across "up to N tokens" / "over N tokens" rows.
  OpenAI's pricing page also splits Standard/Batch/Flex/Priority pricing into
  "Short context" and "Long context" columns, but doesn't state the token
  threshold where Long-context pricing kicks in, so only Short-context
  (base) pricing is captured; `<272K context length` models still get
  `context_window` set, but aren't tiered.

Data last refreshed: 2026-11-13.
