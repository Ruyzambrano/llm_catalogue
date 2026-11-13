# llm-catalogue

Model names, pricing, and free-tier metadata for OpenAI, Anthropic, and Google
Gemini, in one small zero-dependency package -- built so that a provider
changing its lineup, its pricing page, or its data format doesn't break your
install.

```bash
pip install llm-catalogue
```

```python
from llm_catalogue import Catalog

catalog = Catalog()
catalog.get_free_models("google")   # -> [<AIModel gemini-2.5-flash>, ...]
catalog.get_free_models("openai")   # -> []
```

This site documents the public API. See the project [README](https://github.com/Ruyzambrano/llm_catalogue#readme)
for full usage examples, the `registry.json` schema, and how to refresh the
bundled pricing data.

## Where to start

- [`Catalog`](api/catalog.md) — the main entry point: querying models by
  vendor, free-tier filtering, refreshing data.
- [Models](api/models.md) — the data returned by every `Catalog` method:
  `AIModel`, `TokenPricing`, `TieredPricing`, `FreeTierPolicy`, `Vendor`,
  `ModelStatus`.
- [Scraper](api/scraper.md) — the dev-only tool that rebuilds
  `registry.json` from each provider's pricing docs, and the one-decorator
  extension point for adding a provider. Not used at runtime.

## Surviving provider changes

Unknown vendors, statuses and pricing fields in the registry are accepted and
preserved rather than rejected, so `Catalog.refresh()` can safely pull data
published after your installed version. The scraper, for its part, treats a
parser that suddenly finds nothing as a failure rather than as a provider with
no models. The README has the
[full table of behaviours](https://github.com/Ruyzambrano/llm_catalogue#handling-provider-changes).
