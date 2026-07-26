# llm-catalogue

Model names, pricing, and free-tier metadata for OpenAI, Anthropic, and Google
Gemini, in one small zero-dependency package.

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
  `registry.json` from each provider's pricing docs. Not used at runtime.
