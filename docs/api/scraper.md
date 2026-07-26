# Scraper

Dev-only tool that rebuilds `data/registry.json` from each provider's public
pricing docs. Not a runtime dependency of `llm_catalogue` -- see the
[README](https://github.com/Ruyzambrano/llm_catalogue#contributing--keeping-the-registry-up-to-date)
for how to run it.

::: llm_catalogue.scraper
    options:
      members:
        - parse_anthropic
        - parse_gemini
        - parse_openai
        - build_registry
        - main
