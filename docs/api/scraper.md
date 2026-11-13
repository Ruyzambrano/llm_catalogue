# Scraper

Dev-only tool that rebuilds `data/registry.json` from each provider's public
pricing docs. Not a runtime dependency of `llm_catalogue` -- see the
[README](https://github.com/Ruyzambrano/llm_catalogue#contributing--keeping-the-registry-up-to-date)
for how to run it.

Adding a provider means writing a parser and decorating it with
[`register_provider`](#llm_catalogue.scraper.register_provider); nothing else
needs editing. Parsers raise [`ParseError`](#llm_catalogue.scraper.ParseError)
rather than returning an empty list, so a redesigned page fails loudly instead
of quietly deleting that provider from the registry.

::: llm_catalogue.scraper
    options:
      members:
        - register_provider
        - Provider
        - ParseError
        - parse_anthropic
        - parse_gemini
        - parse_openai
        - scrape_provider
        - build_registry
        - main

## Shared parsing helpers

Small, format-agnostic pieces a new parser can build on.

::: llm_catalogue.scraper
    options:
      show_root_heading: false
      members:
        - money
        - table_rows
        - markdoc_tables
        - find_table
