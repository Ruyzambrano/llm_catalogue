import json

import pytest

from llm_catalogue import scraper
from llm_catalogue.models import AIModel, TokenPricing, Vendor
from llm_catalogue.scraper import (
    PROVIDERS,
    ParseError,
    Provider,
    parse_anthropic,
    parse_gemini,
    parse_openai,
    register_provider,
    scrape_provider,
)

_TWO_MODEL_TEXT = """\
## Gemini Example Live

*`gemini-example-live`*

No Standard tier -- priced by audio minute only.

|   | Free Tier | Paid Tier, per 1M tokens in USD |
|---|---|---|
| Input price | Free of charge | $3.50 |
| Output price (including thinking tokens) | Free of charge | $21.00 |

## Gemini Example Lite

*`gemini-example-lite`*

### Standard

|   | Free Tier | Paid Tier, per 1M tokens in USD |
|---|---|---|
| Input price | Free of charge | $0.30 |
| Output price (including thinking tokens) | Free of charge | $2.50 |
"""


def test_section_without_standard_table_is_skipped_not_merged():
    """A section with no '### Standard' table of its own (e.g. a Live/Translate
    model billed per audio minute, not per-token) must be skipped entirely --
    not matched against the *next* section's Standard table, and not swallow
    that next section's heading in the process."""
    models = parse_gemini(_TWO_MODEL_TEXT)
    ids = [m.id for m in models]

    assert "gemini-example-live" not in ids
    assert ids == ["gemini-example-lite"]

    lite = models[0]
    assert lite.pricing.standard_input == 0.3
    assert lite.pricing.output == 2.5


_OPENAI_STANDARD_TABLE_TEXT = """\
Standard

### Standard pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.6-sol | $5.00 | $0.50 | $6.25 | $30.00 | $10.00 | $1.00 | $12.50 | $45.00 |
| gpt-5.5 (<272K context length) | $5.00 | $0.50 | - | $30.00 | $10.00 | $1.00 | - | $45.00 |
| gpt-4o | $2.50 | $1.25 | - | $10.00 | - | - | - | - |

Regional processing (data residency) endpoints are charged a 10% uplift.

Batch

### Batch pricing data

| Model | Short context input | Short context cached input | Short context cache writes | Short context output | Long context input | Long context cached input | Long context cache writes | Long context output |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| gpt-5.6-sol | $2.50 | $0.25 | $3.125 | $15.00 | $5.00 | $0.50 | $6.25 | $22.50 |
"""


def test_parse_openai_reads_short_context_columns_only():
    """The Standard table has separate Short-context and Long-context price
    columns (a 2026-07 page redesign); only Short-context is captured since
    the page doesn't state the token threshold for Long-context pricing.
    Parsing must also stop at the Batch table rather than reading into it."""
    models = parse_openai(_OPENAI_STANDARD_TABLE_TEXT)
    by_id = {m.id: m for m in models}

    assert set(by_id) == {"gpt-5.6-sol", "gpt-5.5", "gpt-4o"}

    sol = by_id["gpt-5.6-sol"]
    assert (sol.pricing.standard_input, sol.pricing.cached_input, sol.pricing.output) == (5.0, 0.5, 30.0)

    assert by_id["gpt-5.5"].context_window == 272_000
    assert by_id["gpt-4o"].pricing.cached_input == 1.25


# ---------------------------------------------------------------------------
# Resilience to provider page changes
# ---------------------------------------------------------------------------


@pytest.fixture()
def restore_providers():
    """Lets a test register a throwaway provider without leaking it."""
    original = dict(PROVIDERS)
    yield
    PROVIDERS.clear()
    PROVIDERS.update(original)


def test_parsers_raise_rather_than_returning_nothing():
    """An empty result would silently erase a provider from the registry, so
    every parser must raise when a page stops matching."""
    redesigned_page = "# Pricing\n\nSee our interactive pricing calculator.\n"

    for parse in (scraper.parse_openai, scraper.parse_gemini, parse_anthropic):
        with pytest.raises(ParseError):
            parse(redesigned_page)


_ANTHROPIC_REORDERED = """\
Some preamble.

| Model | Output | Base Input | Cache Hits & Refreshes | Batch Input | Batch Output |
| --- | --- | --- | --- | --- | --- |
| Claude Opus 9 | $75 / MTok | $15 / MTok | $1.50 / MTok | $7.50 / MTok | $37.50 / MTok |
| Claude Haiku 9 ([deprecated](x)) | $4 / MTok | $0.80 / MTok | $0.08 / MTok | - | - |
"""


def test_anthropic_columns_are_matched_by_header_not_position():
    """Anthropic reordering columns or adding a batch column must not shift
    every price by one."""
    models = {m.id: m for m in parse_anthropic(_ANTHROPIC_REORDERED)}

    opus = models["claude-opus-9"]
    assert (opus.pricing.standard_input, opus.pricing.output) == (15.0, 75.0)
    assert opus.pricing.cached_input == 1.5
    assert (opus.pricing.batch_input, opus.pricing.batch_output) == (7.5, 37.5)
    assert models["claude-haiku-9"].status == "deprecated"


def test_gemini_tier_threshold_is_read_from_the_page():
    """Google moving the long-context boundary off 200k must change the
    threshold, not silently mis-price long prompts at the old one."""
    page = """\
## Gemini Example Pro

*`gemini-example-pro`*

### Standard

|   | Free Tier | Paid Tier, per 1M tokens in USD |
|---|---|---|
| Input price | Not available | $1.25, prompts \\<= 500k tokens $2.50, prompts \\> 500k tokens |
| Output price (including thinking tokens) | Not available | $10.00, prompts \\<= 500k tokens $15.00, prompts \\> 500k tokens |
"""
    model = scraper.parse_gemini(page)[0]

    assert model.tiered_pricing.threshold_tokens == 500_000
    assert model.calculate_cost(input_tokens=600_000, output_tokens=0) == 1.5
    assert model.is_free is False


def test_registering_a_new_provider_takes_one_decorator(restore_providers):
    @register_provider("acme", Vendor("acme"), "https://acme.test/pricing.md")
    def parse_acme(text):
        return [AIModel(id="acme-1", name="Acme 1", vendor=Vendor("acme"),
                        pricing=TokenPricing(standard_input=1.0, output=2.0))]

    assert PROVIDERS["acme"].vendor == "acme"
    assert PROVIDERS["acme"].parse("") [0].id == "acme-1"


def test_a_broken_parser_keeps_the_previous_models(monkeypatch):
    """A page redesign must carry the existing entries forward and report the
    failure -- never hand back an empty provider."""
    previous = [AIModel(id="old-1", vendor=Vendor.OPENAI), AIModel(id="old-2", vendor=Vendor.OPENAI)]
    provider = Provider("broken", Vendor.OPENAI, "https://x.test", lambda text: (_ for _ in ()).throw(ParseError("boom")))
    monkeypatch.setattr(scraper, "fetch", lambda url, timeout=15.0: "whatever")

    models, error = scrape_provider(provider, previous)

    assert models == previous
    assert "ParseError" in error


def test_a_collapsed_model_count_is_treated_as_a_failure(monkeypatch):
    """Providers retire models gradually; losing most of them in one run means
    the parser broke, not that the lineup shrank."""
    previous = [AIModel(id=f"old-{i}", vendor=Vendor.OPENAI) for i in range(10)]
    provider = Provider("shrunk", Vendor.OPENAI, "https://x.test", lambda text: [AIModel(id="only-one", vendor=Vendor.OPENAI)])
    monkeypatch.setattr(scraper, "fetch", lambda url, timeout=15.0: "whatever")

    models, error = scrape_provider(provider, previous)

    assert models == previous
    assert "down from 10" in error


def test_main_refuses_to_overwrite_the_registry_when_a_provider_breaks(monkeypatch, tmp_path, restore_providers):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(scraper.build_registry(
        [AIModel(id="kept", vendor=Vendor.OPENAI, pricing=TokenPricing(standard_input=1.0, output=2.0))]
    )), encoding="utf-8")

    PROVIDERS.clear()
    register_provider("openai", Vendor.OPENAI, "https://x.test")(
        lambda text: (_ for _ in ()).throw(ParseError("page redesigned"))
    )
    monkeypatch.setattr(scraper, "fetch", lambda url, timeout=15.0: "whatever")

    exit_code = scraper.main(["--output", str(registry)])

    assert exit_code == 1
    assert [m["id"] for m in json.loads(registry.read_text())["models"]] == ["kept"]


def test_main_allow_partial_writes_with_carried_over_entries(monkeypatch, tmp_path, restore_providers):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(scraper.build_registry([
        AIModel(id="kept-openai", vendor=Vendor.OPENAI, pricing=TokenPricing(standard_input=1.0, output=2.0)),
    ])), encoding="utf-8")

    PROVIDERS.clear()
    register_provider("openai", Vendor.OPENAI, "https://x.test")(
        lambda text: (_ for _ in ()).throw(ParseError("page redesigned"))
    )
    register_provider("acme", Vendor("acme"), "https://acme.test")(
        lambda text: [AIModel(id="acme-1", vendor=Vendor("acme"), pricing=TokenPricing(standard_input=1.0, output=2.0))]
    )
    monkeypatch.setattr(scraper, "fetch", lambda url, timeout=15.0: "whatever")

    exit_code = scraper.main(["--output", str(registry), "--allow-partial"])
    written = json.loads(registry.read_text())

    assert exit_code == 1  # still reported as a failure
    assert sorted(m["id"] for m in written["models"]) == ["acme-1", "kept-openai"]
    assert written["schema_version"] == scraper.SCHEMA_VERSION


def test_check_mode_never_writes(monkeypatch, tmp_path, restore_providers):
    registry = tmp_path / "registry.json"
    PROVIDERS.clear()
    register_provider("acme", Vendor("acme"), "https://acme.test")(
        lambda text: [AIModel(id="acme-1", vendor=Vendor("acme"), pricing=TokenPricing(standard_input=1.0, output=2.0))]
    )
    monkeypatch.setattr(scraper, "fetch", lambda url, timeout=15.0: "whatever")

    assert scraper.main(["--check", "--output", str(registry)]) == 0
    assert not registry.exists()


def test_only_flag_leaves_other_vendors_untouched(monkeypatch, tmp_path, restore_providers):
    registry = tmp_path / "registry.json"
    registry.write_text(json.dumps(scraper.build_registry([
        AIModel(id="claude-x", vendor=Vendor.ANTHROPIC, pricing=TokenPricing(standard_input=1.0, output=2.0)),
        AIModel(id="gpt-old", vendor=Vendor.OPENAI, pricing=TokenPricing(standard_input=1.0, output=2.0)),
    ])), encoding="utf-8")

    PROVIDERS.clear()
    register_provider("openai", Vendor.OPENAI, "https://x.test")(
        lambda text: [AIModel(id="gpt-new", vendor=Vendor.OPENAI, pricing=TokenPricing(standard_input=1.0, output=2.0))]
    )
    register_provider("anthropic", Vendor.ANTHROPIC, "https://y.test")(
        lambda text: (_ for _ in ()).throw(AssertionError("should not run"))
    )
    monkeypatch.setattr(scraper, "fetch", lambda url, timeout=15.0: "whatever")

    assert scraper.main(["--only", "openai", "--output", str(registry)]) == 0
    assert sorted(m["id"] for m in json.loads(registry.read_text())["models"]) == ["claude-x", "gpt-new"]


# ---------------------------------------------------------------------------
# Format changes the providers have actually made
# ---------------------------------------------------------------------------

_GEMINI_MARKDOC = """\
## Gemini 9 Flash  {% id="gemini-9-flash" %}

*[`gemini-9-flash`](/docs/models/gemini-9-flash)*

{% tabs %}
  {% tab title="Standard" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Free of charge
  * $0.30 (text / image / video)
  $1.00 (audio)

  ---

  * Output price (including thinking tokens)
  * Free of charge
  * $2.50

  ---

  * Context caching price
  * Not available
  * $0.03 (text / image / video)
  {% /table %}
  {% /tab %}
  {% tab title="Batch" %}
  {% table %}
  * Feature
  * Free Tier
  * Paid Tier, per 1M tokens in USD

  ---

  * Input price
  * Not available
  * $0.15
  {% /table %}
  {% /tab %}
{% /tabs %}
"""


def test_gemini_markdoc_tables_are_parsed():
    """Google moved its pricing off markdown tables onto Markdoc
    ``{% table %}`` blocks; both shapes must parse."""
    model = scraper.parse_gemini(_GEMINI_MARKDOC)[0]

    assert model.id == "gemini-9-flash"
    assert (model.pricing.standard_input, model.pricing.output) == (0.3, 2.5)
    assert model.pricing.cached_input == 0.03
    assert model.is_free is True


def test_gemini_batch_tab_is_read_separately_from_standard():
    """Batch pricing lives in its own tab. It must land in batch_input --
    never be mistaken for the standard rate."""
    model = scraper.parse_gemini(_GEMINI_MARKDOC)[0]

    assert model.pricing.standard_input == 0.3
    assert model.pricing.batch_input == 0.15


def test_gemini_still_parses_the_legacy_markdown_layout():
    """The previous '*`id`*' + '### Standard' markdown layout keeps working,
    so a provider reverting a redesign doesn't break the parser either."""
    models = scraper.parse_gemini(_TWO_MODEL_TEXT)

    assert [m.id for m in models] == ["gemini-example-lite"]


_OPENAI_SPLIT_TABLES = """\
### Standard pricing data

| Model | Short context input | Short context output | Short context cached input | Short context 30m write |
| --- | --- | --- | --- | --- |
| gpt-9-astra | $10.00 | $50.00 | $1.00 | $12.50 |
| gpt-9 (<272K context length) | $5.00 | $30.00 | $0.50 | - |

| Model | Input | Output | Cached input |
| --- | --- | --- | --- |
| gpt-9-mini | $0.75 | $4.50 | $0.075 |

### Batch pricing data

#### Batch

| Model | Short context input | Short context output | Short context cached input |
| --- | --- | --- | --- |
| gpt-9-astra | $5.00 | $25.00 | $0.50 |

| Model | Input | Output | Cached input |
| --- | --- | --- | --- |
| gpt-9-mini | $0.375 | $2.25 | $0.0375 |

#### Flex

| Model | Input | Output | Cached input |
| --- | --- | --- | --- |
| gpt-9-mini | $99.00 | $99.00 | $99.00 |
"""


def test_openai_reads_every_standard_table_and_merges_batch():
    """OpenAI splits its models across two tables with different column
    layouts, and reordered output ahead of cached input. Both tables must be
    read, by header name, and batch rates merged from the Batch tier only."""
    models = {m.id: m for m in scraper.parse_openai(_OPENAI_SPLIT_TABLES)}

    assert set(models) == {"gpt-9-astra", "gpt-9", "gpt-9-mini"}

    astra = models["gpt-9-astra"]
    assert (astra.pricing.standard_input, astra.pricing.output) == (10.0, 50.0)
    assert astra.pricing.cached_input == 1.0
    assert (astra.pricing.batch_input, astra.pricing.batch_output) == (5.0, 25.0)

    mini = models["gpt-9-mini"]
    assert (mini.pricing.standard_input, mini.pricing.output) == (0.75, 4.5)
    assert mini.pricing.batch_input == 0.375  # the Batch tier, not Flex's $99

    assert models["gpt-9"].context_window == 272_000


_ANTHROPIC_CONTEXT_TIERED = """\
| Model | Base input tokens | Cache hits and refreshes | Output tokens |
| --- | --- | --- | --- |
| Claude Haiku 9 (for prompts up to 100,000 tokens) | $0.10 / MTok | $0.01 / MTok | $0.50 / MTok |
| Claude Haiku 9 (for prompts over 100,000 tokens) | $0.50 / MTok | $0.05 / MTok | $2.50 / MTok |
"""


def test_anthropic_context_tiered_rows_merge_instead_of_colliding():
    """Anthropic now splits a context-tiered model across two rows with the
    same name. Both must survive as one model with tiered pricing -- not as
    two entries sharing an id, where one silently shadows the other."""
    models = scraper.parse_anthropic(_ANTHROPIC_CONTEXT_TIERED)

    assert len(models) == 1
    haiku = models[0]
    assert haiku.id == "claude-haiku-9"
    assert haiku.pricing.standard_input == 0.10
    assert haiku.tiered_pricing.threshold_tokens == 100_000
    assert haiku.tiered_pricing.over_threshold_rate.standard_input == 0.50
    assert haiku.calculate_cost(input_tokens=50_000, output_tokens=0) == 0.005
    assert haiku.calculate_cost(input_tokens=150_000, output_tokens=0) == 0.075


def test_anthropic_batch_and_cache_columns_never_shadow_the_base_rates():
    """If Anthropic adds batch columns ahead of the base ones, the base input
    and output rates must still be the ones that get picked up."""
    page = """\
| Model | Batch input tokens | Batch output tokens | Base input tokens | 5m cache writes | Cache hits and refreshes | Output tokens |
| --- | --- | --- | --- | --- | --- | --- |
| Claude Opus 9 | $7.50 / MTok | $37.50 / MTok | $15 / MTok | $18.75 / MTok | $1.50 / MTok | $75 / MTok |
"""
    model = scraper.parse_anthropic(page)[0]

    assert (model.pricing.standard_input, model.pricing.output) == (15.0, 75.0)
    assert (model.pricing.batch_input, model.pricing.batch_output) == (7.5, 37.5)
    assert model.pricing.cached_input == 1.5
