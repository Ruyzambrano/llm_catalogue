from llm_catalogue.scraper import parse_gemini, parse_openai

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
