import json

from llm_catalogue.models import AIModel, FreeTierPolicy, ModelStatus, TieredPricing, TokenPricing, Vendor


def make_model(**overrides):
    defaults = dict(
        id="test-model",
        name="Test Model",
        vendor=Vendor.OPENAI,
        pricing=TokenPricing(standard_input=2.0, output=10.0, cached_input=0.5, batch_input=1.0, batch_output=5.0),
    )
    defaults.update(overrides)
    return AIModel(**defaults)


def test_calculate_cost_standard():
    model = make_model()
    cost = model.calculate_cost(input_tokens=1_000_000, output_tokens=1_000_000)
    assert cost == 12.0  # 1M * $2 input + 1M * $10 output


def test_calculate_cost_with_cache():
    model = make_model()
    cost = model.calculate_cost(input_tokens=1_000_000, output_tokens=0, cached_tokens=500_000)
    # 500k fresh @ $2 + 500k cached @ $0.5
    assert cost == 1.25


def test_calculate_cost_batch():
    model = make_model()
    cost = model.calculate_cost(input_tokens=1_000_000, output_tokens=1_000_000, is_batch=True)
    assert cost == 6.0  # 1M * $1 batch input + 1M * $5 batch output


def test_calculate_cost_tiered_over_threshold():
    base = TokenPricing(standard_input=2.0, output=10.0)
    over = TokenPricing(standard_input=4.0, output=18.0)
    model = make_model(pricing=base, tiered_pricing=TieredPricing(
        threshold_tokens=200_000, base_rate=base, over_threshold_rate=over
    ))

    under_threshold = model.calculate_cost(input_tokens=100_000, output_tokens=0)
    over_threshold = model.calculate_cost(input_tokens=300_000, output_tokens=0)

    assert under_threshold == 0.2   # 100k @ $2/M base rate
    assert over_threshold == 1.2    # 300k @ $4/M over-threshold rate


def test_is_free():
    free_model = make_model(free_tier=FreeTierPolicy(has_free_tier=True))
    paid_model = make_model(free_tier=FreeTierPolicy(has_free_tier=False))
    no_policy_model = make_model(free_tier=None)

    assert free_model.is_free is True
    assert paid_model.is_free is False
    assert no_policy_model.is_free is False


def test_round_trip_to_dict_from_dict():
    model = make_model(
        status=ModelStatus.DEPRECATED,
        free_tier=FreeTierPolicy(has_free_tier=True, rate_limit_rpm=15, data_used_for_training=True),
        tiered_pricing=TieredPricing(
            threshold_tokens=200_000,
            base_rate=TokenPricing(standard_input=1.0, output=2.0),
            over_threshold_rate=TokenPricing(standard_input=2.0, output=4.0),
        ),
    )

    restored = AIModel.from_dict(json.loads(json.dumps(model.to_dict())))

    assert restored == model
