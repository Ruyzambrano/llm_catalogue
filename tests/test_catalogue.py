import pytest

from llm_catalogue.catalogue import Catalog
from llm_catalogue.models import Vendor


@pytest.fixture()
def catalog():
    return Catalog()


def test_construction_does_not_require_network(catalog):
    # If this fixture built successfully, Catalog() only read local data.
    assert catalog.all_models()


def test_get_models_for_each_vendor(catalog):
    expected_vendor = {"openai": Vendor.OPENAI, "anthropic": Vendor.ANTHROPIC, "google": Vendor.GOOGLE}
    for vendor, enum_value in expected_vendor.items():
        models = catalog.get_models(vendor)
        assert models, f"expected at least one model for {vendor}"
        assert all(m.vendor == enum_value for m in models)


def test_vendor_aliases_are_equivalent(catalog):
    assert catalog.get_models("google") == catalog.get_models("gemini")
    assert catalog.get_models("anthropic") == catalog.get_models("claude")


def test_unknown_vendor_raises(catalog):
    with pytest.raises(ValueError):
        catalog.get_models("mistral")


def test_openai_has_no_free_tier(catalog):
    assert catalog.get_free_models("openai") == []
    assert catalog.has_free_tier("openai") is False


def test_google_has_free_tier(catalog):
    free_models = catalog.get_free_models("google")
    assert free_models
    assert catalog.has_free_tier("google") is True
    assert all(m.is_free for m in free_models)


def test_find_free_models_spans_vendors(catalog):
    free_models = catalog.find_free_models()
    assert free_models
    assert all(m.is_free for m in free_models)
    assert any(m.vendor == Vendor.GOOGLE for m in free_models)
    assert not any(m.vendor == Vendor.OPENAI for m in free_models)


def test_get_model_by_id(catalog):
    model = catalog.get_model("gemini-2.5-flash")
    assert model is not None
    assert model.vendor == Vendor.GOOGLE
    assert model.is_free is True


def test_get_model_unknown_id_returns_none(catalog):
    assert catalog.get_model("does-not-exist") is None


def test_refresh_never_raises_when_unreachable(catalog):
    # This environment has no route to raw.githubusercontent.com, so this
    # exercises the real failure path rather than a mocked one.
    result = catalog.refresh(timeout=2.0, force=True)
    assert result is False
    assert catalog.all_models()  # existing data must survive a failed refresh
