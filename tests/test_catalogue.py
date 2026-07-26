import json
import urllib.error

import pytest

from llm_catalogue import catalogue
from llm_catalogue.catalogue import Catalog
from llm_catalogue.models import Vendor


@pytest.fixture()
def catalog():
    return Catalog()


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    """Points the module-level cache paths at a scratch dir for every test in
    this file, so refresh() tests never read or write the real
    ~/.cache/llm_catalogue on the machine running the suite."""
    monkeypatch.setattr(catalogue, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(catalogue, "CACHE_FILE", tmp_path / "registry.json")


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


def test_refresh_returns_false_and_keeps_data_on_network_failure(catalog, monkeypatch):
    def raise_network_error(*args, **kwargs):
        raise urllib.error.URLError("simulated network failure")

    monkeypatch.setattr(catalogue.urllib.request, "urlopen", raise_network_error)
    existing_models = catalog.all_models()

    result = catalog.refresh(timeout=1.0, force=True)

    assert result is False
    assert catalog.all_models() == existing_models


def test_refresh_updates_data_on_success(catalog, monkeypatch):
    fake_registry = {
        "updated_at": "2099-01-01",
        "models": [
            {
                "id": "fake-model",
                "name": "Fake Model",
                "vendor": "openai",
                "pricing": {"standard_input": 1.0, "output": 2.0},
                "free_tier": {"has_free_tier": False},
            }
        ],
    }
    raw = json.dumps(fake_registry).encode("utf-8")

    class FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *exc_info):
            return False

        def read(self):
            return raw

    monkeypatch.setattr(catalogue.urllib.request, "urlopen", lambda *a, **k: FakeResponse())

    result = catalog.refresh(force=True)

    assert result is True
    assert catalog.updated_at == "2099-01-01"
    assert [m.id for m in catalog.all_models()] == ["fake-model"]
    assert catalogue.CACHE_FILE.exists()


def test_refresh_skips_fetch_when_cache_is_fresh(catalog, monkeypatch):
    catalogue.CACHE_FILE.write_text(json.dumps({"updated_at": "cached", "models": []}), encoding="utf-8")

    def fail_if_called(*args, **kwargs):
        raise AssertionError("urlopen should not be called when the cache is still fresh")

    monkeypatch.setattr(catalogue.urllib.request, "urlopen", fail_if_called)

    result = catalog.refresh(force=False)

    assert result is True
    assert catalog.updated_at == "cached"
