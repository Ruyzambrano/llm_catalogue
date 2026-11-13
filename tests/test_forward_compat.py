"""Tests for the one thing this package can't fix with a release: data from
the future.

``Catalog.refresh()`` feeds a registry published *today* into whatever
version of llm-catalogue the user installed *whenever*. Every test here
describes a provider change that hasn't happened yet and asserts that an
unchanged, already-installed copy of this code survives it.
"""
import json

import pytest

from llm_catalogue import catalogue
from llm_catalogue.catalogue import Catalog, IncompatibleRegistryError, parse_registry
from llm_catalogue.models import AIModel, ModelStatus, TokenPricing, Vendor


@pytest.fixture(autouse=True)
def isolated_cache(monkeypatch, tmp_path):
    monkeypatch.setattr(catalogue, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(catalogue, "CACHE_FILE", tmp_path / "registry.json")


def write_registry(path, models, **top_level):
    document = {"updated_at": "2099-01-01", "models": models}
    document.update(top_level)
    path.write_text(json.dumps(document), encoding="utf-8")
    return path


# -- new providers ---------------------------------------------------------


def test_unknown_vendor_loads_instead_of_raising():
    """A provider that didn't exist at release time must not break the load."""
    model = AIModel.from_dict({"id": "mistral-large", "vendor": "mistral", "pricing": {"standard_input": 2.0, "output": 6.0}})

    assert model.vendor == "mistral"
    assert model.vendor.value == "mistral"
    assert model.vendor.is_known is False
    assert Vendor.OPENAI.is_known is True


def test_unknown_vendor_is_queryable_by_name(tmp_path):
    """You can ask for a provider this release has never heard of, as long as
    the loaded registry contains it."""
    path = write_registry(tmp_path / "r.json", [
        {"id": "grok-9", "vendor": "xai", "pricing": {"standard_input": 3.0, "output": 15.0}},
        {"id": "gpt-9", "vendor": "openai", "pricing": {"standard_input": 1.0, "output": 2.0}},
    ])
    catalog = Catalog(registry_path=path)

    assert [m.id for m in catalog.get_models("xai")] == ["grok-9"]
    assert catalog.vendors() == [Vendor.OPENAI, Vendor("xai")]
    with pytest.raises(ValueError):
        catalog.get_models("not-a-real-provider")


def test_unknown_status_is_preserved():
    model = AIModel.from_dict({"id": "x", "status": "sunset", "pricing": {}})

    assert model.status == "sunset"
    assert model.status.is_known is False
    assert model.to_dict()["status"] == "sunset"
    assert ModelStatus("active") is ModelStatus.ACTIVE


# -- new fields ------------------------------------------------------------


def test_unknown_fields_survive_a_round_trip():
    """A pricing dimension or metadata field added to the registry after this
    release must be readable and must not be dropped when re-serialized."""
    raw = {
        "id": "future-model",
        "vendor": "openai",
        "pricing": {"standard_input": 1.0, "output": 2.0, "audio_input": 40.0},
        "knowledge_cutoff": "2031-02",
        "modalities": ["text", "video"],
    }

    model = AIModel.from_dict(raw)

    assert model.extra["modalities"] == ["text", "video"]
    assert model.pricing.extra["audio_input"] == 40.0
    assert AIModel.from_dict(json.loads(json.dumps(model.to_dict()))) == model


def test_missing_and_malformed_values_degrade_instead_of_raising():
    model = AIModel.from_dict({"id": "announced-but-unpriced", "vendor": "google"})

    assert model.name == "announced-but-unpriced"
    assert model.is_priced is False
    assert model.pricing.standard_input is None
    with pytest.raises(ValueError, match="No published pricing"):
        model.calculate_cost(1000, 1000)


def test_prices_given_as_strings_are_coerced():
    pricing = TokenPricing.from_dict({"standard_input": "$2.50", "output": "1,250"})

    assert pricing.standard_input == 2.5
    assert pricing.output == 1250.0


# -- malformed data --------------------------------------------------------


def test_one_bad_entry_does_not_sink_the_registry(tmp_path):
    path = write_registry(tmp_path / "r.json", [
        {"id": "good", "vendor": "openai", "pricing": {"standard_input": 1.0, "output": 2.0}},
        {"no_id": True},
        "not even an object",
    ])
    catalog = Catalog(registry_path=path)

    assert [m.id for m in catalog.all_models()] == ["good"]
    assert len(catalog.load_errors) == 2


def test_corrupt_cache_falls_back_to_bundled_data(monkeypatch, tmp_path):
    """A half-written cache file (crashed process, full disk) must not stop
    the package from starting."""
    cache = tmp_path / "registry.json"
    cache.write_text('{"models": [{"id": "trunca', encoding="utf-8")
    monkeypatch.setattr(catalogue, "CACHE_FILE", cache)

    catalog = Catalog()

    assert catalog.all_models()


def test_newer_bundled_registry_wins_over_stale_cache(monkeypatch, tmp_path):
    """After pip install -U, a months-old cache must not shadow the fresher
    registry that shipped with the upgrade."""
    cache = tmp_path / "registry.json"
    write_registry(cache, [{"id": "stale", "vendor": "openai", "pricing": {}}], updated_at="2000-01-01")
    monkeypatch.setattr(catalogue, "CACHE_FILE", cache)

    catalog = Catalog()

    assert "stale" not in [m.id for m in catalog.all_models()]
    assert catalog.updated_at != "2000-01-01"


def test_newer_cache_wins_over_bundled(monkeypatch, tmp_path):
    cache = tmp_path / "registry.json"
    write_registry(cache, [{"id": "fresh", "vendor": "openai", "pricing": {}}], updated_at="2099-01-01")
    monkeypatch.setattr(catalogue, "CACHE_FILE", cache)

    assert [m.id for m in Catalog().all_models()] == ["fresh"]


# -- schema versioning -----------------------------------------------------


def test_future_schema_version_is_refused_over_the_network(monkeypatch):
    """The escape hatch for a genuinely breaking format change: old clients
    keep the data they have rather than half-reading the new shape."""
    raw = json.dumps({
        "schema_version": 99,
        "updated_at": "2099-01-01",
        "models": [{"id": "x", "vendor": "openai", "pricing": {}}],
    }).encode("utf-8")

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def read(self): return raw

    monkeypatch.setattr(catalogue.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    catalog = Catalog()
    before = catalog.all_models()

    assert catalog.refresh(force=True) is False
    assert catalog.all_models() == before

    with pytest.raises(IncompatibleRegistryError):
        parse_registry(json.loads(raw), strict=True)


def test_future_schema_version_is_read_best_effort_from_disk(tmp_path):
    """Locally, refusing would leave the caller with nothing -- so warn and read."""
    path = write_registry(
        tmp_path / "r.json",
        [{"id": "x", "vendor": "openai", "pricing": {"standard_input": 1.0, "output": 2.0}}],
        schema_version=99,
    )

    with pytest.warns(RuntimeWarning, match="schema_version 99"):
        catalog = Catalog(registry_path=path)

    assert [m.id for m in catalog.all_models()] == ["x"]


# -- bad upstream deploys --------------------------------------------------


def test_refresh_rejects_an_empty_registry_and_does_not_cache_it(monkeypatch):
    """If the upstream registry is ever published empty or broken, one bad
    fetch must not become a poisoned cache on every later start."""
    raw = json.dumps({"updated_at": "2099-01-01", "models": []}).encode("utf-8")

    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def read(self): return raw

    monkeypatch.setattr(catalogue.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    catalog = Catalog()
    before = catalog.all_models()

    assert catalog.refresh(force=True) is False
    assert catalog.all_models() == before
    assert not catalogue.CACHE_FILE.exists()


def test_refresh_survives_garbage_html(monkeypatch):
    class FakeResponse:
        def __enter__(self): return self
        def __exit__(self, *exc): return False
        def read(self): return b"<html>503 Service Unavailable</html>"

    monkeypatch.setattr(catalogue.urllib.request, "urlopen", lambda *a, **k: FakeResponse())
    catalog = Catalog()

    assert catalog.refresh(force=True) is False
    assert catalog.all_models()


def test_custom_registry_url_uses_its_own_cache_file(tmp_path, monkeypatch):
    """A mirror must not overwrite the default registry's cache."""
    monkeypatch.setattr(catalogue, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(catalogue, "CACHE_FILE", tmp_path / "registry.json")
    mirrored = Catalog(registry_url="https://example.internal/registry.json")

    assert mirrored._cache_file() != catalogue.CACHE_FILE
    assert mirrored._cache_file().parent == tmp_path


# -- model id lookup -------------------------------------------------------


def test_get_model_tolerates_gateway_style_prefixes():
    catalog = Catalog()
    direct = catalog.get_model("gemini-2.5-flash")

    assert catalog.get_model("google/gemini-2.5-flash") == direct
    assert catalog.get_model("Gemini-2.5-Flash") == direct
    assert catalog.get_model(None) is None
