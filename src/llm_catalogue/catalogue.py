"""The Catalog class -- llm_catalogue's main entry point.

Typical usage::

    from llm_catalogue import Catalog

    catalog = Catalog()
    catalog.get_free_models("google")
"""
from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional, Union

from .models import AIModel, Vendor

BUNDLED_REGISTRY_PATH = Path(__file__).parent / "data" / "registry.json"
CACHE_DIR = Path.home() / ".cache" / "llm_catalogue"
CACHE_FILE = CACHE_DIR / "registry.json"
REMOTE_REGISTRY_URL = (
    "https://raw.githubusercontent.com/Ruyzambrano/llm_catalogue/main/src/llm_catalogue/data/registry.json"
)

_VENDOR_ALIASES = {
    "google": Vendor.GOOGLE,
    "gemini": Vendor.GOOGLE,
    "google-ai": Vendor.GOOGLE,
    "openai": Vendor.OPENAI,
    "anthropic": Vendor.ANTHROPIC,
    "claude": Vendor.ANTHROPIC,
}


def _normalize_vendor(vendor: Union[str, Vendor]) -> Vendor:
    """Resolves a vendor name or alias (case-insensitive) to a Vendor.

    Accepts "openai", "anthropic"/"claude", and "google"/"gemini"/"google-ai".

    Raises:
        ValueError: If ``vendor`` doesn't match any known provider or alias.
    """
    if isinstance(vendor, Vendor):
        return vendor
    key = vendor.lower().strip()
    try:
        return _VENDOR_ALIASES[key]
    except KeyError:
        raise ValueError(f"Unknown vendor {vendor!r}. Expected one of: {sorted(set(_VENDOR_ALIASES))}") from None


class Catalog:
    """Queries model pricing and free-tier metadata for OpenAI, Anthropic, and Google.

    Reads only the registry bundled with the package (or a previously cached
    one) by default -- constructing a Catalog never makes a network call.
    Call refresh() explicitly to pull the latest registry from GitHub.
    """

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        auto_update: bool = False,
        cache_ttl_seconds: int = 3_600,
    ):
        """Loads model data, without making a network call unless asked to.

        Args:
            registry_path: Load registry data from this path instead of the
                default lookup (a previously cached refresh under
                ``~/.cache/llm_catalogue/``, falling back to the registry
                bundled with the package). Mainly useful for tests or for
                pointing at a registry built by your own scraper run.
            auto_update: If True, calls :meth:`refresh` immediately after
                loading -- so construction *can* make a network call, but
                only if you opt in. Recommended for any long-running app
                that wants to notice new models without restarting.
            cache_ttl_seconds: How long a cached refresh is considered
                fresh before :meth:`refresh` will re-fetch it. Defaults to
                1 hour -- short enough that a long-running process notices
                new pricing within the hour, long enough that frequent
                ``refresh()`` calls (e.g. once per request) stay cheap
                file-stat checks rather than network round-trips.
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self.updated_at: Optional[str] = None
        self._models: List[AIModel] = []

        path = registry_path or (CACHE_FILE if CACHE_FILE.exists() else BUNDLED_REGISTRY_PATH)
        self._load(path)

        if auto_update:
            self.refresh()

    def _load(self, path: Path) -> None:
        """Reads a registry.json-shaped file from disk and replaces the loaded data."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        self.updated_at = data.get("updated_at")
        self._models = [AIModel.from_dict(m) for m in data.get("models", [])]

    def refresh(self, timeout: float = 5.0, force: bool = False) -> bool:
        """Pulls the latest registry.json from GitHub and reloads it.

        Skips the network call if a cached copy is younger than
        ``cache_ttl_seconds``, unless ``force=True``. Never raises: on any
        failure (offline, timeout, bad JSON) the currently loaded data is
        left untouched and ``False`` is returned.

        Args:
            timeout: Socket timeout in seconds for the HTTP request.
            force: If True, fetches even if a fresh-enough cache exists.

        Returns:
            True if the loaded data is now up to date (either freshly
            fetched or served from a cache within TTL), False if the fetch
            failed and the previously loaded data is still in place.
        """
        if not force and CACHE_FILE.exists():
            age = time.time() - CACHE_FILE.stat().st_mtime
            if age < self.cache_ttl_seconds:
                self._load(CACHE_FILE)
                return True

        try:
            request = urllib.request.Request(
                REMOTE_REGISTRY_URL, headers={"User-Agent": "llm-catalogue-python/0.1"}
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            data = json.loads(raw)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError):
            return False

        CACHE_DIR.mkdir(parents=True, exist_ok=True)
        CACHE_FILE.write_text(raw, encoding="utf-8")
        self.updated_at = data.get("updated_at")
        self._models = [AIModel.from_dict(m) for m in data.get("models", [])]
        return True

    def all_models(self) -> List[AIModel]:
        """Returns every model in the currently loaded registry, across all vendors."""
        return list(self._models)

    def get_models(self, vendor: Union[str, Vendor], free_only: bool = False) -> List[AIModel]:
        """Returns all models for a vendor, or [] if the vendor has none.

        Args:
            vendor: A :class:`~llm_catalogue.models.Vendor` or one of its
                string aliases (``"openai"``, ``"anthropic"``/``"claude"``,
                ``"google"``/``"gemini"``).
            free_only: If True, only returns models where
                :attr:`~llm_catalogue.models.AIModel.is_free` is True.

        Raises:
            ValueError: If ``vendor`` isn't a recognized provider or alias.
        """
        vendor = _normalize_vendor(vendor)
        results = [m for m in self._models if m.vendor == vendor]
        if free_only:
            results = [m for m in results if m.is_free]
        return results

    def get_free_models(self, vendor: Union[str, Vendor]) -> List[AIModel]:
        """Returns free-tier-eligible models for a vendor, or [] if none exist.

        Equivalent to ``get_models(vendor, free_only=True)``. Useful for
        driving "show me what's free" UI without special-casing providers
        that have no free tier at all (e.g. OpenAI, Anthropic today).
        """
        return self.get_models(vendor, free_only=True)

    def has_free_tier(self, vendor: Union[str, Vendor]) -> bool:
        """Returns True if the given vendor has at least one free-tier-eligible model.

        Handy for a UI toggle: ``if catalog.has_free_tier("google"): show_free_toggle()``.
        """
        return len(self.get_free_models(vendor)) > 0

    def find_free_models(self) -> List[AIModel]:
        """Returns free-tier-eligible models across every vendor."""
        return [m for m in self._models if m.is_free]

    def get_model(self, model_id: str) -> Optional[AIModel]:
        """Looks up a single model by its provider-native id (e.g. ``"gpt-4o"``).

        Returns:
            The matching :class:`~llm_catalogue.models.AIModel`, or ``None``
            if no model in the loaded registry has that id.
        """
        return next((m for m in self._models if m.id == model_id), None)
