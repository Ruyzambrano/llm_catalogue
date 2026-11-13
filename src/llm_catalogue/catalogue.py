"""The Catalog class -- llm_catalogue's main entry point.

Typical usage::

    from llm_catalogue import Catalog

    catalog = Catalog()
    catalog.get_free_models("google")

**Forward compatibility.** :meth:`Catalog.refresh` pulls a registry published
later than the installed package, so this module treats registry data as
untrusted-but-well-meant input: a model entry it can't parse is skipped (and
reported on :attr:`Catalog.load_errors`) instead of failing the load, a
corrupt cache file falls back to the bundled copy, a vendor nobody had heard
of at release time is queryable by name, and a registry declaring a
``schema_version`` newer than this package understands is refused *over the
network* (keeping the data you already have) rather than half-read.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ._version import __version__
from .models import AIModel, Vendor, _as_int

#: Highest ``schema_version`` of registry.json this release can read. A
#: registry declaring a higher one is assumed to have changed in a way this
#: code can't honour, and is refused when it arrives over the network.
SCHEMA_VERSION = 1

BUNDLED_REGISTRY_PATH = Path(__file__).parent / "data" / "registry.json"
CACHE_DIR = Path(os.environ.get("LLM_CATALOGUE_CACHE_DIR") or (Path.home() / ".cache" / "llm_catalogue"))
CACHE_FILE = CACHE_DIR / "registry.json"
DEFAULT_REGISTRY_URL = (
    "https://raw.githubusercontent.com/Ruyzambrano/llm_catalogue/main/src/llm_catalogue/data/registry.json"
)
#: Where :meth:`Catalog.refresh` fetches from. Override with the
#: ``LLM_CATALOGUE_REGISTRY_URL`` environment variable (or the ``registry_url``
#: constructor argument) to point at a mirror, an internal registry, or a new
#: location if this project ever moves -- no code change or release needed.
REMOTE_REGISTRY_URL = os.environ.get("LLM_CATALOGUE_REGISTRY_URL") or DEFAULT_REGISTRY_URL

_VENDOR_ALIASES = {
    "google": Vendor.GOOGLE,
    "gemini": Vendor.GOOGLE,
    "google-ai": Vendor.GOOGLE,
    "googleai": Vendor.GOOGLE,
    "vertex": Vendor.GOOGLE,
    "openai": Vendor.OPENAI,
    "open-ai": Vendor.OPENAI,
    "gpt": Vendor.OPENAI,
    "anthropic": Vendor.ANTHROPIC,
    "claude": Vendor.ANTHROPIC,
}


class IncompatibleRegistryError(ValueError):
    """Raised when registry data declares a ``schema_version`` newer than
    :data:`SCHEMA_VERSION`, i.e. it was written by a future release that
    changed the format in a way this one can't be trusted to read."""


@dataclass
class RegistryData:
    """A parsed registry.json document.

    Attributes:
        updated_at: ISO date the registry was rebuilt, if stated.
        schema_version: Format version the document declared (1 if absent).
        models: Every entry that parsed successfully.
        errors: One message per entry that didn't, so a malformed row is
            visible rather than silently missing.
        extra: Unmodelled top-level keys, preserved for forward compatibility.
    """

    updated_at: Optional[str] = None
    schema_version: int = SCHEMA_VERSION
    models: List[AIModel] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    extra: Dict[str, Any] = field(default_factory=dict)


def parse_registry(data: Any, strict: bool = False) -> RegistryData:
    """Turns a decoded registry.json document into a :class:`RegistryData`.

    Args:
        data: The decoded JSON document (``{"models": [...], ...}``).
        strict: If True, a ``schema_version`` newer than :data:`SCHEMA_VERSION`
            raises :class:`IncompatibleRegistryError`. Used for data arriving
            over the network, where refusing it safely leaves the previously
            loaded data in place. Local loads pass False: a best-effort read
            beats leaving the caller with no data at all, so they only warn.

    Returns:
        The parsed document. Entries that fail to parse are collected in
        :attr:`RegistryData.errors` rather than aborting the whole load --
        one malformed model shouldn't cost you the other eighty.

    Raises:
        IncompatibleRegistryError: See ``strict``.
        ValueError: If ``data`` isn't a JSON object at all.
    """
    if not isinstance(data, dict):
        raise ValueError(f"Registry must be a JSON object, got {type(data).__name__}")

    version = _as_int(data.get("schema_version")) or SCHEMA_VERSION
    if version > SCHEMA_VERSION:
        message = (
            f"Registry declares schema_version {version}, but this release of "
            f"llm-catalogue understands up to {SCHEMA_VERSION}. Upgrade with "
            "'pip install -U llm-catalogue'."
        )
        if strict:
            raise IncompatibleRegistryError(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    raw_models = data.get("models")
    if not isinstance(raw_models, list):
        raw_models = []

    models: List[AIModel] = []
    errors: List[str] = []
    for index, entry in enumerate(raw_models):
        try:
            models.append(AIModel.from_dict(entry))
        except Exception as exc:  # noqa: BLE001 -- one bad row must not sink the rest
            errors.append(f"models[{index}]: {exc}")

    extra = {k: v for k, v in data.items() if k not in {"updated_at", "schema_version", "models"}}
    updated_at = data.get("updated_at")
    return RegistryData(
        updated_at=updated_at if isinstance(updated_at, str) else None,
        schema_version=version,
        models=models,
        errors=errors,
        extra=extra,
    )


def _normalize_vendor(vendor: Union[str, Vendor], available: Optional[List[Vendor]] = None) -> Vendor:
    """Resolves a vendor name or alias (case-insensitive) to a Vendor.

    Accepts the canonical names and aliases this release ships with
    ("openai", "anthropic"/"claude", "google"/"gemini"/"vertex", ...) *and*
    any vendor string actually present in the loaded registry -- so a
    provider added to the data after this release is still queryable by
    name, while an outright typo still raises.

    Args:
        vendor: A :class:`~llm_catalogue.models.Vendor` or a name/alias.
        available: Vendors present in the currently loaded registry, if any.

    Raises:
        ValueError: If ``vendor`` matches neither a known alias nor anything
            in the loaded data.
    """
    if isinstance(vendor, Vendor):
        return vendor
    if not isinstance(vendor, str):
        raise ValueError(f"Vendor must be a string or Vendor, got {type(vendor).__name__}")

    key = vendor.lower().strip().replace("_", "-")
    if key in _VENDOR_ALIASES:
        return _VENDOR_ALIASES[key]

    for candidate in available or []:
        if candidate.value == key:
            return candidate

    choices = sorted({*_VENDOR_ALIASES, *(c.value for c in available or [])})
    raise ValueError(f"Unknown vendor {vendor!r}. Expected one of: {choices}")


class Catalog:
    """Queries model pricing and free-tier metadata across AI providers.

    Reads only the registry bundled with the package (or a previously cached
    one) by default -- constructing a Catalog never makes a network call.
    Call refresh() explicitly to pull the latest registry from GitHub.

    The query methods are vendor-agnostic: they filter the loaded data rather
    than switching on a hardcoded list of providers, so a registry that adds
    a provider works immediately. Use :meth:`vendors` to discover what the
    loaded data actually contains instead of assuming the three this release
    was written against.
    """

    def __init__(
        self,
        registry_path: Optional[Path] = None,
        auto_update: bool = False,
        cache_ttl_seconds: int = 3_600,
        registry_url: Optional[str] = None,
    ):
        """Loads model data, without making a network call unless asked to.

        Args:
            registry_path: Load registry data from this path instead of the
                default lookup (the newer of a previously cached refresh
                under ``~/.cache/llm_catalogue/`` and the registry bundled
                with the package). Mainly useful for tests or for pointing
                at a registry built by your own scraper run.
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
            registry_url: Fetch refreshes from here instead of the project's
                GitHub copy -- e.g. an internal mirror you control, or a new
                upstream location. Defaults to the ``LLM_CATALOGUE_REGISTRY_URL``
                environment variable, then the project's own URL. A custom
                URL gets its own cache file, so mirrors never clobber each
                other's data.
        """
        self.cache_ttl_seconds = cache_ttl_seconds
        self.registry_url = registry_url or REMOTE_REGISTRY_URL
        self.updated_at: Optional[str] = None
        self.schema_version: int = SCHEMA_VERSION
        #: Messages for any registry entries that couldn't be parsed by the
        #: most recent load. Empty in the normal case; non-empty means the
        #: data contained something this release couldn't make sense of, and
        #: those models are missing from query results.
        self.load_errors: List[str] = []
        self._models: List[AIModel] = []

        if registry_path is not None:
            self._load_path(registry_path, required=True)
        else:
            self._load_best_available()

        if auto_update:
            self.refresh()

    # -- loading ---------------------------------------------------------

    def _cache_file(self) -> Path:
        """Returns the cache path for this instance's registry URL.

        The default URL uses the shared ``registry.json``; a custom mirror
        gets its own file keyed by a hash of the URL, so two Catalogs
        pointed at different registries don't overwrite each other.
        """
        if self.registry_url == REMOTE_REGISTRY_URL:
            return CACHE_FILE
        digest = hashlib.sha256(self.registry_url.encode("utf-8")).hexdigest()[:12]
        return CACHE_DIR / f"registry-{digest}.json"

    def _read(self, path: Path) -> Optional[RegistryData]:
        """Reads and parses a registry file, returning None if unreadable.

        Never raises on a missing, truncated or corrupt file -- a cache
        that got half-written by a crashed process must not stop the
        package from starting.
        """
        try:
            return parse_registry(json.loads(Path(path).read_text(encoding="utf-8")))
        except (OSError, ValueError):
            return None

    @staticmethod
    def _is_newer(candidate: Optional[RegistryData], current: Optional[RegistryData]) -> bool:
        """Whether ``candidate`` carries a later ``updated_at`` than ``current``.

        Dates are ISO-8601, so a string compare orders them correctly; an
        unparseable or missing date loses, which keeps a junk value from
        shadowing real data.
        """
        if candidate is None:
            return False
        if current is None:
            return True
        return (candidate.updated_at or "") > (current.updated_at or "")

    def _load_best_available(self) -> None:
        """Loads whichever of the cached and bundled registries is newer.

        A cached refresh is usually newer than the bundled copy -- but not
        after ``pip install -U`` ships a fresher one, and not if the cache
        is corrupt. Comparing ``updated_at`` instead of blindly preferring
        the cache means an upgrade is never silently undone by a stale file.
        """
        bundled = self._read(BUNDLED_REGISTRY_PATH)
        cached = self._read(self._cache_file())

        chosen = cached if self._is_newer(cached, bundled) else bundled
        if chosen is None:
            chosen = cached or bundled
        if chosen is None:
            raise FileNotFoundError(
                f"No readable registry found at {BUNDLED_REGISTRY_PATH} or {self._cache_file()}. "
                "The package install looks incomplete; try reinstalling llm-catalogue."
            )
        self._apply(chosen)

    def _load_path(self, path: Path, required: bool = False) -> None:
        """Loads an explicitly requested registry file.

        Raises if ``required`` and the file can't be read -- an explicit
        ``registry_path`` that silently fell back to bundled data would be
        worse than an error.
        """
        parsed = self._read(path)
        if parsed is None:
            if required:
                raise ValueError(f"Could not read a registry from {path}")
            return
        self._apply(parsed)

    def _load(self, path: Path) -> None:
        """Reads a registry.json-shaped file from disk and replaces the loaded data.

        Kept for backwards compatibility; prefer :meth:`_load_path`.
        """
        self._load_path(path, required=True)

    def _apply(self, parsed: RegistryData) -> None:
        """Swaps in a parsed registry as this Catalog's data."""
        self.updated_at = parsed.updated_at
        self.schema_version = parsed.schema_version
        self.load_errors = parsed.errors
        self._models = parsed.models

    def refresh(self, timeout: float = 5.0, force: bool = False) -> bool:
        """Pulls the latest registry.json from GitHub and reloads it.

        Skips the network call if a cached copy is younger than
        ``cache_ttl_seconds``, unless ``force=True``. Never raises: on any
        failure (offline, timeout, bad JSON, a registry whose schema is
        newer than this release understands) the currently loaded data is
        left untouched and ``False`` is returned.

        A response that doesn't parse, or that contains no usable models at
        all, is treated as a failure and is *not* written to the cache --
        so a bad deploy upstream can't poison every subsequent start.

        Args:
            timeout: Socket timeout in seconds for the HTTP request.
            force: If True, fetches even if a fresh-enough cache exists.

        Returns:
            True if the loaded data is now up to date (either freshly
            fetched or served from a cache within TTL), False if the fetch
            failed and the previously loaded data is still in place.
        """
        cache_file = self._cache_file()
        if not force and cache_file.exists():
            age = time.time() - cache_file.stat().st_mtime
            if age < self.cache_ttl_seconds:
                cached = self._read(cache_file)
                if cached is not None:
                    self._apply(cached)
                    return True

        try:
            request = urllib.request.Request(
                self.registry_url,
                headers={"User-Agent": f"llm-catalogue-python/{__version__}"},
            )
            with urllib.request.urlopen(request, timeout=timeout) as response:
                raw = response.read().decode("utf-8")
            parsed = parse_registry(json.loads(raw), strict=True)
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError, OSError, ValueError, UnicodeDecodeError):
            return False

        if not parsed.models:
            return False

        try:
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            cache_file.write_text(raw, encoding="utf-8")
        except OSError:
            pass  # A read-only or full disk shouldn't stop us using the data we just fetched.

        self._apply(parsed)
        return True

    # -- queries ---------------------------------------------------------

    def all_models(self) -> List[AIModel]:
        """Returns every model in the currently loaded registry, across all vendors."""
        return list(self._models)

    def vendors(self) -> List[Vendor]:
        """Returns every vendor present in the loaded registry, in name order.

        Use this instead of hardcoding provider names: it reflects the data
        you actually have, including providers added after this release.
        """
        return sorted({m.vendor for m in self._models}, key=lambda v: v.value)

    def get_models(
        self,
        vendor: Union[str, Vendor],
        free_only: bool = False,
        include_retired: bool = True,
    ) -> List[AIModel]:
        """Returns all models for a vendor, or [] if the vendor has none.

        Args:
            vendor: A :class:`~llm_catalogue.models.Vendor`, one of its
                string aliases (``"openai"``, ``"anthropic"``/``"claude"``,
                ``"google"``/``"gemini"``), or the name of any vendor
                present in the loaded registry.
            free_only: If True, only returns models where
                :attr:`~llm_catalogue.models.AIModel.is_free` is True.
            include_retired: If False, drops models the provider has marked
                deprecated or retired.

        Raises:
            ValueError: If ``vendor`` isn't a recognized provider, alias, or
                vendor present in the loaded data.
        """
        vendor = _normalize_vendor(vendor, self.vendors())
        results = [m for m in self._models if m.vendor == vendor]
        if free_only:
            results = [m for m in results if m.is_free]
        if not include_retired:
            results = [m for m in results if m.status not in {"deprecated", "retired"}]
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

        The match is case-insensitive and ignores a ``vendor/`` or
        ``vendor:`` prefix, so ids copied from gateways and routers
        (``"anthropic/claude-opus-4-1"``) resolve too.

        Returns:
            The matching :class:`~llm_catalogue.models.AIModel`, or ``None``
            if no model in the loaded registry has that id.
        """
        if not isinstance(model_id, str):
            return None
        exact = next((m for m in self._models if m.id == model_id), None)
        if exact is not None:
            return exact

        key = model_id.strip().lower()
        for separator in ("/", ":"):
            if separator in key:
                key = key.rsplit(separator, 1)[-1]
        return next((m for m in self._models if m.id.lower() == key), None)
