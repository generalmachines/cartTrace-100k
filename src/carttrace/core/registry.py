"""Components are Lego blocks: one interface, any number of implementations, and config.json choosing one by name.

Why a registry and not if-chains. The project wants every part replaceable without touching its callers: storage (local
or S3), the chat backend, the scorer (a decision model or recorded), the notifier (Slack or a file), the sheet backend
(Google Sheets or CSV), the infographic renderer (Excalidraw or matplotlib), imagery (H3, Qwen-Image or none), the
explainer (Sonnet or a template). Each package declares its implementations as import strings in a module-level
`PLUGINS` table:

    PLUGINS = {"storage.blobstore": {"local": "carttrace.storage.local:LocalBlobStore", "s3": "carttrace.storage.s3:S3BlobStore"}}

A kind is `<package>.<interface>`. `resolve(kind, name)` imports `mc.<package>`, looks the name up in its PLUGINS and
imports the target lazily, so choosing "local" never imports boto3. Adding a block means adding one line to PLUGINS and
one name to config.json; no caller changes.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, cast

from carttrace.core.errors import ConfigError


def _plugins_of(kind: str) -> dict[str, str]:
    package = kind.split(".", 1)[0]
    if not package or "." not in kind:
        raise ConfigError(f"component kind {kind!r} is not '<package>.<interface>'")
    try:
        mod = importlib.import_module(f"carttrace.{package}")
    except ModuleNotFoundError as exc:
        raise ConfigError(f"no package carttrace.{package} for component kind {kind!r}") from exc
    table = getattr(mod, "PLUGINS", {})
    impls = table.get(kind)
    if not isinstance(impls, dict):
        raise ConfigError(
            f"carttrace.{package} declares no implementations of {kind!r}",
            hint=f"add {kind!r} to PLUGINS in mc/{package}/__init__.py",
        )
    return cast(dict[str, str], impls)


def names(kind: str) -> list[str]:
    """Every implementation name declared for a kind, sorted."""
    return sorted(_plugins_of(kind))


def resolve(kind: str, name: str) -> Callable[..., Any]:
    """The factory (a class or a function) registered as `name` for `kind`."""
    impls = _plugins_of(kind)
    target = impls.get(name)
    if target is None:
        raise ConfigError(
            f"unknown {kind} implementation {name!r}", hint=f"choose one of {sorted(impls)} in config.json components"
        )
    module_name, _, attr = target.partition(":")
    try:
        obj = getattr(importlib.import_module(module_name), attr)
    except (ModuleNotFoundError, AttributeError) as exc:
        raise ConfigError(f"{kind} implementation {name!r} points at {target!r}, which does not import: {exc}") from exc
    if not callable(obj):
        raise ConfigError(f"{target} is not callable")
    return cast(Callable[..., Any], obj)
