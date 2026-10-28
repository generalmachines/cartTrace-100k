"""Load settings: config.json, then config.local.json, then the environment, validated once and frozen.

Why one loader with a fixed order. The lab moved models by rewriting os.environ from scripts, so what a process ran
with depended on who imported what first. Here the order is fixed and visible in `carttrace config show`:

1. config.json at the repository root (checked in);
2. config.local.json beside it (untracked, per machine), deep-merged: objects merge key by key, anything else
   replaces, lists included;
3. the documented shortcuts CARTTRACE_LIVE, CARTTRACE_STORAGE and CARTTRACE_CONTROL_FILE;
4. CARTTRACE__SECTION__KEY=value overrides: double underscores nest, keys match case-insensitively with `.` and `_` treated
   alike (CARTTRACE__ROLES__DECIDE_FLASH sets roles["decide.flash"]), and the value is parsed as JSON when it parses, else
   kept as a string;
5. CARTTRACE_GPU_MAX_INFLIGHT, which lowers every endpoint's cap, and every cap of its schedule, to at most its value. It
   is a brake, never a raise.

The repository root is CARTTRACE_HOME, or the nearest directory at or above the working directory that holds config.json.
Errors name the setting and where to fix it, and never echo a value: a secret pasted into the wrong place is exactly
the value most likely to fail validation.
"""

from __future__ import annotations

import functools
import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pydantic

from carttrace.config.schema import Settings
from carttrace.core.canon import loads
from carttrace.core.errors import ConfigError

CONFIG_FILE = "config.json"
LOCAL_FILE = "config.local.json"
HOME_ENV = "CARTTRACE_HOME"
OVERRIDE_PREFIX = "CARTTRACE__"
GPU_CAP_ENV = "CARTTRACE_GPU_MAX_INFLIGHT"
SHORTCUTS: dict[str, tuple[str, str]] = {
    "CARTTRACE_CONTROL_FILE": ("live", "control_file"),
    "CARTTRACE_LIVE": ("live", "enabled"),
    "CARTTRACE_STORAGE": ("storage", "url"),
}
"""Variables the docs name, mapped to the setting each one sets. An empty value leaves the setting alone, except
CARTTRACE_LIVE, where empty means off."""

_TRUE = frozenset({"1", "true", "yes", "on"})
_FALSE = frozenset({"0", "false", "no", "off", ""})


@dataclass(frozen=True, slots=True)
class Sources:
    """Where the effective settings came from: the files merged and the variables applied, by name only."""

    home: Path
    files: tuple[Path, ...]
    variables: tuple[str, ...]


def find_home(env: Mapping[str, str] | None = None, cwd: Path | None = None) -> Path:
    """The repository root: CARTTRACE_HOME, else the nearest directory at or above `cwd` that holds config.json."""
    env = os.environ if env is None else env
    explicit = env.get(HOME_ENV, "").strip()
    if explicit:
        home = Path(explicit).expanduser().resolve()
        if not (home / CONFIG_FILE).is_file():
            raise ConfigError(
                f"{HOME_ENV}={home} holds no {CONFIG_FILE}", hint=f"point {HOME_ENV} at the repository root"
            )
        return home
    start = (cwd or Path.cwd()).resolve()
    for directory in (start, *start.parents):
        if (directory / CONFIG_FILE).is_file():
            return directory
    raise ConfigError(
        f"no {CONFIG_FILE} in {start} or any directory above it",
        hint=f"run inside the repository or set {HOME_ENV} to its root",
    )


def load_settings(home: Path, env: Mapping[str, str]) -> Settings:
    """Settings for the repository at `home` under the environment `env`. Tests pass both explicitly."""
    return load_with_sources(home, env)[0]


def load_with_sources(home: Path, env: Mapping[str, str]) -> tuple[Settings, Sources]:
    """`load_settings`, plus the files and variable names that shaped the result."""
    home = home.expanduser().resolve()
    files = [home / CONFIG_FILE]
    raw = _read_object(files[0])
    local = home / LOCAL_FILE
    if local.is_file():
        raw = deep_merge(raw, _read_object(local))
        files.append(local)
    if "home" in raw:
        raise ConfigError("config files must not set 'home'", hint=f"remove it; set {HOME_ENV} to move the root")
    variables = _apply_shortcuts(raw, env)
    variables += _apply_overrides(raw, env)
    settings = _validate(raw, home, files, variables)
    cap = _gpu_cap(env)
    if cap is not None:
        settings = _clamp(settings, cap)
        variables.append(GPU_CAP_ENV)
    return settings, Sources(home=home, files=tuple(files), variables=tuple(variables))


@functools.cache
def settings() -> Settings:
    """The process's settings, loaded on first use from the working directory and os.environ, then cached.

    Call `settings.cache_clear()` to reload, for example after a test changes the environment.
    """
    env = dict(os.environ)
    return load_settings(find_home(env), env)


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict[str, Any]:
    """`over` on top of `base`: objects merge key by key, any other value (lists included) replaces. Inputs are not
    modified."""
    merged = dict(base)
    for key, value in over.items():
        current = merged.get(key)
        if isinstance(value, Mapping) and isinstance(current, Mapping):
            merged[key] = deep_merge(current, value)
        else:
            merged[key] = value
    return merged


def _read_object(path: Path) -> dict[str, Any]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise ConfigError(f"no {path}", hint="run inside the repository or set CARTTRACE_HOME") from None
    except OSError as exc:
        raise ConfigError(f"cannot read {path}: {exc.strerror}", hint="check the file's permissions") from exc
    try:
        value = loads(text)
    except json.JSONDecodeError as exc:
        raise ConfigError(
            f"{path} is not valid JSON: {exc.msg} at line {exc.lineno} column {exc.colno}",
            hint="fix the file; JSON allows no comments and no trailing commas",
        ) from exc
    except ValueError as exc:
        raise ConfigError(
            f"{path} uses NaN, Infinity or a number too large for a float", hint="write a finite number"
        ) from exc
    if not isinstance(value, dict):
        raise ConfigError(f"{path} must hold one JSON object", hint="wrap the settings in { ... }")
    return value


def _apply_shortcuts(raw: dict[str, Any], env: Mapping[str, str]) -> list[str]:
    applied: list[str] = []
    for variable, (section, key) in sorted(SHORTCUTS.items()):
        if variable not in env:
            continue
        value: object = env[variable]
        if variable == "CARTTRACE_LIVE":
            value = _flag(variable, env[variable])
        elif not env[variable].strip():
            continue
        _object_at(raw, section, variable)[key] = value
        applied.append(variable)
    return applied


def _flag(variable: str, value: str) -> bool:
    lowered = value.strip().lower()
    if lowered in _TRUE:
        return True
    if lowered in _FALSE:
        return False
    raise ConfigError(f"{variable} must be 1 or 0", hint=f"unset {variable}, or set it to 1 to enable or 0 to disable")


def _object_at(raw: dict[str, Any], section: str, variable: str) -> dict[str, Any]:
    node = raw.setdefault(section, {})
    if not isinstance(node, dict):
        raise ConfigError(f"{variable} sets {section}, which is not an object in the config files")
    return node


def _apply_overrides(raw: dict[str, Any], env: Mapping[str, str]) -> list[str]:
    applied: list[str] = []
    for variable in sorted(name for name in env if name.startswith(OVERRIDE_PREFIX)):
        segments = variable[len(OVERRIDE_PREFIX) :].split("__")
        if not all(segments):
            raise ConfigError(
                f"{variable} has an empty segment",
                hint="write CARTTRACE__SECTION__KEY=value, with double underscores between levels",
            )
        _set_at(raw, segments, _parse_value(env[variable]), variable)
        applied.append(variable)
    return applied


def _set_at(node: dict[str, Any], segments: list[str], value: Any, variable: str) -> None:
    trail: list[str] = []
    for depth, segment in enumerate(segments):
        key = _matching_key(node, segment, variable)
        trail.append(key)
        if depth == len(segments) - 1:
            node[key] = value
            return
        child = node.get(key)
        if child is None:
            child = node[key] = {}
        elif not isinstance(child, dict):
            raise ConfigError(
                f"{variable} sets a key inside {'.'.join(trail)}, which is not an object",
                hint=f"set {'.'.join(trail)} itself, or fix the variable name",
            )
        node = child


def _matching_key(node: Mapping[str, Any], segment: str, variable: str) -> str:
    wanted = _normal(segment)
    hits = sorted(key for key in node if _normal(key) == wanted)
    if len(hits) > 1:
        raise ConfigError(f"{variable} matches several keys: {hits}", hint="set the key in config.local.json instead")
    return hits[0] if hits else segment.lower()


def _normal(key: str) -> str:
    return key.lower().replace(".", "_").replace("-", "_")


def _parse_value(value: str) -> Any:
    try:
        return loads(value)
    except ValueError:  # not JSON (JSONDecodeError is a ValueError), or not finite: keep the text
        return value


def _validate(raw: dict[str, Any], home: Path, files: list[Path], variables: list[str]) -> Settings:
    try:
        return Settings.model_validate({**raw, "home": home}, context={"home": home})
    except pydantic.ValidationError as exc:
        problems = [
            f"{'.'.join(str(p) for p in err['loc']) or '(top level)'}: {err['msg']}"
            for err in exc.errors(include_url=False, include_input=False, include_context=False)
        ]
        where = ", ".join(str(f) for f in files)
        if variables:
            where += " or the variables " + ", ".join(variables)
        # `from None`: pydantic's own message quotes the offending input, which may be a misplaced secret.
        raise ConfigError("config is invalid: " + "; ".join(problems), hint=f"fix {where}") from None


def _gpu_cap(env: Mapping[str, str]) -> int | None:
    raw = env.get(GPU_CAP_ENV, "").strip()
    if not raw:
        return None
    try:
        cap = int(raw)
    except ValueError:
        cap = 0
    if cap < 1:
        raise ConfigError(f"{GPU_CAP_ENV} must be a whole number of calls, at least 1", hint=f"unset {GPU_CAP_ENV}")
    return cap


def _clamp(settings: Settings, cap: int) -> Settings:
    endpoints = {
        name: endpoint.model_copy(
            update={
                "max_inflight": min(endpoint.max_inflight, cap),
                "schedule": [
                    window.model_copy(update={"max_inflight": min(window.max_inflight, cap)})
                    for window in endpoint.schedule
                ],
            }
        )
        for name, endpoint in settings.endpoints.items()
    }
    return settings.model_copy(update={"endpoints": endpoints})
