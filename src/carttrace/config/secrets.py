"""Secrets, looked up by name: the environment first, then each file in ops.secrets_files.

Why by name only. Config files are committed and printed, so they hold the name of a variable (ANTHROPIC_API_KEY,
SLACK_WEBHOOK_URL), never its value. The value is read here at the moment of use and handed straight to the caller.
Nothing in this module logs, prints or caches a value, no error message quotes one, and `SecretStatus` (what
`carttrace config show` and `carttrace config check` report) carries where a secret was found, never what it is.

A secrets file holds KEY=VALUE lines: blank lines and lines starting with # are skipped, an `export ` prefix is
allowed, a value may be wrapped in single or double quotes, and an unquoted value ends at " #". Files are read in
the configured order; a file that does not exist is skipped, because the list names candidates.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from carttrace.config.schema import ENV_NAME, Settings
from carttrace.core.errors import ConfigError

ENV_SOURCE = "env"

_NAME = re.compile(ENV_NAME)
_LINE = re.compile(r"^\s*(?:export\s+)?([A-Za-z_][A-Za-z0-9_]*)\s*=\s*(.*?)\s*$")


@dataclass(frozen=True, slots=True)
class SecretStatus:
    """Whether a named secret is available and where it was found. Never the value."""

    name: str
    source: str | None
    """Where it was found: env, or the path of the secrets file that holds it. None when it is set nowhere."""

    @property
    def is_set(self) -> bool:
        return self.source is not None

    def describe(self) -> str:
        """One of: set (env), set (~/.config/mc/secrets.env), unset."""
        return f"set ({self.source})" if self.source else "unset"


def secret(name: str, *, settings: Settings | None = None, env: Mapping[str, str] | None = None) -> str | None:
    """The secret's value, or None when it is set nowhere. Hand it straight to the call that needs it."""
    found = _lookup(name, settings=settings, env=env)
    return found[0] if found else None


def secret_status(name: str, *, settings: Settings | None = None, env: Mapping[str, str] | None = None) -> SecretStatus:
    found = _lookup(name, settings=settings, env=env)
    return SecretStatus(name=name, source=found[1] if found else None)


def secrets_files(settings: Settings | None = None) -> list[Path]:
    """The candidate files, in lookup order, from the given settings or the process settings."""
    if settings is None:
        from carttrace.config.loader import settings as process_settings

        settings = process_settings()
    return list(settings.ops.secrets_files)


def _lookup(name: str, *, settings: Settings | None, env: Mapping[str, str] | None) -> tuple[str, str] | None:
    if not _NAME.match(name):
        raise ConfigError(f"{name!r} is not a variable name", hint="name secrets like ANTHROPIC_API_KEY")
    value = (os.environ if env is None else env).get(name)
    if value:
        return value, ENV_SOURCE
    return _from_files(name, secrets_files(settings))


def _from_files(name: str, files: Sequence[Path]) -> tuple[str, str] | None:
    for path in files:
        value = _read_file(path).get(name)
        if value:
            return value, str(path)
    return None


def _read_file(path: Path) -> dict[str, str]:
    try:
        text = path.read_text(encoding="utf-8")
    except FileNotFoundError:
        return {}
    except OSError as exc:
        raise ConfigError(
            f"cannot read secrets file {path}: {exc.strerror}",
            hint="fix its permissions, or remove it from ops.secrets_files",
        ) from None
    values: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        match = _LINE.match(line)
        if match:
            values[match.group(1)] = _unquote(match.group(2))
    return values


def _unquote(value: str) -> str:
    if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
        return value[1:-1]
    comment = value.find(" #")
    return value[:comment].rstrip() if comment >= 0 else value
