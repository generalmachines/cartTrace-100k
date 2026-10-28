"""Layer 1: the one typed settings loader, roles to endpoints, secrets by name, and the one HTTP client factory.

- `settings()`: the process's settings (config.json, then config.local.json, then CARTTRACE__* overrides), cached
- `load_settings(home, env)`: the same, explicitly, for tests and tools
- `Settings.role(name)`: a role's route to a model server, as a `ModelEndpoint`
- `secret(name)`: a secret's value from the environment or ops.secrets_files; never logged or printed
- `carttrace.config.http`: `async_client`, `sync_client` and `RetryPolicy` (imported on use, so this package stays light)
- `carttrace config show | check | schema`

See carttrace/config/loader.py for the order overrides apply in, and carttrace/config/schema.py for every setting.
"""

from carttrace.config.loader import Sources, deep_merge, find_home, load_settings, load_with_sources, settings
from carttrace.config.schema import ConfigFile, Endpoint, EndpointKind, ModelEndpoint, Settings, json_schema_text
from carttrace.config.secrets import SecretStatus, secret, secret_status

__all__ = [
    "ConfigFile",
    "Endpoint",
    "EndpointKind",
    "ModelEndpoint",
    "SecretStatus",
    "Settings",
    "Sources",
    "deep_merge",
    "find_home",
    "json_schema_text",
    "load_settings",
    "load_with_sources",
    "secret",
    "secret_status",
    "settings",
]
