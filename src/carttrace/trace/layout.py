"""Where a bundle's files live and how its JSON is read back: what the writer, the reader and recovery share.

Why its own module. The writer is a state machine with staging locks; the reader and the CLI only need the names of
the files, the storage key of a bundle, the shape of a store, and a way to read a file as a model that never leaks
input into an error. Keeping those here lets each depend on the layout without depending on the writer.

Storage keys: `episodes/<UTC date of started_at>/<run_id>/<episode_id>/<file>`. The date is the UTC day the episode
started, so a reader that wants one day's episodes lists one folder instead of the whole archive.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Protocol

from pydantic import BaseModel
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.errors import ValidationError
from carttrace.trace.schema import SCHEMA, EpisodeHeader

if TYPE_CHECKING:
    from carttrace.storage.service import BlobStore

EPISODES_PREFIX = "episodes/"
EPISODE_FILE = "episode.json"
TURNS_FILE = "turns.jsonl"
WIRE_FILE = "wire.jsonl"
STATE_FILE = "state.jsonl"
DECISIONS_FILE = "decisions.jsonl"
OUTCOME_FILE = "outcome.json"
MANIFEST_FILE = "bundle.json"
EVENTS_FILE = "events.jsonl"
"""The sandbox's own event log (D33): on a sandbox bundle whose sandbox opened, never on a live one."""
PRUNED_FILE = "pruned.json"
"""A tombstone's record (D79, carttrace.trace.prune): the bundle's logs were removed on purpose, and this says what and why."""
AGENT_FILE = "agent.json"
"""What the agent was given (D69): its system prompt and every tool definition, once per episode; absent before."""
EXPOSURE_FILE = "exposure.json"
"""What the store exposed, per data type and surface (D93, carttrace.trace.exposure): on sandbox bundles since D93."""
EXPERIMENT_FILE = "experiment.json"
"""What kind of trace a sandbox v1 bundle is, for the judge (D96, carttrace.trace.experiment): on v1 rows only."""
LOG_FILES = (TURNS_FILE, WIRE_FILE, STATE_FILE, DECISIONS_FILE)
BUNDLE_FILES = (EPISODE_FILE, *LOG_FILES, OUTCOME_FILE)
"""Every file bundle.json lists on every bundle. bundle.json itself goes after all of them."""
OPTIONAL_FILES = (EVENTS_FILE, AGENT_FILE, EXPOSURE_FILE, EXPERIMENT_FILE)
"""Files bundle.json lists only when the episode wrote them; a bundle without them is whole."""
UPLOAD_ORDER = (EPISODE_FILE, AGENT_FILE, EXPOSURE_FILE, EXPERIMENT_FILE, *LOG_FILES, EVENTS_FILE, OUTCOME_FILE)
"""The order a bundle's files go to the store, bundle.json after all of them."""

_ID_SEGMENT = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")


class TraceStore(Protocol):
    """The four BlobStore methods a trace needs, so this package depends on a shape and not on carttrace.storage.

    Keys are "/"-separated and relative. `list(prefix)` returns every key that starts with `prefix` as a plain
    string, at any depth. `get_bytes` may raise when a key is missing; the reader asks `list` or `exists` first.
    Parameters are positional-only, so any store whose methods take (key) and (key, data) fits, whatever it calls them.
    """

    def put_bytes(self, key: str, data: bytes, /) -> object: ...

    def get_bytes(self, key: str, /) -> bytes: ...

    def exists(self, key: str, /) -> bool: ...

    def list(self, prefix: str, /) -> Iterable[str]: ...


if TYPE_CHECKING:

    def _blob_stores_fit(store: BlobStore) -> TraceStore:
        """Never called: mypy checks here that every carttrace.storage BlobStore can be handed to a trace writer."""
        return store


def bundle_prefix(header: EpisodeHeader) -> str:
    """The store key prefix of a bundle: `episodes/<YYYY-MM-DD of started_at, in UTC>/<run_id>/<episode_id>`."""
    for field_name, value in (("run_id", header.run_id), ("episode_id", header.episode_id)):
        if not _ID_SEGMENT.fullmatch(value):
            raise ValidationError(
                f"{field_name} {value!r} cannot be a storage key segment",
                hint="ids become directory names and keys; make them with carttrace.core.ids.stable_id",
            )
    return f"{EPISODES_PREFIX}{_utc_date(header.started_at)}/{header.run_id}/{header.episode_id}"


def _utc_date(started_at: str) -> str:
    hint = "write it with carttrace.core.iso(clock.now())"
    try:
        moment = datetime.fromisoformat(started_at)
    except ValueError as exc:
        raise ValidationError(f"started_at {started_at!r} is not an RFC 3339 time", hint=hint) from exc
    if moment.tzinfo is None:
        raise ValidationError(f"started_at {started_at!r} has no timezone", hint=hint)
    return moment.astimezone(UTC).date().isoformat()


def validate_as[M: BaseModel](model: type[M], data: Any, *, where: str) -> M:
    """`data` as a `model`, or a ValidationError naming where it came from.

    A file that declares another schema version is refused by name instead of failing on its first unknown field.
    Messages never quote input values, because unredacted input must not reach a log through an error.
    """
    declared = model.model_fields.get("schema_")
    expected = declared.default if declared is not None and isinstance(declared.default, str) else SCHEMA
    if declared is not None and isinstance(data, Mapping) and data.get("schema") != expected:
        raise ValidationError(
            f"{where} declares schema {data.get('schema')!r}; this code reads {expected}",
            hint="read it with the mc version that wrote it, or migrate it to the current schema",
        )
    try:
        return model.model_validate(data)
    except PydanticValidationError as exc:
        problems = "; ".join(
            f"{'.'.join(str(part) for part in err['loc']) or '<root>'}: {err['msg']}"
            for err in exc.errors(include_url=False, include_input=False, include_context=False)
        )
        raise ValidationError(
            f"{where} is not a valid {model.__name__}: {problems}",
            hint=f"pass the fields of carttrace.trace.{model.__name__}",
        ) from None


def parse_json_as[M: BaseModel](model: type[M], data: bytes, *, where: str) -> M:
    """JSON bytes (a staged or stored bundle file) as a `model`."""
    try:
        obj = json.loads(data)
    except ValueError as exc:
        raise ValidationError(
            f"{where} is not JSON: {exc}",
            hint="bundle files are written once and never edited; compare it with bundle.json using verify()",
        ) from exc
    return validate_as(model, obj, where=where)
