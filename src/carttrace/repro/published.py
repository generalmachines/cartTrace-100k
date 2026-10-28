"""Published worlds: an envpack with its internal provenance withheld, answering to the id it was published under.

Why. An envpack's id is the sha256 of its canonical bytes, and the id is part of the sandbox's state and of every event
it chains. A world as built names the internal queue store it stood in for and the capture it was matched to, in its
provenance; the public release withholds those (D102: the roster map stays private), which changes the bytes and so
the id. A replay on the withheld world would then reach a different state hash for that reason alone. So, while a
replay runs, a withheld world answers to its published id: the base world to the id its file is named for, and the
world the episode served (the base with the episode's faults sealed in) to the id the trace's outcome names. Nothing
else about the world changes, and the replay still has to reach the trace's state hash to pass.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from carttrace.core.canon import loads
from carttrace.core.errors import ValidationError
from carttrace.sandbox.envpack import EnvPack

ALIASES: dict[str, str] = {}
"""The id a withheld world's bytes give, to the id it was published under."""

_real_id = EnvPack.id


def raw_id(pack: EnvPack) -> str:
    """The id `pack`'s own bytes give, whatever it answers to."""
    return str(_real_id.fget(pack))  # type: ignore[attr-defined]


def _published_id(self: EnvPack) -> str:
    own = raw_id(self)
    return ALIASES.get(own, own)


@contextmanager
def published_ids() -> Iterator[None]:
    """Withheld worlds answer to their published ids for the length of the block."""
    EnvPack.id = property(_published_id)  # type: ignore[method-assign,assignment]
    try:
        yield
    finally:
        EnvPack.id = _real_id  # type: ignore[method-assign]
        ALIASES.clear()


def load_published(path: Path) -> EnvPack:
    """The world in `path`, registered under the id its file is named for. The file holds the withheld world's
    canonical document; its name is the id the release published it under."""
    from carttrace.sandbox.envpack.pack import _from_dict

    doc: Any = loads(path.read_bytes())
    pack = _from_dict(doc)
    if pack.to_bytes() != path.read_bytes():
        raise ValidationError(f"{path} is not a world's canonical document", hint="download the world file again")
    ALIASES[raw_id(pack)] = path.stem
    return pack


def serve_as(pack: EnvPack, published: str | None) -> None:
    """The world an episode serves answers to `published`, the id the trace's outcome names."""
    if published and raw_id(pack) != published:
        ALIASES[raw_id(pack)] = published
