"""Run one published journey again with its recorded agent choices, and say whether the store ended in the same place.

The row is run by the collector's own runner (`carttrace.collect.runner.run_manifest`) on the row's own world, as the
original was, with two differences only: the agent's model is `RecordedChat` over the trace's turns, and the shopper
words its replies by rule (its decisions are code either way, D118). A v1 row decides by rule alone (D115), so nothing
else asks a model. Same world, task, seed and choices give the same sandbox: the replay passes when the final state
hash and the events chain head equal the trace's `state_digest`.
"""

from __future__ import annotations

import asyncio
import dataclasses
import itertools
import json
import re
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

from carttrace.collect.bindings import Bindings, default_bindings
from carttrace.collect.manifest import Manifest, seal
from carttrace.config.schema import Settings
from carttrace.core.canon import dumps, loads
from carttrace.core.errors import NotFoundError
from carttrace.repro.published import load_published, published_ids, serve_as
from carttrace.repro.recorded import (
    ANY_FIELD,
    Phone,
    RecordedChat,
    RecordedWorder,
    ReplayClock,
    call_times,
    phone_fields,
    recorded_calls,
)


@dataclass(frozen=True, slots=True)
class ReplayResult:
    episode_id: str
    end_reason: str | None
    """How the replay ended, against `published_end` from the trace."""
    published_end: str | None
    state: tuple[str | None, str | None]
    """(published, replayed) final sandbox state hash."""
    events: tuple[str | None, str | None]
    """(published, replayed) head of the sandbox's events chain."""
    calls: tuple[int, int]
    """(recorded model calls, calls the replay asked for)."""
    bundle: str | None

    @property
    def same(self) -> bool:
        """The replay reached the trace's place: the same events chain head (and state hash, where the trace
        publishes one), the same end, and every recorded model call asked for."""
        if self.events[0] is None:
            return False
        state = self.state[0] is None or self.state[0] == self.state[1]
        return (
            state
            and self.events[0] == self.events[1]
            and self.end_reason == self.published_end
            and self.calls[0] == self.calls[1]
        )


def one_row(manifest: Manifest, episode_id: str) -> Manifest:
    """The manifest cut to the one row of `episode_id`, sealed under the id of what it holds."""
    if not any(row.episode_id == episode_id for row in manifest.rows):
        raise NotFoundError(f"no row {episode_id} in manifest {manifest.id}", hint="check the world's manifest.json")
    body = loads(dumps(manifest.model_dump(mode="json", by_alias=True)))
    body.pop("id", None)
    body.pop("schema", None)
    body["rows"] = [r for r in body["rows"] if r["episode_id"] == episode_id]
    return seal(**body)


def _buyer(manifest: Manifest, header: Mapping[str, Any]) -> dict[str, str]:
    """The persona's email and phone from the task's buyer block, the values the placeholders stand for."""
    row = next(r for r in manifest.rows if r.episode_id == header["episode_id"])
    task = manifest.task(row)
    buyer = (task.hidden_spec or {}).get("buyer") or {}
    out: dict[str, str] = {}
    for key in ("email", "phone"):
        found = buyer.get(key)
        value = found.get("value") if isinstance(found, Mapping) else None
        if isinstance(value, str):
            out[key] = value
    return out


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def replay_settings(settings: Settings) -> Settings:
    """The settings a replay runs under: the shopper worded by rule, so no model is asked for anything."""
    shopper = settings.agents.shopper.model_copy(update={"wording": "rules"})
    return settings.model_copy(update={"agents": settings.agents.model_copy(update={"shopper": shopper})})


def recorded_bindings(
    base: Bindings,
    chat: RecordedChat,
    worder: RecordedWorder | None = None,
    clock: Any = None,
    served: str | None = None,
) -> Bindings:
    """The run's bindings with every agent built answering from `chat` instead of its model, and the shopper's
    replies worded as the trace has them (`worder`; None words them by rule)."""
    make = base.agent

    def voice(settings: Settings, task_id: str, persona: str | None) -> Any:
        if worder is None:
            return None
        from carttrace.agents.wording import ShopperVoice

        return ShopperVoice.from_settings(worder, settings.agents.shopper, task_id=task_id, persona=persona)

    def agent(mode: Any, **kwargs: Any) -> Any:
        built = make(mode, **kwargs)
        if hasattr(built, "_chat"):  # the reference agent asks no model: it is run again as it is
            built._chat = chat
        return built

    async def surface(envpack: Any, **kwargs: Any) -> Any:
        """The row's sandbox surface on the trace's clock (its own default is the system clock), its world answering
        to the id the trace served it under (`carttrace.repro.published`)."""
        serve_as(envpack, served)
        if clock is None or kwargs.get("kind") == "website":
            return await base.open_sandbox_surface(envpack, **kwargs)
        from carttrace.sandbox.server import open_sandbox_surface

        options, app, guard = kwargs.pop("options", None), kwargs.pop("app", None), kwargs.pop("guard", None)
        extra = {k: v for k, v in (("options", options), ("app", app), ("guard", guard)) if v is not None}
        return clock.watch(await open_sandbox_surface(envpack, clock=clock, **kwargs, **extra))

    def app(envpack: Any) -> Any:
        serve_as(envpack, served)  # the store is built on the served world before any surface opens
        return base.sandbox_app(envpack)

    return dataclasses.replace(base, agent=agent, shopper_voice=voice, open_sandbox_surface=surface, sandbox_app=app)


@contextmanager
def _harness_on(clock: ReplayClock) -> Iterator[None]:
    """The wallet's tools on the trace's clock too: each call moves it to when the trace made that call. The wallet
    has no surface to wrap, so its class is patched for the length of one replay (replays run one at a time)."""
    from carttrace.agents.checkout import CheckoutHarness

    call = CheckoutHarness.call

    async def timed(self: Any, name: str, arguments: Mapping[str, Any], **kwargs: Any) -> Any:
        clock.calling(name)
        return await call(self, name, arguments, **kwargs)

    CheckoutHarness.call = timed  # type: ignore[method-assign]
    try:
        yield
    finally:
        CheckoutHarness.call = call  # type: ignore[method-assign]


def phone_spellings(phone: str | None) -> list[str | None]:
    """The spellings of the persona's phone an agent writes, the task's own first. The trace's placeholder keys a
    phone on its digits alone, so which spelling the agent sent is not in it; the replay tries each, and the state
    hash says which one the agent used."""
    if not phone:
        return [None]
    digits = re.sub(r"[^0-9]", "", phone)
    local = digits[1:] if len(digits) == 11 and digits.startswith("1") else digits
    found = [phone, f"+{digits}", digits, local]
    if len(local) == 10:
        found += [
            f"{local[:3]}-{local[3:6]}-{local[6:]}",
            f"({local[:3]}) {local[3:6]}-{local[6:]}",
            f"+1 {local[:3]}-{local[3:6]}-{local[6:]}",
            f"+1 ({local[:3]}) {local[3:6]}-{local[6:]}",
        ]
    return list(dict.fromkeys(found))


async def replay(
    manifest: Manifest,
    bundle_dir: Path,
    *,
    settings: Settings,
    out: Path,
) -> ReplayResult:
    """Run the trace in `bundle_dir` again from `manifest` (its world's), writing the new bundle under `out`: once per
    spelling of the persona's phone until one reaches the trace's state, or every one has been tried."""
    header = loads((bundle_dir / "episode.json").read_bytes())
    buyer = _buyer(manifest, header)
    turns = read_jsonl(bundle_dir / "turns.jsonl")
    spellings = phone_spellings(buyer.get("phone")) if any("phone-" in json.dumps(t) for t in turns) else [None]
    result: ReplayResult | None = None
    for i, phone in enumerate(spellings):
        result = await _replay_once(
            manifest,
            bundle_dir,
            header,
            turns,
            email=buyer.get("email"),
            phone=phone,
            settings=settings,
            out=out / f"try-{i}",
        )
        if result.same:
            break
    assert result is not None
    if not result.same and len(spellings) > 1:
        tried = out / "try-0"
        for i, mixed in enumerate(mixed_spellings(spellings, phone_fields(turns), bundle_dir, tried)):
            result = await _replay_once(
                manifest,
                bundle_dir,
                header,
                turns,
                email=buyer.get("email"),
                phone=mixed,
                settings=settings,
                out=out / f"mixed-{i}",
            )
            if result.same:
                break
    return result


MAX_MIXED = 64
"""At most this many per-field spellings are tried after every single spelling failed."""


def _sizes(bundle: Path) -> dict[str, int]:
    """Each agent call's first request body size (`content-length`), by the call it served."""
    out: dict[str, int] = {}
    path = bundle / "wire.jsonl"
    for record in read_jsonl(path) if path.exists() else []:
        call, size = record.get("source_call_id"), (record.get("request_headers") or {}).get("content-length")
        if isinstance(call, str) and call not in out and str(size or "").isdigit():
            out[call] = int(str(size))
    return out


def mixed_spellings(
    spellings: Sequence[str | None], fields: Mapping[str, Mapping[str, int]], bundle_dir: Path, tried: Path
) -> list[Phone]:
    """Spellings per phone field, for an agent that wrote the persona's phone two ways in one trace (one way in
    `buyer.phone_number`, another in `fulfillment_details.phone_number`): the placeholder keys a phone on its digits,
    so the trace cannot say which. A choice is kept only when every request it touches would be as long as the trace
    recorded, measured against `tried`, the replay that spelled every field `spellings[0]`."""
    names = sorted({f for counts in fields.values() for f in counts})
    first = spellings[0]
    if len(names) < 2 or first is None:
        return []
    recorded = _sizes(bundle_dir)
    found = sorted(tried.rglob(f"{bundle_dir.name}/wire.jsonl"))
    replayed = _sizes(found[-1].parent) if found else {}
    out: list[Phone] = []
    for choice in itertools.product([s for s in spellings if s], repeat=len(names)):
        if len(set(choice)) == 1:
            continue
        chosen = dict(zip(names, choice, strict=True))
        fits = all(
            replayed[call] + sum(n * (len(chosen[f]) - len(first)) for f, n in counts.items()) == recorded[call]
            for call, counts in fields.items()
            if call in recorded and call in replayed
        )
        if fits:
            out.append({**chosen, ANY_FIELD: first})
    return out[:MAX_MIXED]


async def _replay_once(
    manifest: Manifest,
    bundle_dir: Path,
    header: Mapping[str, Any],
    turns: list[dict[str, Any]],
    *,
    email: str | None,
    phone: Phone,
    settings: Settings,
    out: Path,
) -> ReplayResult:
    from carttrace.collect.runner import run_manifest
    from carttrace.storage.local import LocalBlobStore

    outcome = loads((bundle_dir / "outcome.json").read_bytes())
    episode_id = str(header["episode_id"])
    calls = recorded_calls(turns, email=email, phone=phone)
    clock = ReplayClock(
        datetime.fromisoformat(str(header["started_at"]).replace("Z", "+00:00")),
        turns,
        read_jsonl(bundle_dir / "wire.jsonl") if (bundle_dir / "wire.jsonl").exists() else (),
    )
    chat = RecordedChat(calls, clock=clock, times=call_times(turns))
    plain = replay_settings(settings)
    store = LocalBlobStore(out)
    row = one_row(manifest, episode_id)
    served = (outcome.get("state_digest") or {}).get("env")
    world = settings.paths.var / "envpacks" / f"{row.rows[0].envpack}.json"
    with published_ids(), _harness_on(clock):
        if world.is_file():
            load_published(world)
        await run_manifest(
            row,
            settings=plain,
            live_flag=False,
            bindings=recorded_bindings(
                default_bindings(plain), chat, RecordedWorder(turns, email=email, phone=phone), clock, served
            ),
            store=store,
            clock=clock,
            handle_signals=False,
        )
    found = await asyncio.to_thread(lambda: sorted(out.rglob(f"{episode_id}/outcome.json")))
    again: Mapping[str, Any] = loads(found[-1].read_bytes()) if found else {}
    published = outcome.get("state_digest") or {}
    replayed = again.get("state_digest") or {}
    return ReplayResult(
        episode_id=episode_id,
        end_reason=again.get("end_reason"),
        published_end=outcome.get("end_reason"),
        state=(published.get("state"), replayed.get("state")),
        events=(published.get("events"), replayed.get("events")),
        calls=(len(calls), chat.asked),
        bundle=str(found[-1].parent) if found else None,
    )
