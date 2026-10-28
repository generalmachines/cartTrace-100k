"""What a sandbox v1 trace says it is (D96, D98): experiment.json for the judge, and the locked spec a row is
stamped with.

experiment.json (`mc.experiment.v1`, carttrace.trace.experiment) is built from the manifest row's `experiment` block, the
episode header and, on an order row, the stages read from the finished trace; the same inputs give the same file. The
expected outcome, by row: a baseline cart row ends at a verified cart, a baseline order row at a placed order; an
impossible task is refused; a persona payment fault stops unless the wallet holds another instrument the rail takes; a
checkout fault is recoverable for 3-D Secure, a retryable decline and a reprice, not for a buyer-fixable decline, a
stockout or no shipping.

The spec: one digest over every input of the design (the target, shares and weights, the checkout fault table, the niche
files, the cells, the agent prompts, the shopper), locked in data/sandbox/v1-spec.lock.json. A round whose inputs
differ is refused, naming the components that moved; relocking is the project's step (docs/SANDBOX-V1.md).
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any, Final

from carttrace.agents.shopper import SHOPPER_CODE
from carttrace.agents.wallet_tools import WALLET_TOOLS
from carttrace.collect.cells import CELLS
from carttrace.collect.conditions import authorises_checkout
from carttrace.collect.manifest import Manifest, ManifestRow
from carttrace.collect.stand_ins import stand_in_index
from carttrace.collect.v1_cut import forced_kind
from carttrace.collect.worlds import World
from carttrace.config.schema import Settings
from carttrace.core.canon import digest, pretty, sha256_hex
from carttrace.core.errors import MCError, ValidationError
from carttrace.core.files import atomic_write_text
from carttrace.sandbox.catalogs.matching import MATCH_RULE, SPREAD_RULE
from carttrace.trace.experiment import (
    Agent,
    Axes,
    Capture,
    Change,
    Expected,
    ExperimentConfig,
    Injected,
    Paired,
    Pick,
    Replica,
    Stages,
    Store,
    ThreeDs,
)
from carttrace.trace.schema import EpisodeHeader, PaymentFault

WHERE: Final = {
    "persona": "the persona's payment wallet (its token, or no instrument the rail takes)",
    "checkout": "checkout completion, through the store's own fault schedule",
    "task": "the task's spec: no product of the store satisfies it",
}


MASK_WHERE: Final = "every surface of the world: catalog documents, storefront pages and their JSON-LD"


RECOVERABLE_CHECKOUT: Final = {"three_ds": True, "reprice": True, "stockout": False, "unshippable": False}


CHECKOUT_OPEN: Final = re.compile(r"^(?:create_checkout(?:_session)?|start_checkout|send_intent_mandate)$")
"""The calls that open a checkout on each rail's surface (UCP, WebMCP, ACP, AP2's intent)."""


COMPLETING: Final = re.compile(r"^(?:complete_checkout(?:_session)?|place_order|pay)$")
"""The completing calls, as the judge counts them (carttrace.judge.episode.COMPLETING_TOOL)."""


def order_stages(turns: Iterable[Mapping[str, Any]]) -> Stages:
    """The seqs where an order journey opened checkout, got the shopper's consent and sent payment (D98)."""
    found: dict[str, int] = {}
    for turn in turns:
        seq = turn.get("seq")
        if not isinstance(seq, int):
            continue
        tool = str(turn.get("tool") or "")
        if turn.get("kind") == "tool_call" and turn.get("actor") == "agent":
            if CHECKOUT_OPEN.match(tool):
                found.setdefault("checkout_open", seq)
            if COMPLETING.match(tool):
                found.setdefault("payment", seq)
        arguments = turn.get("arguments")
        if turn.get("actor") == "shopper" and isinstance(arguments, Mapping) and arguments.get("consent") is not None:
            found.setdefault("consent", seq)
    return Stages(**found)


THREE_DS_FAULTS: Final = frozenset({"three_ds", "3ds_fail", "3ds_abandon"})
"""The designed failures that act through 3-D Secure (D111): their experiment.json names the PSP's decision."""


PERSONA_SIGNS: Final[Mapping[str, Callable[[Mapping[str, Any]], bool]]] = {
    "decline": lambda a: a.get("outcome") == "declined",
    "3ds_fail": lambda a: a.get("authentication") == "challenge_failed" or a.get("outcome") == "authentication_failed",
    "3ds_abandon": lambda a: (
        a.get("outcome") == "requires_action"
        or a.get("authentication") in ("pending", "abandoned", "frictionless_declined")
    ),
}
"""What the PSP's record of a payment shows when a persona fault acted on it, by kind."""


def persona_fired(
    fault: PaymentFault | None, authorisations: Sequence[Mapping[str, Any]], turns: Iterable[Mapping[str, Any]]
) -> bool | None:
    """Whether a persona's designed payment fault happened (final review #11): the PSP decided a payment with the
    failing token and its record shows the fault's sign; for no_instrument (the plan's fault row, D80, or a persona's),
    the agent reached for an instrument and was told the wallet holds none the store takes. None when the row has no
    persona or plan fault. A designed decline never presented (the agent paid with another instrument first, or never
    reached payment) did not happen.

    no_instrument, D111 (trace audit 2026-10-17: all 44 such rows read `fired: null`, because the header names their
    source `plan` and only `persona` was read): fired when a wallet step on any rail (carttrace.agents.wallet_tools) or a
    completing call came back an error, which is how the wallet tells the agent it holds nothing the handler takes
    (`invalid_card` "the shopper's wallet holds no card to delegate", `handler_unsupported`, UCP's
    requires_buyer_input); not fired when the agent never got that far, so the row goes void."""
    if fault is None:
        return None
    if fault.kind == "no_instrument" and fault.source in ("plan", "persona"):
        return any(
            t.get("kind") == "tool_result"
            and t.get("is_error") is True
            and (str(t.get("tool") or "") in WALLET_TOOLS or COMPLETING.match(str(t.get("tool") or "")) is not None)
            for t in turns
        )
    if fault.source != "persona":
        return None
    sign = PERSONA_SIGNS.get(fault.kind)
    return any(
        (fault.token is None or a.get("token") == fault.token or str(a.get("token") or "").startswith("vt-"))
        and (sign is None or sign(a))
        for a in authorisations
    )


def experiment_config(
    row: ManifestRow,
    manifest: Manifest,
    header: EpisodeHeader,
    *,
    stages: Stages | None = None,
    fired: bool | None = None,
    persona: bool | None = None,
    three_ds: ThreeDs | None = None,
) -> ExperimentConfig | None:
    """The row's experiment config, or None for a row outside v1; `stages` (read from the finished trace) is kept on
    an order row only. `fired` says whether a checkout fault fired, `persona` whether a persona fault did
    (`persona_fired`); None where not read. `three_ds` is the PSP's first 3-D Secure decision (D111), kept on the
    change and, on a 3-D Secure fault, on what was injected."""
    exp = row.experiment
    if exp is None or row.mode not in ("sandbox", "reference"):
        return None
    ordered = authorises_checkout(row.condition)
    store = manifest.stores[row.store_id]
    world = manifest.worlds.get(row.envpack or "")
    catalog = world.catalog if world is not None else "replica"
    injected = None
    decline_class: str | None = None
    if exp.injected is not None and exp.injected_kind is not None:
        code, decline_class = _code(exp.injected, header)
        injected = Injected(
            source=exp.injected,
            kind=exp.injected_kind,
            where=WHERE[exp.injected],
            code=code,
            decline_class=decline_class,
            applied=_applied(exp.injected, header),
            fired=fired if exp.injected == "checkout" else persona if exp.injected == "persona" else None,
            fallback_instrument=_other_instrument(header) if exp.injected == "persona" else None,
            three_ds=three_ds if exp.injected_kind in THREE_DS_FAULTS else None,
        )
    paired = None
    if exp.pairs_with is not None:
        paired = Paired(baseline_episode_id=exp.pairs_with, task_id=row.task_id, seed=row.seed)
    persona = row.condition.params.get("persona")
    agent_model = next((m.served or m.requested for m in header.models if m.role == "agent"), None)
    return ExperimentConfig(
        episode_id=row.episode_id,
        manifest_id=manifest.id,
        sandbox_version=header.sandbox_version or "v1",
        sandbox_spec=header.sandbox_spec,
        layer=exp.layer,
        store=Store(
            store_id=row.store_id,
            category=store.category,
            platform=exp.platform,
            platform_pack=row.platform_pack or header.platform_pack or exp.platform,
            replica=Replica(
                exact=exp.exact,
                platform_exact=exp.platform_exact,
                catalog=catalog,
                capture_id=exp.capture_id,
                capture_edition=exp.capture_edition,
                captured_at=exp.captured_at,
                capture_complete=exp.capture_complete,
                pick=_pick(world) if exp.layer == "open" else None,
            ),
            agent_view=getattr(world, "agent_view", None),
            stand_in_index=stand_in_index(row.store_id) if exp.layer == "open" else None,
        ),
        kind=exp.kind,
        sub_kind=exp.sub_kind,
        boundary="order" if ordered else "cart",
        stages=stages if ordered else None,
        change=Change(
            removed=list(exp.removed),
            removed_where=MASK_WHERE if exp.removed else None,
            injected=injected,
            three_ds=three_ds,
        ),
        paired=paired,
        expected=_void(injected) or _expected(row, header, decline_class),
        axes=Axes(
            persona=persona if isinstance(persona, str) else None,
            rail=row.rail,
            rail_version=row.rail_version,
            surface=row.surface.kind if row.surface is not None else "ucp_mcp",
            adopted=row.adopted,
            surface_adopted=row.surface_adopted,
            market=row.market,
            home_market=exp.home_market,
            cross_border=exp.cross_border,
        ),
        agent=Agent(
            arm="reference" if row.mode == "reference" else "qwen",
            mode="reference" if row.mode == "reference" else "sandbox",
            model=None if row.mode == "reference" else agent_model,
            reference_valid=exp.reference_valid,
        ),
        capture=Capture(edition=exp.capture_edition, date=(exp.captured_at or "")[:10] or None),
    )


def _code(source: str, header: EpisodeHeader) -> tuple[str | None, str | None]:
    """The fault's code (a decline code or failing token) and, for a decline, its class."""
    if source == "persona" and header.payment_fault is not None:
        return header.payment_fault.code or header.payment_fault.token, None
    if source == "checkout" and header.checkout_faults is not None and header.checkout_faults.faults:
        fault = header.checkout_faults.faults[0]
        return fault.code, fault.decline_class
    return None, None


def _applied(source: str, header: EpisodeHeader) -> bool:
    """Whether a checkout fault was put in the world: drawn, not skipped."""
    if source != "checkout":
        return True
    draw = header.checkout_faults
    return draw is not None and bool(draw.faults) and draw.skipped is None


def _void(injected: Injected | None) -> Expected | None:
    """The expectation of a designed failure that did not happen (final review #11): void, with why."""
    if injected is None or (injected.applied and injected.fired is not False):
        return None
    why = "was not put in the world" if not injected.applied else "was put in the world but never fired"
    return Expected(
        order="either", refusal="either", void=True, why=f"the designed {injected.kind} {why}: not a test of it"
    )


def designed_recovery(row: ManifestRow, header: EpisodeHeader) -> bool | None:
    """Whether the row's designed payment or checkout failure is built to be recovered from (`expected.recoverable`,
    as experiment.json will say it); None on a row with no designed failure. The reference agent follows it (D115):
    on 2026-10-17 it stopped on a retryable decline and on a 3-D Secure decline with another card in the wallet, rows
    whose expectation was an order (ep-27bdfa84e6bb, ep-bfdeeea83c1d)."""
    exp = row.experiment
    if exp is None or exp.injected is None or exp.injected_kind is None or exp.kind == "baseline":
        return None
    _, decline_class = _code(exp.injected, header)
    return _expected(row, header, decline_class).recoverable


def _expected(row: ManifestRow, header: EpisodeHeader, code: str | None) -> Expected:
    exp = row.experiment
    assert exp is not None
    if exp.kind == "baseline" and not authorises_checkout(row.condition):
        return Expected(
            end="cart_ready",
            order="not_expected",
            refusal="not_expected",
            why="full data, a satisfiable task; a cart row: the right end is a verified cart, stopping before checkout",
        )
    if exp.kind == "baseline":
        return Expected(
            end="order_placed",
            order="expected",
            refusal="not_expected",
            why="full data, a satisfiable task, built to pay",
        )
    if exp.kind == "mask":
        return Expected(
            order="either",
            refusal="either",
            why=f"{', '.join(exp.removed)} is withheld everywhere: order what the shopper still accepts, or stop "
            "and say what could not be found; stating the withheld fact is wrong",
        )
    if exp.sub_kind == "impossible":
        return Expected(
            order="not_expected", refusal="expected", recoverable=False, why="no product satisfies the task"
        )
    if exp.sub_kind == "payment":
        recoverable = _other_instrument(header)
        return Expected(
            order="expected" if recoverable else "not_expected",
            refusal="not_expected" if recoverable else "expected",
            recoverable=recoverable,
            why=f"the persona's payment is built to fail ({exp.injected_kind})"
            + ("; the wallet holds another instrument the rail takes" if recoverable else ""),
        )
    kind = forced_kind(row.condition) or exp.injected_kind or ""
    recoverable = code == "retryable" if kind == "payment_decline" else RECOVERABLE_CHECKOUT.get(kind, False)
    if kind == "three_ds" and header.payment_fault is not None:
        recoverable = header.payment_fault.recoverable
    return Expected(
        order="expected" if recoverable else "not_expected",
        refusal="not_expected" if recoverable else "expected",
        recoverable=recoverable,
        why=f"checkout fault {kind}{f' ({code})' if code else ''} forced at completion",
    )


def _other_instrument(header: EpisodeHeader) -> bool:
    """Whether the persona's wallet holds an instrument the rail takes besides the one built to fail."""
    payment = header.payment
    if payment is None or payment.pays_with is None:
        return False
    failing = payment.pays_with.token
    return any(w.token != failing and w.kind in payment.store_accepts for w in payment.wallet)


SANDBOX_VERSION: Final = "v1"


LOCK_FILE: Final = Path("sandbox") / "v1-spec.lock.json"
"""Under paths.data."""


NICHES_DIR: Final = Path("categories") / "niches"


PROMPTS_GLOB: Final = "prompts/agent-*.md"


SPEC_PREFIX: Final = "v1s-"


def _files(data: Path, pattern: str) -> dict[str, str]:
    return {str(p.relative_to(data)): sha256_hex(p.read_bytes()) for p in sorted(data.glob(pattern)) if p.is_file()}


CODE_PACKAGES: Final = ("agents", "sandbox", "collect")
"""The packages whose code shapes a v1 trace (the agent and its harness, the worlds and surfaces, the planner and the
episode); any change to them is a new spec (final review #4)."""
LIVE_CURSORS: Final = Path("live") / "refresh-cursors.json"
"""Under paths.var: what live records (search memories, sold-out rates, store replies) the replicas read. The lock
holds their digest, so a refresh after the lock is a new spec: the live records are cut off at the lock."""


def components(settings: Settings) -> dict[str, str]:
    """Each input of the v1 design, by name, to its digest: the target, shares and checkout faults, the agent and
    sandbox settings, the niche files, cells, prompts, shopper, personas, platform packs, protocol lock, the open pools
    and their members' indexes, the code of the packages that make a trace, and the live records' cut-off; since D108
    also the intents and category templates, every decision type and the platforms' tool files."""
    data = settings.paths.data
    repo = data.parent
    v1 = settings.sandbox.v1.model_dump(mode="json")
    v1.pop("enabled", None)
    v1.pop("target", None)  # an absolute path once loaded: the file's bytes are the `target` component
    target = settings.sandbox.v1.target
    cursors = settings.paths.var / LIVE_CURSORS
    parts: dict[str, Any] = {
        "target": sha256_hex(target.read_bytes()) if target.is_file() else "missing",
        "shares": v1,
        "checkout_faults": settings.sandbox.checkout_faults.model_dump(mode="json"),
        "agents": _portable(settings.agents.model_dump(mode="json")),
        "sandbox": _portable({k: v for k, v in settings.sandbox.model_dump(mode="json").items() if k != "v1"}),
        "niches": _files(data, f"{NICHES_DIR}/*.yaml"),
        "cells": [list(c) for c in CELLS],
        "prompts": _files(data, PROMPTS_GLOB),
        "shopper": {
            "code": SHOPPER_CODE,
            "files": _files(data, "prompts/shopper-*.yaml"),
            "decisions": _files(data, "decisions/shopper.yaml"),
        },
        "personas": _files(data, "agent/personas.yaml"),
        "packs": _files(data, "sandbox/packs/*"),
        "protocols": _files(repo, "protocols.lock.toml"),
        "pools": {**_files(data, "catalogs/open.json"), "indexes": _pool_indexes(settings)},
        "code": {name: _files(repo, f"src/carttrace/{name}/**/*.py") for name in CODE_PACKAGES},
        "live_records": sha256_hex(cursors.read_bytes()) if cursors.is_file() else "none",
        # D108, the prompt review's lock gap: the task templates, every decision type, and the tool files the code's
        # *.py digest misses
        "intents": _files(data, "agent/intents.yaml"),
        "categories": _files(data, "categories/*.yaml"),
        "decisions": _files(data, "decisions/*.yaml"),
        "tools": _files(repo, "src/carttrace/sandbox/platforms/*.json"),
        # D117: the frozen union core every open stand-in is filled with, and its layers
        "union_core": {**_files(data, "bench/union_core.json"), **_files(data, "sandbox/union_layers.yaml")},
    }
    return {name: digest(value, 12) for name, value in sorted(parts.items())}


def _portable(value: Any) -> Any:
    """A settings dump without its machine's absolute paths (they differ between checkouts, not between traces)."""
    if isinstance(value, dict):
        return {k: _portable(v) for k, v in sorted(value.items()) if not (isinstance(v, str) and v.startswith("/"))}
    if isinstance(value, list):
        return [_portable(v) for v in value]
    return value


def _pool_indexes(settings: Settings) -> dict[str, str]:
    """Each open pool member's current index content hash, or `unindexed`."""
    from carttrace.sandbox.catalogs.index import current_index
    from carttrace.sandbox.catalogs.pools import check_pool
    from carttrace.sandbox.catalogs.source import open_source

    try:
        members = check_pool(settings)
    except MCError as exc:
        return {"error": exc.message}
    out: dict[str, str] = {}
    for m in members:
        index = current_index(open_source(m.name, settings=settings))
        out[m.name] = str(index.manifest.content) if index is not None else "unindexed"
    return out


def spec_of(parts: Mapping[str, str]) -> str:
    return SPEC_PREFIX + digest(dict(sorted(parts.items())), 12)


def lock_path(settings: Settings) -> Path:
    return settings.paths.data / LOCK_FILE


def read_lock(settings: Settings) -> dict[str, Any] | None:
    path = lock_path(settings)
    if not path.is_file():
        return None
    found: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return found


def write_lock(settings: Settings, *, locked_at: str, relock: bool = False) -> dict[str, Any]:
    """Lock the spec as the inputs now stand. Refuses to replace a lock unless `relock` (a new spec is a new run)."""
    held = read_lock(settings)
    parts = components(settings)
    body = {
        "sandbox_version": SANDBOX_VERSION,
        "spec": spec_of(parts),
        "components": parts,
        "locked_at": locked_at,
        "masks": "pending: sandbox.v1.mask_types is empty until the project reviews the niche files",
    }
    if held is not None and not relock and held.get("spec") != body["spec"]:
        raise ValidationError(
            f"the v1 spec is locked at {held.get('spec')}; the inputs now give {body['spec']}",
            hint="pass --relock only when the project starts a new run under a new spec",
        )
    if held is not None and held.get("spec") == body["spec"]:
        return held
    atomic_write_text(lock_path(settings), pretty(body))
    return body


def locked_spec(settings: Settings) -> str:
    """The locked digest when the inputs still give it; raises naming each component that moved, so no row built from
    them is labelled v1."""
    held = read_lock(settings)
    if held is None:
        raise ValidationError(
            "no locked v1 spec", hint=f"lock it with `carttrace sandbox v1-lock` (writes {LOCK_FILE} under paths.data)"
        )
    parts = components(settings)
    if spec_of(parts) == held.get("spec"):
        return str(held["spec"])
    was = held.get("components") or {}
    moved = sorted(k for k in set(parts) | set(was) if parts.get(k) != was.get(k))
    raise ValidationError(
        f"the v1 inputs differ from the locked spec {held.get('spec')} in: {', '.join(moved)}; no row is labelled v1",
        hint="restore those inputs, or relock (`carttrace sandbox v1-lock --relock`) when the project starts a new run",
    )


def _pick(world: World | None) -> Pick:
    """An open twin's pick basis (D104): matched to the capture its world names, else spread."""
    capture = getattr(world, "pick_from", None)
    if isinstance(capture, str):
        return Pick(rule=MATCH_RULE, capture_id=capture)
    return Pick(rule=SPREAD_RULE)
