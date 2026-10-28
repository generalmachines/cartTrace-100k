"""Replica fidelity: does a store's replica answer what the store answered?

Why a replay. A replica is worth benchmarking only if it answers like its store (decision D12). The check takes the
catalog requests a capture recorded (the search_catalog and get_product calls that got an answer), sends a seeded
sample of them, as captured, to the replica in process (`open_sandbox_surface`: the agent's own wire, no socket), and
compares each normalised answer with the store's, field by field:
- which products, in which order, and the pagination (has a next page, the cursor);
- per product: its title, price range, options and their values (with availability signals where the store sent
  them), its selection, the variants it listed and their order, each detail it carries (description, handle, url,
  categories, list price range, media, rating, tags, metadata) and the policies that cover it;
- per variant: its title, price and currency, availability and status, options, and each detail it carries (its own
  description, SKU, barcodes, list price, unit price, media, metadata, ...).
A detail sent by one side only is one difference, so the presence of each detail is compared as well as its value. The
score is the share of fields that agree; the report keeps the first differences with their path and both values.

What it cannot see: products, variants and option values the capture never saw (only what was recorded can be
replayed), what the store would have answered to requests nobody sent, and anything outside catalog replies. Policy
tool calls are listed as skipped, because the sandbox serves no store MCP surface.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal

from pydantic import BaseModel, ConfigDict, Field

from carttrace.core.canon import digest, loads
from carttrace.core.clock import Clock
from carttrace.core.ids import seeded_rng
from carttrace.sandbox.catalogs.replica import DETAIL_TOOL, SEARCH_TOOL, covered_products
from carttrace.sandbox.envpack import EnvPack
from carttrace.trace.capture import CapturedCall, StoreCapture, load_capture
from carttrace.trace.products import product_documents, read_product, reply_payload

FIDELITY_SCHEMA: Final = "mc.replica-fidelity.v1"
MAX_DIFFERENCES = 20
TEXT_CAP = 160
"""Characters of a long value a difference keeps."""
SESSION = "fidelity"
"""The sandbox session the replay acts for."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Difference(_Frozen):
    call: int
    """The capture's call index."""
    tool: str
    asked: str
    """The query or the id the call asked for."""
    path: str
    live: Any = None
    replica: Any = None


class ToolScore(_Frozen):
    replayed: int = 0
    exact: int = 0
    """Answers whose every field agreed."""
    fields: int = 0
    equal: int = 0


class FidelityReport(_Frozen):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.replica-fidelity.v1"] = Field(default=FIDELITY_SCHEMA, alias="schema")
    capture_id: str
    envpack_id: str
    store_id: str
    replayed: int
    exact: int
    fields: int
    equal: int
    score: float
    """equal / fields, 0 when nothing was compared."""
    by_tool: dict[str, ToolScore] = Field(default_factory=dict)
    skipped: list[str] = Field(default_factory=list)
    differences: list[Difference] = Field(default_factory=list)
    """The first MAX_DIFFERENCES, in replay order."""

    @property
    def perfect(self) -> bool:
        return self.fields > 0 and self.equal == self.fields


async def replica_fidelity(
    capture: StoreCapture | Path,
    envpack: EnvPack,
    *,
    sample: int | None = None,
    seed: int = 0,
    clock: Clock | None = None,
) -> FidelityReport:
    """Replay `sample` of the capture's recorded catalog requests (all of them when None, a seeded draw otherwise)
    against `envpack`, the capture's replica, and compare the answers (see the module docstring)."""
    from carttrace.sandbox.server import open_sandbox_surface

    cap = capture if isinstance(capture, StoreCapture) else load_capture(Path(capture))
    calls = [c for c in cap.successful(SEARCH_TOOL, DETAIL_TOOL) if _answered(c)]
    skipped = [
        f"call {c.index} {c.tool}: the sandbox serves no {c.tool}"
        for c in cap.successful()
        if c.tool not in (SEARCH_TOOL, DETAIL_TOOL)
    ]
    if sample is not None and sample < len(calls):
        drawn = seeded_rng("fidelity", cap.id, seed).sample(calls, max(0, sample))
        calls = sorted(drawn, key=lambda c: c.index)
    scores: dict[str, dict[str, int]] = {}
    differences: list[Difference] = []
    surface = await open_sandbox_surface(envpack, session_id=SESSION, clock=clock)
    try:
        for call in calls:
            result = await surface.call(call.tool, dict(call.arguments))
            replica = result.structured if result.structured is not None else _json(result.content)
            live_fields = answer_fields(call.tool, call.reply)
            replica_fields = answer_fields(call.tool, replica)
            paths = sorted(set(live_fields) | set(replica_fields))
            unequal = [p for p in paths if live_fields.get(p) != replica_fields.get(p)]
            tally = scores.setdefault(call.tool, {"replayed": 0, "exact": 0, "fields": 0, "equal": 0})
            tally["replayed"] += 1
            tally["exact"] += not unequal
            tally["fields"] += len(paths)
            tally["equal"] += len(paths) - len(unequal)
            for path in unequal[: max(0, MAX_DIFFERENCES - len(differences))]:
                differences.append(
                    Difference(
                        call=call.index,
                        tool=call.tool,
                        asked=_asked(call),
                        path=path,
                        live=_brief(live_fields.get(path)),
                        replica=_brief(replica_fields.get(path)),
                    )
                )
    finally:
        await surface.aclose()
    fields = sum(t["fields"] for t in scores.values())
    equal = sum(t["equal"] for t in scores.values())
    return FidelityReport(
        capture_id=cap.id,
        envpack_id=envpack.id,
        store_id=cap.store.store_id,
        replayed=sum(t["replayed"] for t in scores.values()),
        exact=sum(t["exact"] for t in scores.values()),
        fields=fields,
        equal=equal,
        score=round(equal / fields, 6) if fields else 0.0,
        by_tool={tool: ToolScore.model_validate(tally) for tool, tally in sorted(scores.items())},
        skipped=skipped,
        differences=differences,
    )


def answer_fields(tool: str, reply: Any) -> dict[str, Any]:
    """A catalog answer as the flat fields a fidelity check compares, keyed by a readable path."""
    payload = reply_payload(reply) or {}
    docs = product_documents(payload)
    products = [read_product(doc) for doc in docs]
    ids = [p.id for p in products]
    out: dict[str, Any] = {"$.products": ids}
    if tool == SEARCH_TOOL:
        raw = payload.get("pagination")
        pagination: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
        out["$.pagination.has_next_page"] = pagination.get("has_next_page")
        out["$.pagination.cursor"] = pagination.get("cursor")
    covering = _covering(payload, ids, single=tool == DETAIL_TOOL)
    for product, doc in zip(products, docs, strict=True):
        at = f"$.products[{product.id}]"
        out[f"{at}.title"] = product.title
        out[f"{at}.price_range"] = doc.get("price_range")
        out[f"{at}.options"] = [
            [name, [[v.label, v.available, v.exists] for v in values]] for name, values in product.options
        ]
        if tool == DETAIL_TOOL:
            out[f"{at}.selected"] = [list(pair) for pair in product.selected or ()]
        out[f"{at}.variants"] = [v.id for v in product.variants]
        out[f"{at}.policies"] = covering.get(product.id, [])
        out.update({f"{at}.{name}": value for name, value in product.details.items()})
        for variant in product.variants:
            vat = f"{at}.variants[{variant.id}]"
            out[f"{vat}.title"] = variant.title
            out[f"{vat}.price"] = [variant.price, variant.currency]
            out[f"{vat}.availability"] = [variant.available, variant.status]
            out[f"{vat}.options"] = [list(pair) for pair in variant.options]
            out.update({f"{vat}.{name}": value for name, value in variant.details.items()})
    return out


def _covering(payload: Mapping[str, Any], ids: Sequence[str], *, single: bool) -> dict[str, list[str]]:
    """Per product, the digests of the policies that cover it in one answer."""
    out: dict[str, set[str]] = {}
    for policy in payload.get("policies") or ():
        if not isinstance(policy, Mapping):
            continue
        key = digest({k: v for k, v in policy.items() if k != "applies_to"}, 12)
        for pid in covered_products(policy.get("applies_to"), ids, single=single):
            out.setdefault(pid, set()).add(key)
    return {pid: sorted(keys) for pid, keys in out.items()}


def _answered(call: CapturedCall) -> bool:
    payload = reply_payload(call.reply)
    ucp = payload.get("ucp") if payload is not None else None
    return payload is not None and not (isinstance(ucp, Mapping) and ucp.get("status") == "error")


def _asked(call: CapturedCall) -> str:
    raw = call.arguments.get("catalog")
    catalog: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    asked = catalog.get("query") if call.tool == SEARCH_TOOL else catalog.get("id")
    raw_pagination = catalog.get("pagination")
    cursor = raw_pagination.get("cursor") if isinstance(raw_pagination, Mapping) else None
    return f"{asked} (page {cursor})" if cursor else str(asked)


def _brief(value: Any) -> Any:
    if isinstance(value, str) and len(value) > TEXT_CAP:
        return value[:TEXT_CAP] + "..."
    if isinstance(value, Mapping | list):
        text = str(value)
        return value if len(text) <= TEXT_CAP else text[:TEXT_CAP] + "..."
    return value


def _json(text: Any) -> Any:
    if not isinstance(text, str) or not text:
        return None
    try:
        return loads(text)
    except ValueError:
        return None
