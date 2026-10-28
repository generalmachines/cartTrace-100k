"""UCP request payloads to core command fields: line items, buyer, context, discount codes, fulfillment and payment.

Why a translation. The core speaks protocol-neutral commands with strict types (`carttrace.sandbox.core.commands`), and a
UCP payload says the same things in UCP's shape: a line item's `item.id` is the core's `variant_id` and its `id` the
core's `line_id`; a buyer's name is not stored (the core keeps contact facts only); a fulfillment method with its
destination and group selections becomes one core fulfillment group per UCP group. Payloads reach here only after
they passed the release's input schema, so this module handles meaning, not types.

Fulfillment is rendered as one method per core group (`carttrace.sandbox.surfaces.ucp.render`), so a method's `id`
(`fm-<group>`) or its group's `id` names the core group it updates. An address the core cannot use (no two-letter
country) is not sent; a recoverable `invalid_address` message says so instead, and the rest of the update goes on.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from carttrace.sandbox.core.state import Group
from carttrace.sandbox.surfaces.ucp.messages import Message, Paths, error
from carttrace.sandbox.surfaces.ucp.render import ADDRESS_FIELDS, group_of_method

METHODS = ("shipping", "pickup")
CONTACT = ("email", "phone_number")
"""Buyer fields the core keeps. Names are not stored: the sandbox keeps no personal names."""


@dataclass(slots=True)
class Translated:
    """Core command fields for one request, with where each request part points and what could not be used."""

    fields: dict[str, Any] = field(default_factory=dict)
    paths: Paths = field(default_factory=Paths)
    notes: list[Message] = field(default_factory=list)


def line_inputs(raw: object) -> tuple[list[dict[str, Any]], tuple[tuple[str | None, str | None], ...]]:
    """UCP line items to core lines (`variant_id`, `line_id`, `quantity`), and (line id, item id) per request line."""
    lines: list[dict[str, Any]] = []
    refs: list[tuple[str | None, str | None]] = []
    for li in raw if isinstance(raw, list) else ():
        if not isinstance(li, Mapping):
            continue
        item = li.get("item")
        item_id = item.get("id") if isinstance(item, Mapping) and isinstance(item.get("id"), str) else None
        line_id = li.get("id") if isinstance(li.get("id"), str) else None
        entry: dict[str, Any] = {"quantity": li.get("quantity")}
        if item_id is not None:
            entry["variant_id"] = item_id
        if line_id is not None:
            entry["line_id"] = line_id
        lines.append(entry)
        refs.append((line_id, item_id))
    return lines, tuple(refs)


def buyer(raw: object, *, keep_out: Mapping[str, str] | None = None) -> dict[str, str] | None:
    """The contact facts of a UCP buyer, leaving out any field `keep_out` already holds (a cart's buyer wins over a
    checkout payload's on conversion)."""
    if not isinstance(raw, Mapping):
        return None
    out = {k: str(raw[k]) for k in CONTACT if isinstance(raw.get(k), str) and k not in (keep_out or {})}
    return out or None


def country(context: object) -> str | None:
    """A context's `address_country` when it is a two-letter code; the core knows markets by those."""
    code = context.get("address_country") if isinstance(context, Mapping) else None
    return code.upper() if isinstance(code, str) and len(code) == 2 and code.isalpha() else None


def discount_codes(payload: Mapping[str, Any]) -> list[str] | None:
    """`discounts.codes` when the payload sends them (a full replacement, `[]` clears), else None."""
    d = payload.get("discounts")
    codes = d.get("codes") if isinstance(d, Mapping) else None
    return [str(c) for c in codes] if isinstance(codes, list) else None


def fulfillment_groups(
    payload: Mapping[str, Any], existing: Sequence[Group]
) -> tuple[list[dict[str, Any]] | None, tuple[str | None, ...], list[Message]]:
    """UCP `fulfillment.methods` to core groups, the core group each method maps to, and refusals of parts that cannot
    be used. None when the payload does not touch fulfillment."""
    f = payload.get("fulfillment")
    methods = f.get("methods") if isinstance(f, Mapping) else None
    if not isinstance(methods, list):
        return None, (), []
    by_id = {g.id: g for g in existing}
    groups: list[dict[str, Any]] = []
    mapped: list[str | None] = []
    notes: list[Message] = []
    for i, m in enumerate(methods):
        at = f"$.fulfillment.methods[{i}]"
        if not isinstance(m, Mapping):
            mapped.append(None)
            continue
        named = group_of_method(m["id"]) if isinstance(m.get("id"), str) else None
        known = by_id.get(named) if named is not None else None
        kind = m.get("type") if isinstance(m.get("type"), str) else (known.method if known else "shipping")
        if kind not in METHODS:
            notes.append(
                error("unsupported_method", f"This store fulfils by {' or '.join(METHODS)}", path=f"{at}.type")
            )
            mapped.append(None)
            continue
        destination, location = _destination(m, kind, at, notes)
        line_ids = m.get("line_item_ids")
        selections = [g for g in m.get("groups") or () if isinstance(g, Mapping)] or [{}]
        first: str | None = None
        for k, g in enumerate(selections):
            gid = g.get("id") if isinstance(g.get("id"), str) else (named if k == 0 else None)
            entry: dict[str, Any] = {"method": kind}
            if gid is not None:
                entry["id"] = group_of_method(gid)
            if k == 0 and isinstance(line_ids, list):
                entry["line_ids"] = [str(x) for x in line_ids]
            if destination is not None:
                entry["destination"] = destination
            if location is not None:
                entry["location_id"] = location
            if isinstance(g.get("selected_option_id"), str):
                entry["selected_option_id"] = g["selected_option_id"]
            groups.append(entry)
            if k == 0:
                first = entry.get("id")
        mapped.append(first)
    return groups, tuple(mapped), notes


def _destination(
    m: Mapping[str, Any], kind: str, at: str, notes: list[Message]
) -> tuple[dict[str, str] | None, str | None]:
    """(core address, pickup location id) for one UCP method."""
    selected = m.get("selected_destination_id") if isinstance(m.get("selected_destination_id"), str) else None
    if kind == "pickup":
        return None, selected
    dests = [d for d in m.get("destinations") or () if isinstance(d, Mapping)]
    if not dests:
        return None, None
    index = next((j for j, d in enumerate(dests) if selected is not None and d.get("id") == selected), 0)
    chosen = dests[index]
    address = {k: str(chosen[k]) for k in ADDRESS_FIELDS if isinstance(chosen.get(k), str) and chosen[k]}
    code = address.get("address_country", "")
    if not (len(code) == 2 and code.isalpha()):
        notes.append(
            error(
                "invalid_address",
                "A shipping address needs address_country as a two-letter ISO 3166-1 code",
                path=f"{at}.destinations[{index}].address_country",
            )
        )
        return None, None
    address["address_country"] = code.upper()
    if isinstance(chosen.get("id"), str) and chosen["id"]:
        address["id"] = chosen["id"]
    return address, None


@dataclass(frozen=True, slots=True)
class PaymentChoice:
    """The selected instrument of a completion, as the core's payment, and where it sits in the request."""

    payment: dict[str, Any]
    index: int

    def path(self, part: str = "") -> str:
        return f"$.payment.instruments[{self.index}]{part}"


def payment(payload: Mapping[str, Any], accepted: Sequence[str]) -> PaymentChoice | None:
    """The instrument marked `selected` (or the first one) as `{handler, instrument?, token?}`. The instrument type is
    passed on only when it is a payment method the market takes (card, cod, upi); any other type (a wallet's own name)
    is left to the credential to decide, as the PSP simulator keys on the token."""
    p = payload.get("payment")
    instruments = [
        x for x in (p.get("instruments") if isinstance(p, Mapping) else None) or () if isinstance(x, Mapping)
    ]
    if not instruments:
        return None
    named = p.get("selected_instrument_id") if isinstance(p, Mapping) else None
    index = next(
        (i for i, x in enumerate(instruments) if x.get("selected") is True or (named and x.get("id") == named)), 0
    )
    chosen = instruments[index]
    out: dict[str, Any] = {"handler": str(chosen.get("handler_id"))}
    kind = chosen.get("type")
    if isinstance(kind, str) and kind.lower() in accepted:
        out["instrument"] = kind.lower()
    credential = chosen.get("credential")
    token = credential.get("token") if isinstance(credential, Mapping) else None
    if isinstance(token, str):
        out["token"] = token
    return PaymentChoice(out, index)
