"""ACP request bodies to core command fields, in one release's terms.

Why a translation. The core speaks protocol-neutral commands (`carttrace.sandbox.core.commands`); an ACP body says the same
things in its release's shape. An ACP item id is the core's variant id. Quantities are whole units: 2025-09-29 types
them as JSON numbers, so a fractional quantity is refused here; from 2026-01-30 an item carries no quantity at all, so
each entry is one unit and a repeated id is one more. The fulfillment address (a `fulfillment_address`, or the address
inside `fulfillment_details`) becomes the core's one shipping group, and the selected fulfillment option (an
`fulfillment_option_id`, or `selected_fulfillment_options` nested per type or flat) its chosen option. The buyer's
names are not kept: the core keeps contact facts only. Bodies reach here only after they passed the release's schema,
so this module handles meaning, not types; every translation is a pure function of the body, so a retried request
becomes the same command and its idempotency key replays.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from carttrace.sandbox.core.config import DELEGATED_NAMESPACE
from carttrace.sandbox.surfaces.acp.dialect import AcpDialect
from carttrace.sandbox.surfaces.problems import Problem
from carttrace.sandbox.surfaces.values import obj, text

GROUP_ID = "g1"
"""The core fulfillment group an ACP session ships in: ACP has one fulfillment address per session."""
HANDLER_ID = "card_tokenized"
"""The id of the sandbox's ACP payment handler (2026-01-30 and later), as the ACP examples name a tokenized card."""


@dataclass(slots=True)
class Translated:
    """Core command fields for one request, the request path of each line, and what could not be used."""

    fields: dict[str, Any] = field(default_factory=dict)
    line_paths: dict[str, str] = field(default_factory=dict)
    """Variant id to the JSONPath of its first entry in the request."""
    problems: list[Problem] = field(default_factory=list)


def lines(dialect: AcpDialect, body: Mapping[str, Any], out: Translated) -> list[dict[str, Any]] | None:
    """The core lines a body asks for (a full replacement), or None when it does not mention items."""
    raw = body.get(dialect.items.field)
    if not isinstance(raw, list):
        return None
    want: dict[str, int] = {}
    for i, entry in enumerate(raw):
        item = obj(entry)
        vid = text(item.get("id"))
        at = f"$.{dialect.items.field}[{i}]"
        if dialect.items.quantity:
            qty = item.get("quantity")
            if isinstance(qty, float) and qty.is_integer():
                qty = int(qty)
            if not isinstance(qty, int) or isinstance(qty, bool) or qty < 1:
                out.problems.append(Problem(f"{at}.quantity", "invalid", "Quantities are whole units of at least 1"))
                continue
        else:
            qty = 1
        out.line_paths.setdefault(vid, at)
        want[vid] = want.get(vid, 0) + qty
    return [{"variant_id": vid, "quantity": q} for vid, q in want.items()]


def buyer(body: Mapping[str, Any]) -> dict[str, str] | None:
    """The contact facts of an ACP buyer (names are never kept), with the fulfillment contact's when given."""
    b = obj(body.get("buyer"))
    details = obj(body.get("fulfillment_details"))
    out: dict[str, str] = {}
    for key in ("email", "phone_number"):
        value = b.get(key) if isinstance(b.get(key), str) else details.get(key)
        if isinstance(value, str) and value:
            out[key] = value
    return out or None


def address(dialect: AcpDialect, body: Mapping[str, Any], out: Translated) -> dict[str, str] | None:
    """The core shipping address an ACP body gives, or None when it gives none."""
    if dialect.fulfillment.address == "fulfillment_address":
        raw, at = obj(body.get("fulfillment_address")), "$.fulfillment_address"
    else:
        raw, at = obj(obj(body.get("fulfillment_details")).get("address")), "$.fulfillment_details.address"
    if not raw:
        return None
    country = text(raw.get("country")).strip()
    if not (len(country) == 2 and country.isalpha()):
        out.problems.append(Problem(f"{at}.country", "invalid", "country takes an ISO 3166-1 alpha-2 code"))
        return None
    core = {
        "first_name": text(raw.get("name")),
        "street_address": text(raw.get("line_one")),
        "extended_address": text(raw.get("line_two")),
        "address_locality": text(raw.get("city")),
        "address_region": text(raw.get("state")),
        "postal_code": text(raw.get("postal_code")),
        "address_country": country.upper(),
    }
    return {k: v for k, v in core.items() if v}


def option(dialect: AcpDialect, body: Mapping[str, Any], out: Translated) -> str | None:
    """The fulfillment option a body selects, or None when it selects none. The sandbox ships every item together,
    so selections that name different options are refused, as are digital, pickup and local delivery."""
    if dialect.fulfillment.selection == "option_id":
        chosen = body.get("fulfillment_option_id")
        return chosen if isinstance(chosen, str) else None
    picks = [obj(p) for p in body.get("selected_fulfillment_options") or () if isinstance(p, Mapping)]
    if not picks:
        return None
    ids: list[str] = []
    for i, p in enumerate(picks):
        kind = text(p.get("type"))
        at = f"$.selected_fulfillment_options[{i}]"
        if kind != "shipping":
            out.problems.append(Problem(f"{at}.type", "unsupported", "This store fulfils by shipping only"))
            continue
        chosen = obj(p.get("shipping")).get("option_id") if dialect.fulfillment.selection == "nested" else None
        chosen = chosen if isinstance(chosen, str) else p.get("option_id")
        if isinstance(chosen, str):
            ids.append(chosen)
    if len(set(ids)) > 1:
        out.problems.append(
            Problem(
                "$.selected_fulfillment_options",
                "unsupported",
                "This store ships every item together: select one shipping option for all items",
            )
        )
        return None
    return ids[0] if ids else None


def fulfillment(dialect: AcpDialect, body: Mapping[str, Any], out: Translated) -> list[dict[str, Any]] | None:
    """The core fulfillment update a body makes (the session's one shipping group), or None when it makes none."""
    dest = address(dialect, body, out)
    chosen = option(dialect, body, out)
    if dest is None and chosen is None:
        return None
    group: dict[str, Any] = {"id": GROUP_ID, "method": "shipping"}
    if dest is not None:
        group["destination"] = dest
    if chosen is not None:
        group["selected_option_id"] = chosen
    return [group]


def codes(dialect: AcpDialect, body: Mapping[str, Any]) -> list[str] | None:
    """Discount codes (2026-01-30 and later: `discounts.codes`, or the older `coupons`), or None when the body sends
    none. Codes replace the applied ones, as UCP's do."""
    if not dialect.session.discounts:
        return None
    sent = obj(body.get("discounts")).get("codes")
    if not isinstance(sent, list):
        sent = body.get("coupons")
    return [str(c) for c in sent if isinstance(c, str)] if isinstance(sent, list) else None


@dataclass(frozen=True, slots=True)
class PaymentChoice:
    """A completion's payment as the core takes it, and where its token sits in the request."""

    handler: str
    token: str
    token_path: str
    instrument: str | None


def payment(dialect: AcpDialect, body: Mapping[str, Any]) -> PaymentChoice:
    """The payment a completion presents. The sandbox's ACP handler is the PSP's delegated handler: it charges only a
    token its delegate payment endpoint issued. A handler id the session never listed is passed on as given, so the
    core answers it as a handler this store does not serve."""
    data = obj(body.get("payment_data"))
    if dialect.payment.data == "provider":
        return PaymentChoice(DELEGATED_NAMESPACE, text(data.get("token")), "$.payment_data.token", None)
    instrument = obj(data.get("instrument"))
    handler = text(data.get("handler_id"))
    kind = text(instrument.get("type")).lower()
    return PaymentChoice(
        DELEGATED_NAMESPACE if handler == HANDLER_ID else handler,
        text(obj(instrument.get("credential")).get("token")),
        "$.payment_data.instrument.credential.token",
        "card" if kind == "card" else None,
    )
