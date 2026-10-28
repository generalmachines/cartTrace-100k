"""Core state to ACP documents in one release's dialect: checkout sessions, errors, and the PSP's vault token reply.

Why from state. A session is rendered from the state the command left, never from the command, so a retrieve after an
update, an idempotent replay and a session after a fault all show the same store. Messages come from the command's
own events and refusal (passed in by the operation) and from facts the state shows whatever the command was (what an
incomplete session still lacks, a last payment that failed, an issuer waiting on the buyer).

Money. Amounts are integers in minor units, as ACP types them. A release whose line items carry their own discount
and tax (up to 2026-01-16) gets the session's discount and tax split across lines by base amount, in integers, largest
remainders first, so the lines add up to the session exactly.

What each release changes (`AcpDialect`) is read here: line item shape, fulfillment option amounts, how the address and
the selected option are named, payment provider or payment handlers, the status map, message codes, links and totals.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import UTC, datetime, timedelta
from typing import Any

from carttrace.core.money import to_major
from carttrace.sandbox.core import pricing
from carttrace.sandbox.core.config import Coupon
from carttrace.sandbox.core.market import Market, countries_served
from carttrace.sandbox.core.state import Checkout, Group, Line, Payment, State, Totals
from carttrace.sandbox.surfaces.acp.dialect import AcpDialect
from carttrace.sandbox.surfaces.acp.requests import HANDLER_ID
from carttrace.sandbox.surfaces.ucp.catalog import Catalog

Message = dict[str, Any]
EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
PSP = "sandbox"
BRANDS = ("visa", "mastercard", "amex")
LINK_PAGES: Mapping[str, tuple[str, str]] = {
    "terms_of_use": ("terms-of-service", "Terms of service"),
    "privacy_policy": ("privacy-policy", "Privacy policy"),
    "return_policy": ("refund-policy", "Refund policy"),
    "seller_shop_policies": ("refund-policy", "Shop policies"),
    "shipping_policy": ("shipping-policy", "Shipping policy"),
}
"""The policy pages the sandbox serves, by the ACP link type that points at them."""
DIRECTORY = {"visa": "visa", "mastercard": "mastercard", "amex": "american_express"}
FAILED = {
    "declined": "The payment was declined",
    "authentication_failed": "The buyer's bank did not authenticate the payment",
}


def iso(ms: int) -> str:
    """Logical milliseconds as RFC 3339, millisecond precision, Z."""
    return (EPOCH + timedelta(milliseconds=ms)).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def allocate(total: int, weights: Sequence[int]) -> list[int]:
    """`total` split across `weights` in integers that sum to it: floor shares, then the largest remainders, earlier
    index first on ties."""
    whole = sum(weights)
    if not weights:
        return []
    if whole <= 0:
        return [total if i == 0 else 0 for i in range(len(weights))]
    shares = [total * w // whole for w in weights]
    order = sorted(range(len(weights)), key=lambda i: (-(total * weights[i] % whole), i))
    for i in order[: total - sum(shares)]:
        shares[i] += 1
    return shares


class AcpRenderer:
    """Renders one store's sessions at one ACP release."""

    def __init__(
        self,
        dialect: AcpDialect,
        catalog: Catalog,
        market: Market,
        codes: Mapping[str, int],
        *,
        merchant: str,
        coupons: Mapping[str, Coupon] | None = None,
        psp: str = PSP,
        neutral: bool = False,
    ) -> None:
        """`codes` is the pack's discount table (code to percent off), for each applied code's share; `coupons` the
        codes' rules where the pack has more than percentages."""
        self.dialect = dialect
        self.catalog = catalog
        self.market = market
        self.codes = codes
        self.coupons: Mapping[str, Coupon] = (
            coupons if coupons is not None else {c: Coupon(c, "percent", int(p)) for c, p in codes.items()}
        )
        self.merchant = merchant
        """The merchant id the payment provider or handler names: the store's presented merchant id (D106)."""
        self.psp = psp
        """The PSP the handler names: the store's presented one (D106), `sandbox` for every other world."""
        self.card_name = "Card" if neutral else "Card (sandbox)"

    # -- messages ------------------------------------------------------------------------------------------------------

    def message(
        self,
        kind: str,
        content: str,
        *,
        code: str | None = None,
        param: str | None = None,
        resolution: str = "recoverable",
        warning: str | None = None,
    ) -> Message:
        """An ACP message: `kind` is error or info; `warning` is the warning code to use where the release has
        warnings (2026-01-30 and later), else the message stays `kind`."""
        d = self.dialect.messages
        out: Message = {"type": kind, "content_type": "plain", "content": content}
        if warning is not None and d.warnings:
            out = {"type": "warning", "code": warning, "content_type": "plain", "content": content}
        elif kind == "error":
            out["code"] = self.dialect.code(code or "invalid")
        if param is not None:
            out["param"] = param
        if d.severity:
            out["severity"] = "high" if out["type"] == "error" else "info"
        if d.resolution and out["type"] != "info":
            out["resolution"] = resolution
        return out

    # -- sessions ------------------------------------------------------------------------------------------------------

    def session(self, state: State, ck: Checkout, messages: Sequence[Message], base: str) -> dict[str, Any]:
        d = self.dialect
        pay = state.payments.get(ck.payment_id) if ck.payment_id else None
        status = d.status_of(ck.status, ck.escalation)
        doc: dict[str, Any] = {
            "id": ck.id,
            "status": status,
            "currency": state.currency.lower(),
            "line_items": self.line_items(state, ck.lines, ck.totals),
            "fulfillment_options": self.options(ck.groups),
            "totals": self.totals(ck.totals),
            "links": self.links(base),
        }
        if d.session.capabilities:
            doc["protocol"] = {"version": d.version}
            doc["capabilities"] = self.capabilities(base)
        elif d.payment.data == "provider":
            doc["payment_provider"] = self.provider()
        if d.buyer == "email" and ck.buyer.get("email"):
            doc["buyer"] = {k: ck.buyer[k] for k in ("email", "phone_number") if ck.buyer.get(k)}
        self._fulfillment(doc, ck)
        notes = self._notes(state, ck, pay, base)
        challenge = _pending_challenge(ck, pay)
        if challenge is not None and d.session.authentication:
            doc["authentication_metadata"] = self.authentication(pay)
        if d.session.continue_url and status in ("requires_escalation", "authentication_required"):
            doc["continue_url"] = f"{base}/3ds/{challenge}" if challenge else f"{base}/checkout/{ck.id}"
        if d.session.created_at:
            doc["created_at"] = iso(ck.created_at)
        if d.session.discounts and ck.codes:
            doc["discounts"] = self.discounts(ck.codes, ck.totals.discount, (), ck.totals.subtotal)
        if ck.status == "completed" and ck.order_id is not None:
            doc["order"] = self.order(state, ck, base)
        doc["messages"] = _unique([*messages, *notes])
        return doc

    def line_items(self, state: State, lines: Sequence[Line], totals: Totals) -> list[dict[str, Any]]:
        d = self.dialect
        bases = [ln.unit_price * ln.quantity for ln in lines]
        discounts = allocate(totals.discount, bases)
        nets = [b - x for b, x in zip(bases, discounts, strict=True)]
        taxes = allocate(totals.tax or 0, nets)
        out = []
        for ln, base_amount, disc, net, tax in zip(lines, bases, discounts, nets, taxes, strict=True):
            entry: dict[str, Any] = {"id": ln.id}
            if d.items.quantity:
                entry["item"] = {"id": ln.variant_id, "quantity": ln.quantity}
            else:
                entry["item"] = {"id": ln.variant_id}
                entry["quantity"] = ln.quantity
            if d.line_items.details:
                entry["name"] = self.catalog.variant_title(state, ln.variant_id)
                entry["unit_amount"] = ln.unit_price
            if d.line_items.shape == "amounts":
                entry.update(base_amount=base_amount, discount=disc, subtotal=net, tax=tax, total=net + tax)
            else:
                entry["totals"] = self._money_rows(
                    [("items_base_amount", "Base amount", base_amount), ("discount", "Discount", disc)],
                    net,
                    tax if totals.tax is not None else None,
                )
            out.append(entry)
        return out

    def _money_rows(self, head: list[tuple[str, str, int]], subtotal: int, tax: int | None) -> list[dict[str, Any]]:
        rows = [{"type": t, "display_text": text, "amount": a} for t, text, a in head if a or t != "discount"]
        rows.append({"type": "subtotal", "display_text": "Subtotal", "amount": subtotal})
        if tax is not None:
            rows.append({"type": "tax", "display_text": self._tax_text(), "amount": tax})
        rows.append({"type": "total", "display_text": "Total", "amount": subtotal + (tax or 0)})
        return rows

    def _tax_text(self) -> str:
        return "Tax included in prices" if self.market.tax_inclusive else "Tax"

    def totals(self, t: Totals) -> list[dict[str, Any]]:
        """ACP totals: items_base_amount, discount (when there is one), subtotal after it, fulfillment and tax once
        known, total."""
        rows: list[dict[str, Any]] = [
            {"type": "items_base_amount", "display_text": "Item(s) total", "amount": t.subtotal}
        ]
        if t.discount:
            rows.append({"type": "discount", "display_text": "Discount", "amount": t.discount})
        rows.append({"type": "subtotal", "display_text": "Subtotal", "amount": t.subtotal - t.discount})
        if t.fulfillment is not None:
            rows.append({"type": "fulfillment", "display_text": "Shipping", "amount": t.fulfillment})
        if t.tax is not None:
            rows.append({"type": "tax", "display_text": self._tax_text(), "amount": t.tax})
        rows.append({"type": "total", "display_text": "Total", "amount": t.total})
        return rows

    def options(self, groups: Sequence[Group]) -> list[dict[str, Any]]:
        """The shipping options of the session's group, priced as the core priced them."""
        shape = self.dialect.fulfillment.option_amounts
        out = []
        for g in groups:
            if g.method != "shipping":
                continue
            for o in g.options:
                amount = int(o["amount"])
                entry: dict[str, Any] = {"type": "shipping", "id": str(o["id"]), "title": str(o["title"])}
                if o.get("carrier"):
                    entry["carrier"] = str(o["carrier"])
                if shape == "totals":
                    entry["totals"] = [{"type": "total", "display_text": "Shipping", "amount": amount}]
                else:
                    entry.update(subtotal=amount, tax=0, total=amount)
                out.append(entry)
        return out

    def _fulfillment(self, doc: dict[str, Any], ck: Checkout) -> None:
        """The address and the selected option, under the names the release uses."""
        d = self.dialect.fulfillment
        group = next((g for g in ck.groups if g.method == "shipping"), None)
        address = _address(group.destination) if group is not None and group.destination else None
        if d.address == "fulfillment_address":
            if address is not None:
                doc["fulfillment_address"] = address
        else:
            details: dict[str, Any] = {}
            if address is not None:
                details["name"] = address["name"]
                details["address"] = address
            for key in ("email", "phone_number"):
                if ck.buyer.get(key):
                    details[key] = ck.buyer[key]
            if details:
                doc["fulfillment_details"] = details
        if group is None or group.selected_option_id is None:
            return
        items = [ln.variant_id for ln in ck.lines if ln.id in group.line_ids]
        if d.selection == "option_id":
            doc["fulfillment_option_id"] = group.selected_option_id
        elif d.selection == "nested":
            doc["selected_fulfillment_options"] = [
                {"type": "shipping", "shipping": {"option_id": group.selected_option_id, "item_ids": items}}
            ]
        else:
            doc["selected_fulfillment_options"] = [
                {"type": "shipping", "option_id": group.selected_option_id, "item_ids": items}
            ]

    def links(self, base: str) -> list[dict[str, Any]]:
        out = []
        for kind in self.dialect.links:
            page = LINK_PAGES.get(kind)
            if page is None:
                continue
            link: dict[str, Any] = {"type": kind, "url": f"{base}/policies/{page[0]}"}
            if self.dialect.link_titles:
                link["title"] = page[1]
            out.append(link)
        return out

    def provider(self) -> dict[str, Any]:
        """The payment provider (up to 2026-01-16). ACP's enum names only `stripe`; the sandbox's PSP answers under
        that name and calls nothing."""
        p = self.dialect.payment
        methods: list[Any] = (
            ["card"] if p.methods == "strings" else [{"type": "card", "supported_card_networks": list(BRANDS)}]
        )
        out: dict[str, Any] = {"provider": "stripe", "supported_payment_methods": methods}
        if p.merchant_id:
            out["merchant_id"] = self.merchant
        return out

    def handler(self, base: str) -> dict[str, Any]:
        """The sandbox's one ACP payment handler (2026-01-30 and later): a tokenized card that needs a delegated
        payment token from the sandbox's PSP."""
        return {
            "id": HANDLER_ID,
            "name": "dev.acp.tokenized.card",
            "display_name": self.card_name,
            "version": self.dialect.version,
            "spec": "https://acp.dev/handlers/tokenized.card",
            "requires_delegate_payment": True,
            "requires_pci_compliance": False,
            "psp": self.psp,
            "config_schema": "https://acp.dev/schemas/handlers/tokenized.card/config.json",
            "instrument_schemas": ["https://acp.dev/schemas/handlers/tokenized.card/instrument.json"],
            "config": {
                "merchant_id": self.merchant,
                "psp": self.psp,
                "accepted_brands": list(BRANDS),
                "environment": "test",
                "delegate_payment_url": f"{base}/psp/agentic_commerce/delegate_payment",
            },
        }

    def capabilities(self, base: str) -> dict[str, Any]:
        handler = self.handler(base)
        if not self.dialect.payment.display_name:
            handler.pop("display_name")
        return {
            "payment": {"handlers": [handler]},
            "interventions": {"supported": ["3ds"], "enforcement": "conditional"},
        }

    def authentication(self, pay: Payment | None) -> dict[str, Any]:
        """What the buyer's bank needs for 3-D Secure, as the store's acquirer would send it; the challenge itself is
        the sandbox's page at `continue_url` (or in the requires_3ds message)."""
        out: dict[str, Any] = {
            "acquirer_details": {
                "acquirer_bin": "000000",
                "acquirer_country": countries_served(self.market.code)[0],
                "acquirer_merchant_id": self.merchant[:35],
                "merchant_name": self.merchant[:40],
            },
            "directory_server": DIRECTORY.get((pay.brand if pay else None) or "visa", "visa"),
            "flow_preference": {"type": "challenge"},
        }
        if self.dialect.session.authentication_channel:
            out["channel"] = {
                "type": "browser",
                "browser": {
                    "accept_header": "text/html",
                    "ip_address": "127.0.0.1",
                    "javascript_enabled": False,
                    "language": "en-US",
                    "user_agent": "carttrace sandbox challenge page",
                },
            }
        return out

    def discounts(
        self, codes: Sequence[str], discount: int, rejected: Sequence[Message], subtotal: int = 0
    ) -> dict[str, Any]:
        """The discount extension (2026-01-30 and later): applied codes, each with its share of the discount split
        by its percentage (or, beside a fixed or free-shipping coupon, each code's own amount, D42), and the codes
        this call refused."""
        rules = [self.coupons.get(c, Coupon(c, "percent", int(self.codes.get(c, 0)))) for c in codes]
        if all(r.kind == "percent" for r in rules):
            shares = allocate(discount, [r.amount for r in rules])
        else:
            shares = pricing.code_amounts(subtotal, codes, self.coupons)
        applied = [
            {"id": f"disc-{code.lower()}", "code": code, "coupon": _coupon(rule, self.market.currency), "amount": share}
            for code, rule, share in zip(codes, rules, shares, strict=True)
        ]
        out: dict[str, Any] = {"codes": list(codes), "applied": applied}
        if rejected:
            out["rejected"] = list(rejected)
        return out

    def order(self, state: State, ck: Checkout, base: str) -> dict[str, Any]:
        oid = str(ck.order_id)
        out: dict[str, Any] = {"id": oid, "checkout_session_id": ck.id, "permalink_url": f"{base}/orders/{oid}"}
        o = state.orders.get(oid)
        if self.dialect.session.order_number and o is not None:
            out["order_number"] = f"#{o.number}"
            out["status"] = "confirmed"
        if self.dialect.session.order_type:
            out["type"] = "order"
        return out

    def _notes(self, state: State, ck: Checkout, pay: Payment | None, base: str) -> list[Message]:
        """Facts the state shows whatever the call was: what is missing, a pending challenge, a failed payment."""
        notes: list[Message] = []
        challenge = _pending_challenge(ck, pay)
        if challenge is not None:
            says = f"The buyer's bank asks for a 3-D Secure challenge; the buyer completes it at {base}/3ds/{challenge}"
            notes.append(self.message("error", says, code="requires_3ds", resolution="requires_buyer_input"))
        elif ck.status == "requires_escalation" and ck.escalation == "payment_handler_unavailable":
            notes.append(
                self.message(
                    "error",
                    "The payment handler named is not served here; pay with the handler this session lists",
                    code=self.dialect.code("unsupported"),
                    param="$.payment_data",
                    resolution="requires_buyer_input",
                )
            )
        if ck.status == "incomplete":
            notes += self._missing(state, ck)
        if pay is not None and ck.status == "ready_for_complete" and pay.status in FAILED:
            # The issuer's reason, as UCP's message gives it, so a buyer can tell a decline a retry may clear from one
            # only they can fix (D66).
            why = f" ({pay.decline_code})" if pay.status == "declined" and pay.decline_code else ""
            text = FAILED[pay.status] + why
            notes.append(self.message("error", text, code="payment_declined", param="$.payment_data"))
        return notes

    def _missing(self, state: State, ck: Checkout) -> list[Message]:
        d = self.dialect
        out: list[Message] = []
        if not ck.buyer.get("email"):
            out.append(self.message("error", "Buyer email is required", code="missing", param="$.buyer.email"))
        group = next((g for g in ck.groups if g.method == "shipping"), None)
        address_path = (
            "$.fulfillment_address"
            if d.fulfillment.address == "fulfillment_address"
            else ("$.fulfillment_details.address")
        )
        if group is None or not group.destination:
            out.append(self.message("error", "A shipping address is required", code="missing", param=address_path))
        elif group.amount is None:
            option_path = (
                "$.fulfillment_option_id"
                if d.fulfillment.selection == "option_id"
                else "$.selected_fulfillment_options"
            )
            out.append(self.message("error", "Select a fulfillment option", code="missing", param=option_path))
        res = state.reservations.get(ck.reservation_id) if ck.reservation_id else None
        if ck.lines and (res is None or res.status != "held"):
            out.append(
                self.message(
                    "error", "The stock for this session could not be held", code="out_of_stock", param="$.line_items"
                )
            )
        return out

    # -- errors and the PSP --------------------------------------------------------------------------------------------

    def error(
        self,
        code: str,
        message: str,
        *,
        kind: str = "invalid_request",
        param: str | None = None,
        versions: bool = False,
    ) -> dict[str, Any]:
        """An ACP `Error`: no session to show. `versions` lists the API-Versions served where the release has the
        field (2026-04-17)."""
        out: dict[str, Any] = {"type": kind, "code": code, "message": message}
        if param is not None:
            out["param"] = param
        if versions and self.dialect.errors.supported_versions:
            out["supported_versions"] = [self.dialect.version]
        return out

    def vault_token(self, credential_id: str, created_at: int, metadata: Mapping[str, str]) -> dict[str, Any]:
        """The PSP's reply to delegate_payment: the vault token, when it was made, and string metadata."""
        return {"id": credential_id, "created": iso(created_at), "metadata": dict(metadata)}


def _address(dest: Mapping[str, str]) -> dict[str, str]:
    """A core address as an ACP Address (every required member present; a missing one is an empty string)."""
    name = " ".join(x for x in (dest.get("first_name"), dest.get("last_name")) if x)
    out = {
        "name": name,
        "line_one": dest.get("street_address", ""),
        "city": dest.get("address_locality", ""),
        "state": dest.get("address_region", ""),
        "country": dest.get("address_country", ""),
        "postal_code": dest.get("postal_code", ""),
    }
    if dest.get("extended_address"):
        out["line_two"] = dest["extended_address"]
    return out


def _pending_challenge(ck: Checkout, pay: Payment | None) -> str | None:
    waiting = ck.status == "requires_escalation" and ck.escalation == "payment_action_required"
    return pay.challenge_id if waiting and pay is not None and pay.status == "requires_action" else None


def _unique(messages: Sequence[Message]) -> list[Message]:
    seen: set[tuple[Any, ...]] = set()
    out: list[Message] = []
    for m in messages:
        key = (m.get("type"), m.get("code"), m.get("param"), m.get("content"))
        if key not in seen:
            seen.add(key)
            out.append(m)
    return out


def _coupon(rule: Coupon, currency: str) -> dict[str, Any]:
    """An applied code's coupon as the discount extension writes it: percent_off, or amount_off for a fixed one."""
    if rule.kind == "percent":
        return {"id": rule.code, "name": f"{rule.code} ({rule.amount}% off)", "percent_off": rule.amount}
    if rule.kind == "fixed_cart":
        name = f"{rule.code} ({to_major(rule.amount, currency)} off)"
        return {"id": rule.code, "name": name, "amount_off": rule.amount, "currency": currency.lower()}
    return {"id": rule.code, "name": f"{rule.code} (free shipping)", "amount_off": 0, "currency": currency.lower()}
