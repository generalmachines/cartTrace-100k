"""The ACP operations: create, update, retrieve, complete and cancel a checkout session, and the PSP's delegate payment,
each as core commands and one rendered reply.

Why here and not in the REST binding. The binding checks what travels in headers (API-Version, the API key, a
signature, the presence of an Idempotency-Key) and reads the body; everything that depends on the body's meaning or on
the store's state is here, so every refusal is logged the same way and every reply is rendered from state.

One log entry per call, at least. A call becomes the core command it means (create_checkout, update_checkout, a
`read`, complete_checkout, cancel_checkout, vault_credential), or a logged refusal when it is refused before
translation (a body that fails the release's schema, a card number, an unknown session, an idempotency conflict). A
completion whose body adds buyer contact facts first updates the checkout, so that is two commands.

Sessions. A checkout session is a core checkout of the caller's session: another session's id, like an id never
issued, is answered 404 without being passed to the core, so a free-text id never reaches the log.

Idempotency, as each release says. The client's key is scoped to the operation and the session id, and the body's
digest rides in the core idempotency key, so the core replays a retry of the same body (`Idempotent-Replayed: true`
where the release defines the header) and the same key with another body is answered as the release says: 409
`request_not_idempotent` up to 2026-01-30, 422 `idempotency_conflict` from 2026-04-17.

Issuer authentication. A completion that meets a 3-D Secure challenge leaves the session `authentication_required`
(`in_progress` at 2025-09-29) with the authentication metadata and a requires_3ds message pointing at the challenge
page the buyer completes. Until the buyer has, a completion is refused (400 requires_3ds, as the 2026-04-17 examples
answer), and an `authentication_result` the issuer has no record of is answered with the waiting session.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Any

from carttrace.sandbox.core.commands import MUTATING
from carttrace.sandbox.core.errors import Conflict, IllegalTransition, InvalidCommand, NotFound
from carttrace.sandbox.core.guard import find_card_number
from carttrace.sandbox.core.state import Checkout
from carttrace.sandbox.events.runtime import Outcome
from carttrace.sandbox.presented import presented
from carttrace.sandbox.surfaces.acp import requests
from carttrace.sandbox.surfaces.acp.dialect import AcpDialect
from carttrace.sandbox.surfaces.acp.render import EPOCH, AcpRenderer, Message
from carttrace.sandbox.surfaces.acp.schemas import Api, check, defines
from carttrace.sandbox.surfaces.host import SandboxHost
from carttrace.sandbox.surfaces.keys import opaque_key
from carttrace.sandbox.surfaces.problems import Problem
from carttrace.sandbox.surfaces.ucp.catalog import Catalog
from carttrace.sandbox.surfaces.ucp.messages import Fault, fault_for
from carttrace.sandbox.surfaces.values import obj, text

TOKEN = re.compile(r"^tok_[a-z0-9_]{1,60}$")
OPS: Mapping[str, tuple[Api, str] | None] = {
    "create": ("checkout", "CheckoutSessionCreateRequest"),
    "update": ("checkout", "CheckoutSessionUpdateRequest"),
    "complete": ("checkout", "CheckoutSessionCompleteRequest"),
    "cancel": ("checkout", "CancelSessionRequest"),
    "get": None,
    "delegate": ("psp", "DelegatePaymentRequest"),
}
"""Each operation and the request definition its body is checked against (none for a retrieve)."""
COMMANDS: Mapping[str, str] = {
    "create": "create_checkout",
    "update": "update_checkout",
    "complete": "complete_checkout",
    "cancel": "cancel_checkout",
    "get": "read",
    "delegate": "vault_credential",
}
"""The core command each operation becomes: what a refusal before translation is logged as."""
PAYMENT_REFUSALS = frozenset(
    {
        "payment_declined",
        "payment_authentication_failed",
        "payment_credential_invalid",
        "payment_credential_missing",
        "payment_instrument_mismatch",
        "payment_instrument_unsupported",
        "credential_used",
        "allowance_exceeded",
        "allowance_expired",
        "allowance_checkout_mismatch",
        "allowance_currency_mismatch",
        "delegated_credential_required",
    }
)
"""Core refusals of a completion that ACP reports as a declined payment, pointed at the token."""
STATE_SHOWN = frozenset({"payment_declined", "payment_authentication_failed", "payment_handler_unavailable"})
"""Refusals the rendered session already explains from state; not repeated as a second message."""


@dataclass(frozen=True, slots=True)
class AcpCall:
    """One ACP request, after the binding checked its headers and read its body."""

    op: str
    session: str
    body: Mapping[str, Any]
    target: str | None
    """The checkout session id in the path."""
    base: str
    key: str | None
    """The client's Idempotency-Key."""
    body_digest: str
    """sha256 of the body exactly as sent: what makes two requests with one key the same request."""


@dataclass(frozen=True, slots=True)
class AcpReply:
    status: int
    body: dict[str, Any]
    replayed: bool = False
    retry_after: int | None = None


class AcpOperations:
    """The ACP operations of one store at one release."""

    def __init__(self, host: SandboxHost, dialect: AcpDialect) -> None:
        self.host = host
        self.dialect = dialect
        config = host.pack.config()
        self.catalog = Catalog(host.pack)
        self.render = AcpRenderer(
            dialect,
            self.catalog,
            config.market,
            config.discount_codes,
            merchant=presented(host.pack).merchant_id,
            psp=presented(host.pack).acp_psp,
            neutral=presented(host.pack).neutral,
            coupons=config.coupon_table(),
        )

    def call(self, c: AcpCall) -> AcpReply:
        api = "psp" if c.op == "delegate" else "checkout"
        spec = OPS[c.op]
        if spec is not None and c.op == "cancel" and not defines(self.dialect.version, "checkout", spec[1]):
            spec = None
        if spec is not None and (c.body or c.op != "cancel"):
            problems = check(self.dialect.version, spec[0], spec[1], dict(c.body))
            if problems:
                return self._invalid(c, problems, api)
        card = find_card_number(dict(c.body))
        if card is not None:
            text_ = "A card number was sent; this store and its payment processor take opaque tokens only"
            return self._invalid(c, [Problem("$", "card_number_refused", text_)], api)
        conflict = self._conflict(c)
        if conflict is not None:
            return conflict
        handler = {
            "create": self._create,
            "update": self._update,
            "get": self._get,
            "complete": self._complete,
            "cancel": self._cancel,
            "delegate": self._delegate,
        }[c.op]
        try:
            return handler(c)
        except Early as early:
            return early.reply

    # -- shared steps --------------------------------------------------------------------------------------------------

    def _scope(self, c: AcpCall) -> str:
        return f"acp|{c.op}|{opaque_key(c.target or '-', c.key)}|"

    def _key(self, c: AcpCall, part: str = "") -> str | None:
        """The core idempotency key of this call: scoped to its operation and session id, with the body's digest."""
        if c.key is None:
            return None
        return f"{self._scope(c)}{part}{_body(c)}"

    def _conflict(self, c: AcpCall) -> AcpReply | None:
        """The same key already used with another body for this operation: the release's conflict answer."""
        if c.key is None:
            return None
        prefix = f"{c.session}|{self._scope(c)}"
        for k in self.host.state.idempotency:
            if k.startswith(prefix) and not k.endswith(_body(c)):
                idem = self.dialect.idempotency
                self.host.reject(
                    COMMANDS[c.op],
                    Conflict(code="idempotency_conflict", message="the key was used with a different request body"),
                )
                kind = idem.psp_conflict_type if c.op == "delegate" else idem.conflict_type
                body = self.render.error(
                    "idempotency_conflict",
                    "Idempotency-Key has already been used with a different request body",
                    kind=kind,
                )
                return AcpReply(idem.conflict_status, body)
        return None

    def _run(self, c: AcpCall, command: Mapping[str, Any], *, key: str | None) -> tuple[Outcome, Fault | None]:
        cmd = {**command, "session": c.session}
        if key is not None and cmd["type"] in MUTATING:
            cmd["idempotency_key"] = key
        out = self.host.execute(cmd)
        return out, fault_for(out.error, tier=self.host.tier(c.session))

    def _fault(self, f: Fault, api: str) -> AcpReply:
        """A protocol fault as an ACP Error, by the HTTP status the core's refusal maps to."""
        kinds = {
            429: "rate_limit_exceeded" if api == "psp" else "service_unavailable",
            500: "processing_error",
            503: "service_unavailable",
        }
        kind = kinds.get(f.status, "invalid_request")
        return AcpReply(f.status, self.render.error(f.reason, f.message, kind=kind), retry_after=f.retry_after)

    def _invalid(self, c: AcpCall, problems: Sequence[Problem], api: str) -> AcpReply:
        """A body the release refuses: logged before translation, answered 400 with the first problem, never with what
        the body held."""
        first = problems[0]
        details = tuple({"field": p.path, "message": p.content} for p in problems)
        self.host.reject(
            COMMANDS[c.op], InvalidCommand(code=first.code, message=first.content, field=first.path, details=details)
        )
        code = "invalid_card" if api == "psp" else first.code
        return AcpReply(400, self.render.error(code, first.content, param=first.path))

    def _missing(self, c: AcpCall) -> AcpReply:
        self.host.reject(
            COMMANDS[c.op], NotFound(code="checkout_not_found", message="Checkout session not found", field="id")
        )
        return AcpReply(404, self.render.error("not_found", "Checkout session not found", param="$.id"))

    def _checkout(self, c: AcpCall) -> Checkout | None:
        ck = self.host.state.checkouts.get(c.target or "")
        return ck if ck is not None and ck.session == c.session else None

    def _session(
        self, c: AcpCall, ck_id: str, notes: Sequence[Message], *, out: Outcome | None = None, status: int = 200
    ) -> AcpReply:
        """The session as the call left it, with the codes this call refused under the discount extension."""
        state = self.host.state
        ck = state.checkouts[ck_id]
        doc = self.render.session(state, ck, notes, c.base)
        rejected = self._rejected_codes(out) if out is not None and self.dialect.session.discounts else []
        if rejected:
            discounts = doc.get("discounts") or {"codes": list(ck.codes), "applied": []}
            discounts["rejected"] = rejected
            doc["discounts"] = discounts
        return AcpReply(status, doc, replayed=out.replayed if out is not None else False)

    def _rejected_codes(self, out: Outcome) -> list[Message]:
        reasons = {
            "invalid": "discount_code_invalid",
            "duplicate": "discount_code_already_applied",
            "not_combinable": "discount_code_combination_disallowed",
        }
        out_list: list[Message] = []
        for e in out.events:
            if e["type"] == "discount.rejected" and not e["sealed"]:
                reason = reasons.get(str(e["data"].get("reason")), "discount_code_invalid")
                out_list.append({"code": str(e["data"].get("code")), "reason": reason})
        return out_list

    def _events(self, out: Outcome, line_paths: Mapping[str, str]) -> list[Message]:
        """Messages for the agent-visible refusals and changes among a command's events."""
        r = self.render
        msgs: list[Message] = []
        for e in out.messages():
            d = e["data"]
            kind = e["type"]
            vid = str(d.get("variant_id") or "")
            at = line_paths.get(vid)
            if kind == "line.rejected" and d.get("reason") == "out_of_stock":
                name = self.catalog.variant_title(self.host.state, vid)
                says = f"{name} is out of stock: {d.get('requested')} requested, {d.get('available')} available"
                msgs.append(r.message("error", says, code="out_of_stock", param=at))
            elif kind == "line.rejected" and d.get("reason") == "sold_out":
                name = self.catalog.variant_title(self.host.state, vid)
                msgs.append(r.message("error", f"{name} is sold out", code="out_of_stock", param=at))
            elif kind == "line.rejected":
                says = "This item is not in the store's catalog; an item id takes a product variant id"
                msgs.append(r.message("error", says, code=self.dialect.code("not_found"), param=at))
            elif kind == "discount.rejected":
                says = f"Code '{d.get('code')}' was not applied"
                msgs.append(r.message("info", says, warning="discount_code_invalid", param="$.discounts.codes"))
            elif kind == "fulfillment.rejected" and d.get("reason") == "destination_not_shippable":
                says = f"This store does not ship to {d.get('country')}"
                msgs.append(r.message("error", says, code=self.dialect.code("region_restricted"), param=self._addr()))
            elif kind == "fulfillment.rejected":
                says = "No such fulfillment option for this session"
                msgs.append(r.message("error", says, code="invalid", param=self._option_path()))
            elif kind == "checkout.repriced":
                says = "A price changed since the session was quoted; the totals are updated"
                msgs.append(r.message("info", says, warning="price_change"))
            elif kind == "checkout.stock_lost":
                msgs.append(r.message("error", "The reserved stock lapsed and is gone", code="out_of_stock"))
        return msgs

    def _addr(self) -> str:
        f = self.dialect.fulfillment
        return "$.fulfillment_address" if f.address == "fulfillment_address" else "$.fulfillment_details.address"

    def _option_path(self) -> str:
        f = self.dialect.fulfillment
        return "$.fulfillment_option_id" if f.selection == "option_id" else "$.selected_fulfillment_options"

    def _refusal(self, err: Mapping[str, Any], token_path: str | None = None) -> Message | None:
        """A core refusal the rendered session does not already explain, as an ACP message."""
        code = str(err.get("code"))
        if code in STATE_SHOWN:
            return None
        message = str(err.get("message") or code)
        content = message[:1].upper() + message[1:]
        if code in PAYMENT_REFUSALS:
            return self.render.message("error", content, code="payment_declined", param=token_path)
        if code == "merchandise_out_of_stock":
            return self.render.message("error", content, code="out_of_stock", param="$.line_items")
        if code == "price_changed":
            return self.render.message("info", content, warning="price_change")
        if code == "checkout_not_ready":
            return self.render.message("error", content, code="missing")
        return self.render.message("error", content, code="invalid")

    def _translate(self, c: AcpCall, *, items: bool = True) -> tuple[requests.Translated, AcpReply | None]:
        t = requests.Translated()
        if items:
            lines = requests.lines(self.dialect, c.body, t)
            if lines is not None:
                t.fields["lines"] = lines
        groups = requests.fulfillment(self.dialect, c.body, t)
        if groups is not None:
            t.fields["fulfillment"] = groups
        codes = requests.codes(self.dialect, c.body)
        if codes is not None:
            t.fields["codes"] = codes
        found = requests.buyer(c.body)
        if found is not None:
            t.fields["buyer"] = found
        if t.problems:
            return t, self._invalid(c, t.problems, "checkout")
        return t, None

    # -- checkout sessions ---------------------------------------------------------------------------------------------

    def _create(self, c: AcpCall) -> AcpReply:
        t, refused = self._translate(c)
        if refused is not None:
            return refused
        currency = c.body.get("currency")
        if isinstance(currency, str) and currency.upper() != self.host.state.currency:
            problem = Problem(
                "$.currency", "invalid_currency", f"This store sells in {self.host.state.currency.lower()}"
            )
            return self._invalid(c, [problem], "checkout")
        out, fault = self._run(c, {"type": "create_checkout", **t.fields}, key=self._key(c))
        if fault is not None:
            return self._fault(fault, "checkout")
        ck_id = str(out.refs.get("checkout_id") or "")
        ck = self.host.state.checkouts.get(ck_id)
        if ck is None or ck.session != c.session:
            msgs = self._events(out, t.line_paths)
            first = next((m for m in msgs if m.get("type") == "error"), None)
            if first is not None:
                return AcpReply(
                    400, self.render.error(str(first["code"]), str(first["content"]), param=first.get("param"))
                )
            err = out.error or {}
            return AcpReply(400, self.render.error(str(err.get("code") or "invalid"), str(err.get("message") or "")))
        return self._session(c, ck.id, self._events(out, t.line_paths), out=out, status=201)

    def _update(self, c: AcpCall) -> AcpReply:
        ck = self._checkout(c)
        if ck is None:
            return self._missing(c)
        t, refused = self._translate(c)
        if refused is not None:
            return refused
        out, fault = self._run(c, {"type": "update_checkout", "checkout_id": ck.id, **t.fields}, key=self._key(c))
        if fault is not None:
            return self._fault(fault, "checkout")
        notes = self._events(out, t.line_paths)
        if out.error is not None:
            msg = self._refusal(out.error)
            if msg is not None:
                notes.append(msg)
        return self._session(c, ck.id, notes, out=out)

    def _get(self, c: AcpCall) -> AcpReply:
        ck = self._checkout(c)
        if ck is None:
            return self._missing(c)
        _, fault = self._run(c, {"type": "read", "target": "checkout", "id": ck.id}, key=None)
        if fault is not None:
            return self._fault(fault, "checkout")
        return self._session(c, ck.id, [])

    def _complete(self, c: AcpCall) -> AcpReply:
        ck = self._checkout(c)
        if ck is None:
            return self._missing(c)
        if ck.status == "requires_escalation" and ck.escalation == "payment_action_required":
            return self._authentication_pending(c, ck)
        self._add_contact(c, ck)
        choice = requests.payment(self.dialect, c.body)
        if not choice.token:
            says = "This store takes card payments through its payment handler; purchase orders are not accepted"
            return self._invalid(c, [Problem("$.payment_data", "unsupported", says)], "checkout")
        payment: dict[str, Any] = {"handler": choice.handler, "token": choice.token}
        if choice.instrument is not None:
            payment["instrument"] = choice.instrument
        command = {"type": "complete_checkout", "checkout_id": ck.id, "payment": payment}
        out, fault = self._run(c, command, key=self._key(c))
        if fault is not None:
            return self._fault(fault, "checkout")
        return self._completed(c, ck.id, choice, out)

    def _add_contact(self, c: AcpCall, ck: Checkout) -> None:
        """Buyer contact facts a completion brings that the checkout lacks: an update before the payment."""
        contact = requests.buyer(c.body)
        if contact is None or all(ck.buyer.get(k) == v for k, v in contact.items()):
            return
        command = {"type": "update_checkout", "checkout_id": ck.id, "buyer": contact}
        _, fault = self._run(c, command, key=self._key(c, "buyer|"))
        if fault is not None:
            raise Early(self._fault(fault, "checkout"))

    def _completed(self, c: AcpCall, ck_id: str, choice: requests.PaymentChoice, out: Outcome) -> AcpReply:
        """A completion's reply: the session, with what refused the payment when it was refused."""
        err = out.error
        if err is not None and err.get("kind") == "validation":
            param = choice.token_path if str(err.get("field") or "").startswith("payment") else None
            return AcpReply(400, self.render.error("invalid_payment", str(err.get("message") or ""), param=param))
        notes: list[Message] = []
        if err is not None and self.host.state.checkouts[ck_id].status == "completed":
            notes.append(self.render.message("info", "This checkout session is already completed"))
        elif err is not None:
            msg = self._refusal(err, choice.token_path)
            if msg is not None:
                notes.append(msg)
        return self._session(c, ck_id, notes, out=out)

    def _authentication_pending(self, c: AcpCall, ck: Checkout) -> AcpReply:
        """The issuer still waits for the buyer's 3-D Secure step: nothing changes until the buyer completes it."""
        pay = self.host.state.payments.get(ck.payment_id or "")
        page = f"{c.base}/3ds/{pay.challenge_id}" if pay is not None else f"{c.base}/checkout/{ck.id}"
        text_ = f"The buyer's bank has not seen the 3-D Secure challenge completed; the buyer completes it at {page}"
        self.host.reject(
            COMMANDS[c.op], IllegalTransition(code="authentication_pending", message=text_, field="status")
        )
        if "authentication_result" in c.body:
            note = self.render.message(
                "error", text_, code="requires_3ds", param="$.authentication_result", resolution="requires_buyer_input"
            )
            return self._session(c, ck.id, [note])
        return AcpReply(400, self.render.error("requires_3ds", text_, param="$.authentication_result"))

    def _cancel(self, c: AcpCall) -> AcpReply:
        ck = self._checkout(c)
        if ck is None:
            return self._missing(c)
        if ck.status in ("completed", "canceled"):
            text_ = f"The checkout session is {ck.status} and cannot be canceled"
            self.host.reject(COMMANDS[c.op], IllegalTransition(code="checkout_locked", message=text_, field="status"))
            return AcpReply(405, self.render.error("not_cancelable", text_))
        out, fault = self._run(c, {"type": "cancel_checkout", "checkout_id": ck.id}, key=self._key(c))
        if fault is not None:
            return self._fault(fault, "checkout")
        return self._session(c, ck.id, [], out=out)

    # -- the PSP: delegate payment -------------------------------------------------------------------------------------

    def _delegate(self, c: AcpCall) -> AcpReply:
        method = obj(c.body.get("payment_method"))
        allowance = obj(c.body.get("allowance"))
        command: dict[str, Any] = {
            "type": "vault_credential",
            "token": text(method.get("number")),
            "max_amount": allowance.get("max_amount"),
            "currency": text(allowance.get("currency")).upper(),
            "checkout_id": self._allowance_checkout(c, method, allowance),
            "expires_at": self._allowance_expiry(c, allowance),
            "merchant": text(allowance.get("merchant_id")) or None,
        }
        out, fault = self._run(c, command, key=self._key(c))
        if fault is not None:
            return self._fault(fault, "psp")
        if out.error is not None:
            param = PSP_FIELDS.get(str(out.error.get("field") or ""))
            status = 400 if out.error.get("kind") == "validation" else 422
            return AcpReply(status, self.render.error("invalid_card", str(out.error.get("message")), param=param))
        vid = str(out.refs["credential_id"])
        cred = self.host.state.credentials[vid]
        metadata = {"source": presented(self.host.pack).vault_source, "merchant_id": text(allowance.get("merchant_id"))}
        if c.key is not None:
            metadata["idempotency_key"] = c.key
        return AcpReply(201, self.render.vault_token(vid, cred.created_at, metadata), replayed=out.replayed)

    def _allowance_checkout(self, c: AcpCall, method: Mapping[str, Any], allowance: Mapping[str, Any]) -> str:
        """The checkout a delegation's allowance names, once its credential is one this PSP vaults: an opaque test
        token, sent as a network token. Raises `Early` with the refusal otherwise."""
        if method.get("card_number_type") != "network_token":
            says = "This payment processor vaults network tokens only: send card_number_type network_token"
            raise Early(self._invalid(c, [Problem("$.payment_method.card_number_type", "invalid_card", says)], "psp"))
        if not TOKEN.match(text(method.get("number"))):
            says = "number takes the network token the shopper's wallet holds"
            raise Early(self._invalid(c, [Problem("$.payment_method.number", "invalid_card", says)], "psp"))
        target = self.host.state.checkouts.get(text(allowance.get("checkout_session_id")))
        if target is None or target.session != c.session:
            says = "The allowance names no checkout session of this store"
            raise Early(self._invalid(c, [Problem("$.allowance.checkout_session_id", "invalid_card", says)], "psp"))
        return target.id

    def _allowance_expiry(self, c: AcpCall, allowance: Mapping[str, Any]) -> int:
        expires = _epoch_ms(text(allowance.get("expires_at")))
        if expires is None:
            says = "expires_at takes an RFC 3339 instant with a time zone"
            raise Early(self._invalid(c, [Problem("$.allowance.expires_at", "invalid_card", says)], "psp"))
        return expires


class Early(Exception):
    """A reply decided part way through an operation (a refusal); `AcpOperations.call` sends it."""

    def __init__(self, reply: AcpReply) -> None:
        super().__init__(reply.status)
        self.reply = reply


PSP_FIELDS: Mapping[str, str] = {
    "token": "$.payment_method.number",
    "expires_at": "$.allowance.expires_at",
    "max_amount": "$.allowance.max_amount",
    "currency": "$.allowance.currency",
    "merchant": "$.allowance.merchant_id",
}
"""Where a core refusal of `vault_credential` points in a delegate payment request."""


def _body(c: AcpCall) -> str:
    """The body's part of a core idempotency key: two requests are the same request when their bodies are."""
    return opaque_key(c.body_digest, length=16)


def _epoch_ms(value: str) -> int | None:
    """An RFC 3339 instant as epoch milliseconds, or None when it is not one (a time zone is required)."""
    try:
        when = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None
    if when.tzinfo is None:
        return None
    return (when - EPOCH) // timedelta(milliseconds=1)
