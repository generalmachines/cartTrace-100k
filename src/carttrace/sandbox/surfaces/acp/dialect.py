"""Per-release ACP dialects, as data: one YAML file per vendored API-Version beside this module.

Why data. ACP changed its wire at almost every release (items became line items without a quantity, the fulfillment
address became fulfillment details, a payment provider became payment handlers, the status enum doubled), and the
same journey must be answered the way the release a store speaks answers it. Each difference is a field in
`<version>.yaml`; the operations and the renderer read the fields and branch only where semantics differ. The
checkout session status map is data too: every status a release's enum defines names the core checkout states it is
rendered for and the payment statuses beside it, or why the sandbox never shows it, and a test fails when a vendored
status has no entry.

A profile that names an ACP version with no file here is refused when the app is built, naming the versions served.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from importlib import resources
from typing import Literal

import pydantic
import yaml
from pydantic import BaseModel, ConfigDict, Field

from carttrace.core.errors import ValidationError
from carttrace.sandbox.surfaces.errors import UnsupportedVersion

CORE_STATES = (
    "incomplete",
    "ready_for_complete",
    "requires_escalation/payment_handler_unavailable",
    "requires_escalation/payment_action_required",
    "complete_in_progress",
    "completed",
    "canceled",
)
"""Every state a core checkout can be in, as a status map names it (`status` or `status/escalation`)."""
NO_PAYMENT = "none"
"""A status map's name for a checkout that has no payment attempt yet."""


class _Doc(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ItemsDoc(_Doc):
    field: Literal["items", "line_items"]
    quantity: bool


class FulfillmentDoc(_Doc):
    address: Literal["fulfillment_address", "fulfillment_details"]
    selection: Literal["option_id", "nested", "flat"]
    option_amounts: Literal["fields_required", "fields", "totals"]


class LineItemsDoc(_Doc):
    shape: Literal["amounts", "totals"]
    details: bool


class PaymentDoc(_Doc):
    data: Literal["provider", "handler"]
    methods: Literal["strings", "objects", "none"]
    merchant_id: bool
    display_name: bool


class SessionDoc(_Doc):
    capabilities: bool
    authentication: bool
    authentication_channel: bool
    continue_url: bool
    created_at: bool
    discounts: bool
    order_number: bool
    order_type: bool


class MessagesDoc(_Doc):
    severity: bool
    resolution: bool
    warnings: bool
    error_codes: tuple[str, ...]


class ErrorsDoc(_Doc):
    types: tuple[str, ...]
    supported_versions: bool


class IdempotencyDoc(_Doc):
    required: bool
    replayed_header: bool
    conflict_status: int
    conflict_type: str
    psp_conflict_type: str


class StatusDoc(_Doc):
    checkout: tuple[str, ...] = ()
    payments: tuple[str, ...] = ()
    unused: str | None = None


class DialectDoc(_Doc):
    version: str = Field(pattern=r"^\d{4}-\d{2}-\d{2}$")
    items: ItemsDoc
    fulfillment: FulfillmentDoc
    line_items: LineItemsDoc
    buyer: Literal["names_required", "email"]
    payment: PaymentDoc
    session: SessionDoc
    messages: MessagesDoc
    links: tuple[str, ...]
    link_titles: bool
    totals: tuple[str, ...]
    errors: ErrorsDoc
    idempotency: IdempotencyDoc
    cancel_body: bool
    discovery: bool
    signature_required: bool
    statuses: dict[str, StatusDoc]


@dataclass(frozen=True, slots=True)
class StatusRule:
    """One checkout session status of a release: the core states it is shown for, or why it never is."""

    name: str
    checkout: tuple[str, ...]
    payments: tuple[str, ...]
    unused: str | None


@dataclass(frozen=True, slots=True)
class AcpDialect:
    """One ACP API-Version as the sandbox renders it. Field names follow the YAML sections."""

    version: str
    items: ItemsDoc
    fulfillment: FulfillmentDoc
    line_items: LineItemsDoc
    buyer: str
    payment: PaymentDoc
    session: SessionDoc
    messages: MessagesDoc
    links: tuple[str, ...]
    link_titles: bool
    totals: tuple[str, ...]
    errors: ErrorsDoc
    idempotency: IdempotencyDoc
    cancel_body: bool
    discovery: bool
    signature_required: bool
    statuses: Mapping[str, StatusRule]

    def status_of(self, checkout_status: str, escalation: str | None) -> str:
        """The session status a core checkout state is rendered as."""
        key = f"{checkout_status}/{escalation}" if checkout_status == "requires_escalation" else checkout_status
        for rule in self.statuses.values():
            if key in rule.checkout:
                return rule.name
        raise ValidationError(
            f"ACP {self.version} maps no session status to the core state {key}",
            hint=f"add {key} to a status in src/carttrace/sandbox/surfaces/acp/{self.version}.yaml",
        )

    def code(self, wanted: str, fallback: str = "invalid") -> str:
        """`wanted` when this release's message error codes include it, else `fallback`."""
        return wanted if wanted in self.messages.error_codes else fallback


def _load(version: str) -> AcpDialect:
    raw = resources.files(__package__).joinpath(f"{version}.yaml").read_text(encoding="utf-8")
    try:
        doc = DialectDoc.model_validate(yaml.safe_load(raw))
    except pydantic.ValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        raise ValidationError(f"acp/{version}.yaml is malformed at {where}: {first['msg']}") from None
    if doc.version != version:
        raise ValidationError(f"acp/{version}.yaml says it is {doc.version}", hint="name the file for its version")
    statuses = {
        name: StatusRule(name, tuple(s.checkout), tuple(s.payments), s.unused) for name, s in doc.statuses.items()
    }
    return AcpDialect(
        version=doc.version,
        items=doc.items,
        fulfillment=doc.fulfillment,
        line_items=doc.line_items,
        buyer=doc.buyer,
        payment=doc.payment,
        session=doc.session,
        messages=doc.messages,
        links=doc.links,
        link_titles=doc.link_titles,
        totals=doc.totals,
        errors=doc.errors,
        idempotency=doc.idempotency,
        cancel_body=doc.cancel_body,
        discovery=doc.discovery,
        signature_required=doc.signature_required,
        statuses=statuses,
    )


def served_versions() -> list[str]:
    """Every ACP API-Version with a dialect file, oldest first."""
    names = (p.name for p in resources.files(__package__).iterdir())
    return sorted(n.removesuffix(".yaml") for n in names if n.endswith(".yaml"))


@cache
def dialect_for(version: str | None) -> AcpDialect:
    """The dialect of an ACP API-Version, or `UnsupportedVersion` naming the versions served."""
    if version not in served_versions():
        raise UnsupportedVersion(
            f"the sandbox has no ACP dialect for {version!r}",
            hint=f"serve a profile at one of {served_versions()} (presets acp-2026-04-17, live-acp-2026-04-17)",
        )
    return _load(str(version))
