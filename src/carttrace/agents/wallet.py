"""The wallet: the agent platform's payment component, which pays from a persona's opaque test tokens, and only after
the shopper has consented to the amount (decision D13).

Why the agent never holds a credential. An agent that can pay whenever it likes is an agent that can buy without
asking, and the judge's worst finding is an order nobody agreed to. So the agent owns no payment data at all: it asks
the wallet for a credential naming a handler, a checkout and an amount, and the wallet answers with an opaque token
only when the shopper has approved exactly that amount for that checkout. Each approval is spent by one credential,
so after a decline the agent cannot try again, with the same card or another, without asking the shopper first. A
newer answer from the shopper about the same checkout replaces any older one.

Personas. `data/agent/personas.yaml` holds synthetic shoppers (carttrace.agents.personas writes it): a fictional name, an
email at example.com, a phone number in the fictional 555-01xx range, a fictional delivery address per market, the
instruments they pay with (opaque tokens such as `tok_visa_3ds_challenge`, which the sandbox's PSP keys its behaviour
on), how they answer a 3-D Secure challenge, and the style and constraints task generation turns into variety. A v1
file (one shared address per market, no names) still loads. No card number exists anywhere, test cards included: the
loader refuses any value that looks like one.

Credential kinds. Every request names its kind, and an issuer per kind is registered in
`carttrace.agents.PLUGINS["agents.credential"]`, behind the same consent check:
- UCP: a handler token (`HandlerTokenRequest`, `HandlerTokenIssuer`), which WebMCP storefronts take too;
- ACP: a delegated payment token with an allowance whose maximum is the approved total (`DelegatedTokenRequest`,
  `carttrace.agents.dialects.acp.DelegatedTokenIssuer`): the persona's card credential, which the checkout harness then
  vaults at the store's PSP through the surface's own `delegate_payment`, so the agent pays with the vaulted token;
- MPP: a credential answering an HTTP 402 challenge (`PaymentChallengeRequest`), for the checkout's approved total
  (`carttrace.agents.dialects.mpp.ChallengeCredentialIssuer`, D66);
- AP2: the credential the payment mandate names (`MandateRequest`), for the approved total; the shopper's wallet then
  signs the checkout and payment mandates over it (`carttrace.agents.dialects.ap2.MandateIssuer`, `WalletSigner`, D66);
- Shop Pay (D73): a single-use Shop Token (`ShopTokenRequest`, `ShopPayTokenIssuer`): the persona's Shop Pay
  credential, which the checkout harness then hands to the store's Shop Pay interface (`ShopPayInterface`), so the
  agent pays with the minted `shop_...` token, scoped to the checkout, never with the credential itself — the spec's
  Token Acquisition Path A (vendor/dev.shopify.shop_pay/2026-04-08/spec.md).

3-D Secure. The buyer's bank, not the store, asks for the challenge, so the wallet presents it to the persona and
answers the challenge page as the persona would (`ChallengePage`). In the sandbox that page is served by the store's
own in-process app (`HttpChallengePage`), localhost only, and the exchange is recorded like any other.
"""

from __future__ import annotations

import dataclasses
import functools
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Protocol, Self

import httpx
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from pydantic import ValidationError as PydanticValidationError

from carttrace.agents.personas import PersonaConstraints, PersonaStyle, fictional_name
from carttrace.agents.reserved import EMAIL_HINT, PHONE_HINT, reserved_email, reserved_phone
from carttrace.core.canon import dumps_bytes, loads
from carttrace.core.cards import looks_like_card_number
from carttrace.core.clock import Clock, SystemClock
from carttrace.core.errors import ExternalServiceError, NotFoundError, SafetyError, ValidationError
from carttrace.core.money import Money, to_major
from carttrace.core.tokens import wallet_alias

PERSONAS_SCHEMA = "mc.personas.v2"
PERSONAS_SCHEMAS = ("mc.personas.v1", "mc.personas.v2")
"""v1: one address per market shared by every persona, no names. v2: each persona's own addresses, a fictional name,
and the style and constraints of carttrace.agents.personas."""
DEFAULT_PATH = Path(__file__).resolve().parents[3] / "data" / "agent" / "personas.yaml"
"""data/agent/personas.yaml in this repository."""
CREDENTIAL_KIND = "agents.credential"
"""The registry kind of the credential issuers (`carttrace.agents.PLUGINS`)."""

CredentialKind = Literal["ucp_handler_token", "acp_delegated_token", "mpp_credential", "ap2_mandate", "shop_pay_token"]
CREDENTIAL_KINDS: tuple[CredentialKind, ...] = (
    "ucp_handler_token",
    "acp_delegated_token",
    "mpp_credential",
    "ap2_mandate",
    "shop_pay_token",
)
InstrumentKind = Literal["card", "google_pay", "offline", "shop_pay"]
SHOP_PAY_NAMESPACE = "dev.shopify.shop_pay"
"""Shop Pay's UCP payment handler namespace (vendor/dev.shopify.shop_pay/2026-04-08/spec.md), named here as
HANDLER_INSTRUMENTS names every handler: the wallet reads no sandbox code."""
ThreeDsBehaviour = Literal["complete", "abandon"]
"""How a persona answers a 3-D Secure challenge: completes the step, or abandons it."""

HANDLER_INSTRUMENTS: Mapping[str, tuple[InstrumentKind, ...]] = MappingProxyType(
    {
        "dev.shopify.card": ("card",),
        "com.google.pay": ("google_pay",),
        "dev.ucp.sandbox.fake": ("card", "google_pay"),
        "com.example.payments.card": ("card", "google_pay"),
        "dev.acp.tokenized.card": ("card",),
        "acp.payment_provider": ("card",),
        "mpp.sandbox": ("card", "google_pay"),
        "ap2.card": ("card", "google_pay"),
        SHOP_PAY_NAMESPACE: ("shop_pay",),
    }
)
"""The instruments the wallet presents through each payment handler namespace, as the handlers' own specs say:
Shopify's card handler takes cards and Google Pay's takes Google Pay tokens; ACP's tokenized card handler (2026-01-30
on) and its earlier payment provider take a card, delegated; MPP's sandbox method and AP2's card instrument take a card
(D66), named here by the wallet, since neither protocol has a handler namespace; Shop Pay's handler takes a Shop Pay
instrument (a `shop_token` credential, its spec's instrument schema). A namespace not listed here is one this wallet
cannot pay through."""
OFFLINE_NAMESPACES: Mapping[str, str] = MappingProxyType({"com.woocommerce.cod": "cod", "com.woocommerce.bacs": "bacs"})
"""Pay-later gateways of a store we own (the WooCommerce playground's cash on delivery and bank transfer, through its
UCP plugin's registry, D44), to the gateway each settles as. Nothing is charged at checkout, so the wallet pays
through them with no card at all, behind the same consent; cash on delivery is preferred, since a bank transfer
leaves the order on hold. No live store declares these."""
FALLBACK_NAMESPACES = frozenset({"dev.ucp.sandbox.fake", "com.example.payments.card"})
"""The sandbox's own handler, which charges nobody, under its legacy name and the name a stand-in presents it by
(data/sandbox/visibility.yaml `identity.card_handler`, D106): chosen only when the store declares nothing else the
persona can pay through."""
MARKET_ALIASES: Mapping[str, str] = MappingProxyType({"UK": "GB"})
"""Market codes that name one market: the sandbox's table writes UK, the store queue and personas.yaml GB."""

_NAME = re.compile(r"^[a-z][a-z0-9-]*$")
_MARKET = re.compile(r"^[A-Z]{2}$")


# -- the personas file -------------------------------------------------------------------------------------------------


class _Doc(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)


class AddressDoc(_Doc):
    """A delivery address as UCP's postal address writes it, without the optional contact names."""

    street_address: str = Field(min_length=1)
    extended_address: str | None = None
    address_locality: str = Field(min_length=1)
    address_region: str | None = None
    postal_code: str = Field(min_length=1)
    address_country: str = Field(pattern=r"^[A-Z]{2}$")


class InstrumentDoc(_Doc):
    kind: InstrumentKind
    token: str = Field(pattern=r"^tok_[a-z0-9_]+$")
    """An opaque test token the sandbox's PSP knows (carttrace.sandbox.core.psp.TOKENS). Never a card number."""
    brand: str | None = Field(default=None, pattern=r"^[a-z]+$")
    """The card network, for display only."""


class PersonaDoc(_Doc):
    description: str = Field(min_length=1)
    email: str
    phone: str
    instruments: list[InstrumentDoc] = Field(min_length=1)
    """In the persona's order of preference."""
    three_ds: ThreeDsBehaviour = "complete"
    name: str | None = None
    """A fictional name, first and last, from carttrace.agents.personas' word lists; the name line of every address."""
    addresses: dict[str, AddressDoc] = Field(default_factory=dict)
    """The persona's own delivery addresses by market (v2); they come before the file's shared ones."""
    style: PersonaStyle = Field(default_factory=PersonaStyle)
    constraints: PersonaConstraints = Field(default_factory=PersonaConstraints)

    @field_validator("name")
    @classmethod
    def _fictional(cls, value: str | None) -> str | None:
        if value is not None and not fictional_name(value):
            raise ValueError("a persona's name is a capitalised first and last name from the word lists")
        return value

    @field_validator("email")
    @classmethod
    def _reserved_email(cls, value: str) -> str:
        if not reserved_email(value):
            raise ValueError(f"a persona's email must be reserved: {EMAIL_HINT}")
        return value

    @field_validator("phone")
    @classmethod
    def _fictional_phone(cls, value: str) -> str:
        if not reserved_phone(value):
            raise ValueError(f"a persona's phone must be reserved: {PHONE_HINT}")
        return value

    @field_validator("instruments")
    @classmethod
    def _one_per_kind(cls, value: list[InstrumentDoc]) -> list[InstrumentDoc]:
        kinds = [i.kind for i in value]
        if len(set(kinds)) != len(kinds):
            raise ValueError("a persona holds at most one instrument of each kind")
        return value


class PersonasFile(_Doc):
    schema_: Literal["mc.personas.v1", "mc.personas.v2"] = Field(alias="schema")
    version: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    default: str
    addresses: dict[str, AddressDoc] = Field(default_factory=dict)
    """Delivery market code (US, GB, EU, ...) to an address every persona ships to there, unless it has its own."""
    personas: dict[str, PersonaDoc]

    @model_validator(mode="after")
    def _consistent(self) -> Self:
        bad_names = sorted(n for n in self.personas if not _NAME.match(n))
        if bad_names:
            raise ValueError(f"persona names are lowercase slugs: {bad_names}")
        keyed = [*self.addresses, *(m for p in self.personas.values() for m in p.addresses)]
        bad_markets = sorted(m for m in keyed if not _MARKET.match(m))
        if bad_markets:
            raise ValueError(f"address keys are two-letter market codes: {bad_markets}")
        if self.default not in self.personas:
            raise ValueError(f"the default persona {self.default!r} is not defined")
        homeless = sorted(n for n, p in self.personas.items() if not p.addresses and not self.addresses)
        if homeless:
            raise ValueError(f"personas with no delivery address in any market: {homeless}")
        return self


# -- personas ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Instrument:
    kind: InstrumentKind
    token: str
    brand: str | None = None


@dataclass(frozen=True, slots=True)
class Persona:
    """One synthetic shopper: contact details, delivery addresses by market, instruments and 3-D Secure behaviour."""

    name: str
    description: str
    email: str
    phone: str
    instruments: tuple[Instrument, ...]
    three_ds: ThreeDsBehaviour
    addresses: Mapping[str, Mapping[str, str]]
    """Delivery market code to a UCP postal address."""
    display_name: str | None = None
    """The fictional name on the persona's addresses; None for a v1 file's personas."""
    style: PersonaStyle = field(default_factory=PersonaStyle)
    constraints: PersonaConstraints = field(default_factory=PersonaConstraints)

    @property
    def markets(self) -> tuple[str, ...]:
        return tuple(sorted(self.addresses))

    def address_for(self, market: str) -> dict[str, str] | None:
        """The persona's delivery address in `market` (GB or UK alike), or None when it has none there."""
        code = market.strip().upper()
        found = self.addresses.get(MARKET_ALIASES.get(code, code))
        return dict(found) if found is not None else None

    def instrument_for(self, namespace: str) -> Instrument | None:
        """The persona's preferred instrument that a handler namespace takes, or None. An offline gateway takes no
        instrument of the persona's: it is paid later, so it is offered as an `offline` one."""
        gateway = OFFLINE_NAMESPACES.get(namespace)
        if gateway is not None:
            return Instrument(kind="offline", token=f"tok_offline_{gateway}")
        takes = HANDLER_INSTRUMENTS.get(namespace, ())
        return next((i for i in self.instruments if i.kind in takes), None)


@dataclass(frozen=True, slots=True)
class Personas:
    version: str
    default: str
    by_name: Mapping[str, Persona]

    def names(self) -> list[str]:
        return sorted(self.by_name)

    def get(self, name: str | None = None) -> Persona:
        """A persona by name, the default one when `name` is None."""
        wanted = self.default if name is None else name
        found = self.by_name.get(wanted)
        if found is None:
            raise NotFoundError(f"no persona {wanted!r}", hint=f"choose one of {', '.join(self.names())}")
        return found

    def markets(self) -> dict[str, tuple[str, ...]]:
        """Persona name to the delivery markets it can ship to."""
        return {name: persona.markets for name, persona in sorted(self.by_name.items())}


def load_personas(path: Path | None = None) -> Personas:
    """The personas at `path` (default: data/agent/personas.yaml), parsed once per process and checked."""
    return _load((path or DEFAULT_PATH).resolve())


@functools.cache
def _load(path: Path) -> Personas:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotFoundError(f"no personas at {path}", hint="restore data/agent/personas.yaml from git") from None
    except yaml.YAMLError as exc:
        raise ValidationError(f"{path.name} is not valid YAML: {exc}") from exc
    try:
        doc = PersonasFile.model_validate(raw)
    except PydanticValidationError as exc:
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"]) or "file"
        raise ValidationError(
            f"{path.name}: {where}: {first['msg']}", hint=f"follow {PERSONAS_SCHEMA} (carttrace.agents.personas)"
        ) from exc
    carded = next(
        (where for where, text in _strings(doc.model_dump(by_alias=True)) if looks_like_card_number(text)), None
    )
    if carded is not None:
        raise SafetyError(
            f"{path.name}: {carded} looks like a card number",
            hint="personas pay with opaque test tokens (tok_visa_ok); no card number belongs anywhere, test cards "
            "included",
        )
    shared = {market: MappingProxyType(a.model_dump(exclude_none=True)) for market, a in sorted(doc.addresses.items())}
    personas = {
        name: Persona(
            name=name,
            description=p.description,
            email=p.email,
            phone=p.phone,
            instruments=tuple(Instrument(i.kind, i.token, i.brand) for i in p.instruments),
            three_ds=p.three_ds,
            addresses=MappingProxyType(
                {
                    **shared,
                    **{m: MappingProxyType(a.model_dump(exclude_none=True)) for m, a in sorted(p.addresses.items())},
                }
            ),
            display_name=p.name,
            style=p.style,
            constraints=p.constraints,
        )
        for name, p in sorted(doc.personas.items())
    }
    return Personas(version=doc.version, default=doc.default, by_name=MappingProxyType(personas))


def _strings(value: Any, where: str = "") -> Iterator[tuple[str, str]]:
    if isinstance(value, str):
        yield where or "value", value
    elif isinstance(value, Mapping):
        for k, v in value.items():
            yield from _strings(v, f"{where}.{k}" if where else str(k))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            yield from _strings(v, f"{where}[{i}]")


# -- credentials -------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HandlerTokenRequest:
    """UCP: a token for one of the checkout's declared payment handlers, for the checkout's total."""

    checkout_id: str
    handler_id: str
    namespace: str
    amount: Money
    kind: Literal["ucp_handler_token"] = "ucp_handler_token"


@dataclass(frozen=True, slots=True)
class DelegatedTokenRequest:
    """ACP: a delegated payment token whose allowance (`max_amount`, merchant, expiry) caps what it may charge, for
    the handler the completion will name (`handler_id`)."""

    checkout_id: str
    allowance: Money
    merchant_id: str
    expires_at: str
    kind: Literal["acp_delegated_token"] = "acp_delegated_token"
    handler_id: str | None = None

    @property
    def amount(self) -> Money:
        return self.allowance


@dataclass(frozen=True, slots=True)
class PaymentChallengeRequest:
    """MPP: a credential answering an HTTP 402 challenge; `checkout_id` is the id the challenge names."""

    checkout_id: str
    challenge: Mapping[str, Any]
    amount: Money
    kind: Literal["mpp_credential"] = "mpp_credential"


@dataclass(frozen=True, slots=True)
class MandateRequest:
    """AP2: the credential a payment mandate names, for the checkout the merchant signed; the wallet signs the
    mandates themselves (`carttrace.agents.dialects.ap2.WalletSigner`)."""

    checkout_id: str
    mandate: Literal["intent", "cart", "payment"]
    contents: Mapping[str, Any]
    amount: Money
    kind: Literal["ap2_mandate"] = "ap2_mandate"


@dataclass(frozen=True, slots=True)
class ShopTokenRequest:
    """Shop Pay (D73): a single-use Shop Token for one checkout, at its approved total. The wallet's platform mints
    it from the persona's Shop Pay credential through the store's Shop Pay interface (the spec's Path A); the agent
    pays with the minted token, never with the credential."""

    checkout_id: str
    handler_id: str
    amount: Money
    kind: Literal["shop_pay_token"] = "shop_pay_token"


CredentialRequest = (
    HandlerTokenRequest | DelegatedTokenRequest | PaymentChallengeRequest | MandateRequest | ShopTokenRequest
)


@dataclass(frozen=True, slots=True)
class Consent:
    """The shopper approved paying `amount` for one checkout. Spent by one credential, replaced by a newer answer."""

    checkout_id: str
    amount: Money
    seq: int
    """Its place among the episode's consents, from 1."""


@dataclass(frozen=True, slots=True)
class Credential:
    """What the wallet hands out: an opaque token for one handler, checkout and amount. Never a card number."""

    kind: CredentialKind
    checkout_id: str
    amount: Money
    token: str
    instrument: InstrumentKind
    instrument_id: str
    consent: int
    """The seq of the consent it spent."""
    handler_id: str | None = None
    brand: str | None = None

    def ucp_instrument(self) -> dict[str, Any]:
        """The UCP payment instrument for complete_checkout's `checkout.payment.instruments`. A Shop Pay credential
        takes the handler's own shape (the spec's `shop_pay` instrument with a `shop_token` credential), and a Google
        Pay token Google's (a `card` instrument with a `PAYMENT_GATEWAY` credential, the com.google.pay handler guide,
        D94)."""
        credential_type = {"shop_pay_token": "shop_token"}.get(self.kind, "token")
        kind: str = self.instrument
        if self.instrument == "google_pay":
            credential_type, kind = "PAYMENT_GATEWAY", "card"
        out: dict[str, Any] = {
            "id": self.instrument_id,
            "handler_id": self.handler_id,
            "type": kind,
            "selected": True,
            "credential": {"type": credential_type, "token": self.token},
        }
        if self.brand:
            out["display"] = {"brand": self.brand}
        return out


@dataclass(frozen=True, slots=True)
class Refusal:
    """The wallet would not hand out a credential, and why. The agent must handle it: ask the shopper, or stop."""

    code: str
    message: str
    hint: str = ""


class CredentialIssuer(Protocol):
    """Issues one kind of credential. The wallet checks consent first; an issuer only says whether it can pay the
    request and, if so, makes the credential."""

    @property
    def kind(self) -> CredentialKind: ...

    def refusal(self, persona: Persona, request: CredentialRequest) -> Refusal | None:
        """Why this request cannot be paid from this persona's wallet, or None when it can."""
        ...

    def issue(self, persona: Persona, request: CredentialRequest, consent: Consent, *, sequence: int) -> Credential:
        """The credential, spending `consent`. Called only after `refusal` said None."""
        ...


class HandlerTokenIssuer:
    """UCP handler tokens: the persona's preferred instrument that the named handler's namespace takes."""

    @property
    def kind(self) -> CredentialKind:
        return "ucp_handler_token"

    def refusal(self, persona: Persona, request: CredentialRequest) -> Refusal | None:
        if not isinstance(request, HandlerTokenRequest):
            return Refusal("credential_kind_mismatch", f"a {request.kind} request reached the UCP token issuer")
        if persona.instrument_for(request.namespace) is None:
            kinds = ", ".join(i.kind for i in persona.instruments)
            return Refusal(
                "handler_unsupported",
                f"the shopper's wallet holds nothing the {request.namespace} handler takes (it holds: {kinds})",
                hint="pay through one of the checkout's handlers the wallet holds a credential for",
            )
        return None

    def issue(self, persona: Persona, request: CredentialRequest, consent: Consent, *, sequence: int) -> Credential:
        assert isinstance(request, HandlerTokenRequest)
        instrument = persona.instrument_for(request.namespace)
        assert instrument is not None
        return Credential(
            kind=self.kind,
            checkout_id=request.checkout_id,
            amount=request.amount,
            token=instrument.token,
            instrument=instrument.kind,
            instrument_id=f"wallet-instrument-{sequence}",
            consent=consent.seq,
            handler_id=request.handler_id,
            brand=instrument.brand,
        )


class ShopPayTokenIssuer:
    """Shop Pay tokens (D73): the persona's Shop Pay credential, issued for one checkout at the approved total. The
    wallet's consent gate runs first; the checkout harness then mints the single-use Shop Token at the store's Shop
    Pay interface (`Wallet.mint_shop_token`, the spec's Path A) and the agent pays with it, never with this one."""

    @property
    def kind(self) -> CredentialKind:
        return "shop_pay_token"

    def refusal(self, persona: Persona, request: CredentialRequest) -> Refusal | None:
        if not isinstance(request, ShopTokenRequest):
            return Refusal("credential_kind_mismatch", f"a {request.kind} request reached the Shop Pay token issuer")
        if persona.instrument_for(SHOP_PAY_NAMESPACE) is None:
            kinds = ", ".join(i.kind for i in persona.instruments)
            return Refusal(
                "handler_unsupported",
                f"the shopper's wallet holds no Shop Pay credential (it holds: {kinds})",
                hint="pay through another of the checkout's handlers, or ask the shopper",
            )
        return None

    def issue(self, persona: Persona, request: CredentialRequest, consent: Consent, *, sequence: int) -> Credential:
        assert isinstance(request, ShopTokenRequest)
        instrument = persona.instrument_for(SHOP_PAY_NAMESPACE)
        assert instrument is not None
        return Credential(
            kind=self.kind,
            checkout_id=request.checkout_id,
            amount=request.amount,
            token=instrument.token,
            instrument=instrument.kind,
            instrument_id=f"wallet-instrument-{sequence}",
            consent=consent.seq,
            handler_id=request.handler_id,
            brand=instrument.brand,
        )


def default_issuers() -> dict[CredentialKind, CredentialIssuer]:
    """Every issuer registered in carttrace.agents PLUGINS["agents.credential"], by kind."""
    from carttrace.core.registry import names, resolve

    out: dict[CredentialKind, CredentialIssuer] = {}
    for kind in CREDENTIAL_KINDS:
        if kind in names(CREDENTIAL_KIND):
            out[kind] = resolve(CREDENTIAL_KIND, kind)()
    return out


# -- handlers ----------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class HandlerOffer:
    """One payment handler a checkout declares, and what the wallet would pay through it."""

    handler_id: str
    namespace: str
    version: str | None
    instrument: InstrumentKind | None
    """The instrument the wallet would present; None when it cannot pay through this handler."""
    brand: str | None = None

    @property
    def payable(self) -> bool:
        return self.instrument is not None


# -- the Shop Pay interface --------------------------------------------------------------------------------------------


class ShopPayInterface(Protocol):
    """The store's Shop Pay interface (the spec's Path A): the wallet's platform mints a single-use Shop Token here,
    scoped to one checkout and its total. The agent never calls it: the checkout harness does, after consent."""

    async def mint(self, credential: Credential, request: ShopTokenRequest) -> Mapping[str, Any]:
        """The minted token's record (`token`, `expires_at_ms`, ...), or an `ExternalServiceError`."""
        ...


class HttpShopPay:
    """The Shop Pay interface of the sandbox's own app, through the in-process client the store's surface uses, so
    no socket opens. Only URLs on that client's own host are answered, as with the 3-D Secure page: a store that
    pointed the interface anywhere else is refused. The exchange is recorded like any other (surface `shop_pay`)."""

    PATH = "/psp/shop_pay/tokens"
    VERSION = "2026-04-08"
    """The Shop Pay handler version whose token flow this speaks (vendor/dev.shopify.shop_pay)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        session_id: str,
        *,
        recorder: WireSink | None = None,
        clock: Clock | None = None,
        session_header: str = "mc-sandbox-session",
    ) -> None:
        """`session_header` is the header the store's surfaces name a session in (its presented one, D106)."""
        self.client = client
        self.session_id = session_id
        self.session_header = session_header
        self.recorder = recorder
        self.clock: Clock = clock or SystemClock()

    async def mint(self, credential: Credential, request: ShopTokenRequest) -> Mapping[str, Any]:
        body = {
            "checkout_id": request.checkout_id,
            "token": credential.token,
            "amount": request.amount.amount,
            "currency": request.amount.currency,
        }
        start = self.clock.monotonic()
        response = await self.client.post(
            self.PATH,
            content=dumps_bytes(body),
            headers={
                "content-type": "application/json",
                "accept": "application/json",
                self.session_header: self.session_id,
            },
        )
        latency = max(0, round((self.clock.monotonic() - start) * 1000))
        parsed = _json(response)
        if self.recorder is not None:
            self.recorder.wire(
                {
                    "surface": "shop_pay",
                    "protocol": "shop_pay",
                    "version": self.VERSION,
                    "method": "POST",
                    "url": str(response.request.url),
                    "request_headers": dict(response.request.headers),
                    "request_body": body,
                    "status": response.status_code,
                    "response_headers": dict(response.headers),
                    "response_body": parsed if parsed is not None else response.text or None,
                    "duration_ms": latency,
                    "error": None,
                }
            )
        if response.status_code != 201 or not isinstance(parsed, Mapping) or not isinstance(parsed.get("token"), str):
            raise ExternalServiceError(
                "shop_pay",
                f"the Shop Pay interface answered {response.status_code}: {response.text[:200]}",
                hint="the store may serve no Shop Pay handler; pay through another of its handlers",
                status=response.status_code,
            )
        return dict(parsed)


# -- 3-D Secure --------------------------------------------------------------------------------------------------------


class WireSink(Protocol):
    """Where an exchange is recorded. A TraceWriter is one."""

    def wire(self, record: Mapping[str, Any]) -> Any: ...


class ChallengePage(Protocol):
    """The buyer's bank's 3-D Secure page: answered by the persona, through the wallet."""

    async def answer(self, url: str, *, approved: bool) -> Mapping[str, Any]:
        """Post the buyer's answer to the challenge page at `url`; what the page answered."""
        ...


@dataclass(frozen=True, slots=True)
class ChallengeOutcome:
    approved: bool
    """The persona completed the step (abandoning it is not approving)."""
    answer: Mapping[str, Any]
    """What the bank's page answered."""


class HttpChallengePage:
    """The challenge page of the sandbox's own app, through the in-process client the store's surface uses, so no
    socket opens. Only URLs on that client's own host are answered: a page anywhere else is refused, whatever a store
    put in its Action or continue_url."""

    def __init__(self, client: httpx.AsyncClient, *, recorder: WireSink | None = None, clock: Clock | None = None):
        self.client = client
        self.recorder = recorder
        self.clock: Clock = clock or SystemClock()

    async def answer(self, url: str, *, approved: bool) -> Mapping[str, Any]:
        target = httpx.URL(url)
        base = self.client.base_url
        if (target.scheme, target.host, target.port) != (base.scheme, base.host, base.port):
            raise SafetyError(
                f"the challenge page {target.host or url!r} is not on the sandbox's own host",
                hint="only the sandbox's 3-D Secure page is answered; check the checkout's Action or continue_url",
            )
        body = {"approved": approved}
        start = self.clock.monotonic()
        response = await self.client.post(
            target.raw_path.decode("ascii"),
            content=dumps_bytes(body),
            headers={"content-type": "application/json", "accept": "application/json"},
        )
        latency = max(0, round((self.clock.monotonic() - start) * 1000))
        parsed = _json(response)
        if self.recorder is not None:
            self.recorder.wire(
                {
                    "surface": "3ds_page",
                    "protocol": "3ds",
                    "version": "2.2.0",  # the EMV 3-D Secure protocol version the challenge page stands for
                    "method": "POST",
                    "url": str(response.request.url),
                    "request_headers": dict(response.request.headers),
                    "request_body": body,
                    "status": response.status_code,
                    "response_headers": dict(response.headers),
                    "response_body": parsed if parsed is not None else response.text or None,
                    "duration_ms": latency,
                    "error": None,
                }
            )
        if response.status_code != 200 or not isinstance(parsed, Mapping):
            raise ExternalServiceError(
                "3ds_page",
                f"the challenge page answered {response.status_code}: {response.text[:200]}",
                hint="get the checkout: the challenge may have expired or been answered already",
                status=response.status_code,
            )
        return dict(parsed)


def _json(response: httpx.Response) -> Any:
    if not response.content or "json" not in response.headers.get("content-type", ""):
        return None
    try:
        return loads(response.content)
    except ValueError:
        return None


# -- the wallet --------------------------------------------------------------------------------------------------------


@dataclass(slots=True)
class _Ledger:
    consents: list[Consent] = field(default_factory=list)
    spent: set[int] = field(default_factory=set)
    """Seqs of consents a credential spent, or a newer answer replaced."""


class Wallet:
    """One shopper's wallet for one episode: the persona's instruments and saved delivery details, the consents the
    shopper gave, and the credentials issued against them."""

    def __init__(
        self,
        persona: Persona,
        *,
        market: str,
        challenge_page: ChallengePage | None = None,
        shop_pay: ShopPayInterface | None = None,
        prefer: str | None = None,
        issuers: Mapping[CredentialKind, CredentialIssuer] | None = None,
        alias_key: str | None = None,
    ) -> None:
        address = persona.address_for(market)
        if address is None:
            raise NotFoundError(
                f"persona {persona.name} has no delivery address in market {market}",
                hint=f"add one to data/agent/personas.yaml, or choose a market of {', '.join(persona.markets)}",
            )
        self.persona = persona
        self.market = market
        self.address = address
        self.challenge_page = challenge_page
        self.shop_pay = shop_pay
        self.prefer = prefer
        """The handler namespace a payment rail asks the wallet to recommend first (D73: Shop Pay on a shop_pay
        row); it only reorders what the store declares and the persona can pay through, it never adds a handler."""
        self.issuers: dict[CredentialKind, CredentialIssuer] = (
            dict(issuers) if issuers is not None else default_issuers()
        )
        self.issued: list[Credential] = []
        self._ledger = _Ledger()
        self.alias_key = alias_key
        """The episode's sandbox session: with it, every token the agent is shown is its opaque alias
        (`carttrace.core.tokens`), never the persona's test token, whose name says what it is built to do."""
        self.offers_challenge = True
        """Whether the agent is offered three_ds_challenge. False where the store's PSP decides 3-D Secure at once,
        accepted or declined, so there is never a bank page to hand the shopper (every v1 row, D111)."""

    # -- what the wallet holds ---------------------------------------------------------------------------------------

    @property
    def consents(self) -> tuple[Consent, ...]:
        return tuple(self._ledger.consents)

    def set_aside(self, token: str) -> bool:
        """Leave out the instrument `token` names after its issuer declined it, as a shopper picks another card, and
        say whether the wallet still holds an instrument to pay with (sandbox v1 final, D115: a designed decline with
        another instrument in the wallet is recoverable, and the reference agent recovers). Nothing is set aside when
        the declined instrument is the wallet's only one."""
        kept = tuple(i for i in self.persona.instruments if token not in (i.token, self.shown(i.token)))
        if not kept or len(kept) == len(self.persona.instruments):
            return False
        self.persona = replace(self.persona, instruments=kept)
        return True

    def shipping_details(self) -> dict[str, Any]:
        """The shopper's saved details for a checkout: contact facts for `buyer`, and the address in the store's
        market for the shipping destination."""
        return {"email": self.persona.email, "phone_number": self.persona.phone, "address": dict(self.address)}

    def offers(self, handlers: Mapping[str, Any]) -> list[HandlerOffer]:
        """Every handler a checkout's `ucp.payment_handlers` declares (namespace to its entries), the ones the wallet
        can pay through first, in the persona's order of preference, with the sandbox's own handler last."""
        out: list[HandlerOffer] = []
        for namespace in sorted(handlers):
            entries = handlers[namespace]
            for entry in entries if isinstance(entries, list) else [entries]:
                if not isinstance(entry, Mapping):
                    continue
                instrument = self.persona.instrument_for(namespace)
                version = entry.get("version")
                out.append(
                    HandlerOffer(
                        handler_id=str(entry.get("id") or namespace),
                        namespace=namespace,
                        version=version if isinstance(version, str) else None,
                        instrument=instrument.kind if instrument is not None else None,
                        brand=instrument.brand if instrument is not None else None,
                    )
                )
        return sorted(out, key=self._rank)

    def _rank(self, offer: HandlerOffer) -> tuple[int, int, int, int, str, str]:
        kinds = [i.kind for i in self.persona.instruments]
        preference = kinds.index(offer.instrument) if offer.instrument in kinds else len(kinds)
        if offer.namespace in OFFLINE_NAMESPACES:
            preference = len(kinds) + list(OFFLINE_NAMESPACES).index(offer.namespace)
        return (
            0 if offer.payable else 1,
            0 if offer.namespace == self.prefer else 1,
            1 if offer.namespace in FALLBACK_NAMESPACES else 0,
            preference,
            offer.namespace,
            offer.handler_id,
        )

    # -- consent -----------------------------------------------------------------------------------------------------

    def approve(self, checkout_id: str, amount: Money) -> Consent:
        """Record that the shopper approved paying `amount` for `checkout_id`. Any older open approval of the same
        checkout is replaced: the shopper's latest answer is the one that counts."""
        self._close(checkout_id)
        consent = Consent(checkout_id, amount, len(self._ledger.consents) + 1)
        self._ledger.consents.append(consent)
        return consent

    def refuse(self, checkout_id: str) -> None:
        """Record that the shopper said no about `checkout_id`: no earlier approval of it can be spent any more."""
        self._close(checkout_id)

    def _close(self, checkout_id: str) -> None:
        for c in self._ledger.consents:
            if c.checkout_id == checkout_id:
                self._ledger.spent.add(c.seq)

    def _open(self, checkout_id: str) -> list[Consent]:
        return [c for c in self._ledger.consents if c.checkout_id == checkout_id and c.seq not in self._ledger.spent]

    # -- credentials -------------------------------------------------------------------------------------------------

    def issue(self, request: CredentialRequest) -> Credential | Refusal:
        """A credential for `request`, or why not. Only an open approval of exactly this checkout and amount is
        honoured, and the credential spends it."""
        issuer = self.issuers.get(request.kind)
        if issuer is None:
            return Refusal(
                "credential_kind_unsupported",
                f"this wallet does not issue {request.kind} credentials yet",
                hint="pay through one of the checkout's UCP payment handlers",
            )
        refused = issuer.refusal(self.persona, request)
        if refused is not None:
            return refused
        open_ = self._open(request.checkout_id)
        if not open_:
            asked = any(c.checkout_id == request.checkout_id for c in self._ledger.consents)
            return Refusal(
                "consent_spent" if asked else "no_consent",
                "the shopper's approval for this checkout was already used, or withdrawn"
                if asked
                else "the shopper has not approved paying for this checkout",
                hint="have the shopper approve the order and its total in the wallet's own step first",
            )
        match = next((c for c in open_ if c.amount == request.amount), None)
        if match is None:
            approved = open_[-1].amount
            return Refusal(
                "consent_amount_differs",
                f"the shopper approved {_shown(approved)} for this checkout, not {_shown(request.amount)}",
                hint="ask the shopper again, stating the checkout's current total",
            )
        credential = issuer.issue(self.persona, request, match, sequence=len(self.issued) + 1)
        credential = dataclasses.replace(credential, token=self.shown(credential.token))
        self._ledger.spent.add(match.seq)
        self.issued.append(credential)
        return credential

    def shown(self, token: str) -> str:
        """The token as the agent may see it: the episode's alias (the v1 prompt review: the agent read
        `tok_visa_3ds_fail` and reasoned that 3-D Secure would fail). An offline gateway's marker names no fault."""
        if self.alias_key is None or token.startswith("tok_offline_"):
            return token
        return wallet_alias(self.alias_key, token)

    # -- Shop Pay ------------------------------------------------------------------------------------------------------

    async def mint_shop_token(self, credential: Credential, request: ShopTokenRequest) -> str | Refusal:
        """The single-use Shop Token for an issued Shop Pay credential, minted at the store's Shop Pay interface
        (D73). The approval the credential spent stays spent when the mint fails: as after a decline, the agent asks
        the shopper before trying another way to pay."""
        if self.shop_pay is None:
            return Refusal(
                "shop_pay_unavailable",
                "this harness cannot reach the store's Shop Pay interface to mint a token",
                hint="tell the shopper the store's Shop Pay payment could not be set up",
            )
        try:
            minted = await self.shop_pay.mint(credential, request)
        except ExternalServiceError as exc:
            return Refusal("shop_pay_refused", f"the Shop Pay interface would not mint a token: {exc}", exc.hint)
        token = minted.get("token")
        if not isinstance(token, str) or not token.startswith("shop_"):
            return Refusal(
                "shop_pay_failed",
                "the Shop Pay interface answered without a Shop Token",
                "get the checkout and try again, or pay through another handler",
            )
        return token

    # -- 3-D Secure --------------------------------------------------------------------------------------------------

    async def present_challenge(self, url: str) -> ChallengeOutcome | Refusal:
        """Show the shopper their bank's 3-D Secure challenge and answer it as the persona does."""
        if self.challenge_page is None:
            return Refusal(
                "challenge_unavailable",
                "this harness cannot present a 3-D Secure challenge to the shopper",
                hint="tell the shopper the bank asked for a challenge you cannot complete",
            )
        approved = self.persona.three_ds == "complete"
        answer = await self.challenge_page.answer(url, approved=approved)
        return ChallengeOutcome(approved=approved, answer=answer)


def wallet_for(
    persona: str | None,
    *,
    market: str,
    challenge_page: ChallengePage | None = None,
    shop_pay: ShopPayInterface | None = None,
    prefer: str | None = None,
    path: Path | None = None,
    alias_key: str | None = None,
) -> Wallet:
    """The wallet of a named persona (the default one when None) shopping in `market`."""
    return Wallet(
        load_personas(path).get(persona),
        market=market,
        challenge_page=challenge_page,
        shop_pay=shop_pay,
        prefer=prefer,
        alias_key=alias_key,
    )


def _shown(money: Money) -> str:
    return f"{to_major(money.amount, money.currency)} {money.currency}"
