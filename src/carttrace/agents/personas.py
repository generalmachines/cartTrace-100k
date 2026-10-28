"""Personas as data: who shops, how they pay, where they live and how they behave, and the generator that writes
data/agent/personas.yaml from word lists so no persona is ever a real person.

Why generated. The project asked for protocol agents to be run as buyers of many kinds and the six payment personas of D13 all shipped
to one sandbox address with no name. A persona here is a fictional name made from two word lists, a fictional address
per delivery market shaped like that market's real addresses (a US ZIP, a UK postcode, a German Postleitzahl), a
reserved email and phone (carttrace.agents.reserved), a payment behaviour (one of the sandbox PSP's opaque test tokens), and
the styles task generation turns into hidden-spec variety (carttrace.agents.intents.Variety): how tight the budget is, how
fast they decide, what they volunteer or keep for asking (one persona forgets a detail and has to be asked twice), how
they like delivery, and the constraints a category may turn into predicates (allergens, dietary needs, body
measurements, conditions, materials to avoid).

Deterministic: `render_personas()` is a pure function of this module's word lists and version, and a test holds the
committed file to it. The loader (carttrace.agents.wallet.load_personas) still refuses any contact value outside the
reserved ranges and anything that looks like a card number. The six personas of the first file keep their names (slugs),
emails, phones and tokens, and `card-ok`, the default, keeps its sandbox addresses, so older tasks and tests that name
them hold.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from random import Random
from typing import Any, Final, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from carttrace.core.ids import seeded_rng

PERSONAS_VERSION: Final = "personas-v2"
BudgetStyle = Literal["tight", "normal", "generous"]
Decisiveness = Literal["fast", "deliberate"]
Disclosure = Literal["volunteers", "mixed", "asks", "forgets"]
"""volunteers: states every decisive fact and most options up front. mixed: the first generator's draws (three tasks
in four state the decisive fact). asks: keeps most for when the agent asks. forgets: as asks, and one on_ask detail is
not to hand the first time it is asked."""
DeliveryStyle = Literal["standard", "express", "pickup"]
_NAME: Final = re.compile(r"^[A-Z][a-z]+(?:[ -][A-Z][a-z]+)+$")
"""Two or more capitalised words: a first name and a last name from the word lists, never a single word."""


class _Doc(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class PersonaStyle(_Doc):
    """How a persona shops; task generation reads these through carttrace.agents.intents.variety_for."""

    budget: BudgetStyle = "normal"
    decisiveness: Decisiveness = "fast"
    disclosure: Disclosure = "mixed"
    delivery: DeliveryStyle = "standard"


class PersonaConstraints(_Doc):
    """Needs a category may turn into predicates: an allergen into `excludes`, a measurement into `fits`, a condition
    into `safe_for`. Values are the shared vocabulary's (data/categories/_fields.yaml); measurements are "102 cm"."""

    allergens: list[str] = Field(default_factory=list)
    dietary: list[str] = Field(default_factory=list)
    conditions: list[str] = Field(default_factory=list)
    measurements: dict[str, str] = Field(default_factory=dict)
    avoid_materials: list[str] = Field(default_factory=list)

    @field_validator("measurements")
    @classmethod
    def _amount_and_unit(cls, value: dict[str, str]) -> dict[str, str]:
        for measure, text in value.items():
            if not re.fullmatch(r"\d+(?:\.\d+)? (?:cm|mm|in)", text):
                raise ValueError(f"measurement {measure!r} must read like '102 cm', got {text!r}")
        return value


def fictional_name(value: str) -> bool:
    """A name of the shape the generator writes: capitalised words, at least two."""
    return bool(_NAME.match(value))


# -- word lists --------------------------------------------------------------------------------------------------------

FIRST_NAMES: Final = (
    "Mara", "Idris", "Noor", "Teodor", "Saoirse", "Kwame", "Ingrid", "Rafael", "Ananya", "Haruto", "Leila", "Bram",
    "Seun", "Priya", "Callum", "Zofia", "Mateo", "Yuki", "Femi", "Astrid", "Devika", "Rosalind", "Tariq", "Elin",
    "Oskar", "Imani", "Wren",
)  # fmt: skip
LAST_NAMES: Final = (
    "Thornvale", "Ashbury", "Quillfeather", "Brightwater", "Fennimore", "Halloway", "Kestrelbrook", "Larkwood",
    "Moorcroft", "Vance-Nightingale", "Oakhurst", "Pennywhistle", "Ravensworth", "Silverleaf", "Tanglewood",
    "Underhill", "Vexley", "Wintermere", "Yarrowdale", "Coldstream", "Dunmore-Ashe", "Everhart", "Fallowfield",
    "Greywell", "Hollins", "Marrowby", "Calloway",
)  # fmt: skip
"""Invented surnames; a first name and a last name are joined by position, never drawn from a real directory."""
STREETS: Final = (
    "Juniper", "Harbour", "Alder", "Meridian", "Larkspur", "Copperfield", "Bramble", "Saltmarsh", "Orchard", "Tallow",
    "Wren", "Foxglove", "Kestrel", "Marigold", "Lantern", "Quarry", "Heron", "Willowmere", "Birchwood", "Sorrel",
    "Pennant", "Ashgrove", "Moss", "Linden", "Fieldfare", "Tamarind",
)  # fmt: skip

Place = tuple[str, str | None, str]
"""(locality, region, postal code template: # a digit, @ a capital letter, other characters literal)."""
MARKET_SHAPES: Final[Mapping[str, tuple[str, tuple[str, ...], tuple[Place, ...]]]] = {
    "US": (
        "US",
        ("{n} {street} Street", "{n} {street} Avenue", "{n} {street} Drive", "{n} {street} Lane"),
        (
            ("Springfield", "IL", "627##"),
            ("Portland", "OR", "972##"),
            ("Austin", "TX", "787##"),
            ("Albany", "NY", "122##"),
        ),
    ),
    "CA": (
        "CA",
        ("{n} {street} Avenue", "{n} {street} Street", "{n} {street} Crescent"),
        (("Ottawa", "ON", "K1P 1@#"), ("Toronto", "ON", "M5V 2@#"), ("Vancouver", "BC", "V6B 3@#")),
    ),
    "GB": (
        "GB",
        ("{n} {street} Road", "{n} {street} Lane", "{n} {street} Close", "{n} {street} Gardens"),
        (
            ("London", None, "EC1A 1@@"),
            ("Manchester", None, "M1 1@@"),
            ("Bristol", None, "BS1 4@@"),
            ("Leeds", None, "LS1 2@@"),
        ),
    ),
    "EU": (
        "DE",
        ("{street}straße {n}", "{street}weg {n}", "{street}allee {n}"),
        (("Berlin", None, "101##"), ("Hamburg", None, "200##"), ("München", None, "803##"), ("Köln", None, "506##")),
    ),
    "IN": (
        "IN",
        ("{n} {street} Marg", "{n} {street} Road", "{n} {street} Lane"),
        (
            ("Mumbai", "MH", "4000##"),
            ("Pune", "MH", "4110##"),
            ("Bengaluru", "KA", "5600##"),
            ("New Delhi", "DL", "1100##"),
        ),
    ),
    "AU": (
        "AU",
        ("{n} {street} Street", "{n} {street} Road", "{n} {street} Parade"),
        (("Sydney", "NSW", "20##"), ("Melbourne", "VIC", "30##"), ("Brisbane", "QLD", "40##")),
    ),
    "NZ": (
        "NZ",
        ("{n} {street} Road", "{n} {street} Street", "{n} {street} Terrace"),
        (("Wellington", None, "60##"), ("Auckland", None, "10##")),
    ),
    "SG": (
        "SG",
        ("{n} {street} Road", "{n} {street} Avenue"),
        (("Singapore", None, "0189##"), ("Singapore", None, "2388##")),
    ),
    "MX": (
        "MX",
        ("Calle {street} {n}", "Avenida {street} {n}"),
        (("Ciudad de Mexico", "CDMX", "060##"), ("Guadalajara", "JAL", "441##"), ("Monterrey", "NL", "640##")),
    ),
    "JP": (
        "JP",
        ("{n}-{m} {street}", "{m}-{n} {street}"),
        (("Chiyoda", "Tokyo", "100-00##"), ("Minato", "Tokyo", "105-00##"), ("Osaka", "Osaka", "530-00##")),
    ),
    "PH": (
        "PH",
        ("{n} {street} Street", "{n} {street} Avenue"),
        (("Makati", "Metro Manila", "12##"), ("Quezon City", "Metro Manila", "11##"), ("Cebu City", "Cebu", "60##")),
    ),
}
"""Per delivery market (the keys personas.yaml and the store queue use; EU is a German address, GB not UK): the country
code, the street patterns and the places with a postal code template shaped like the market's real ones."""

TOKENS: Final = (
    ("card", "tok_visa_ok", "visa", "complete", "Pays by Visa card; the bank approves without a challenge."),
    ("card", "tok_mc_ok", "mastercard", "complete", "Pays by Mastercard; the bank approves without a challenge."),
    ("card", "tok_amex_ok", "amex", "complete", "Pays by Amex; the bank approves without a challenge."),
    ("card", "tok_visa_3ds_challenge", "visa", "complete", "Pays by Visa card; completes the 3-D Secure challenge."),
    ("card", "tok_visa_3ds_challenge", "visa", "abandon", "Pays by Visa card; abandons the 3-D Secure challenge."),
    ("card", "tok_visa_3ds_fail", "visa", "complete", "Pays by Visa card; the bank fails the 3-D Secure check."),
    ("card", "tok_visa_decline_insufficient_funds", "visa", "complete", "Pays by Visa card; declined for funds."),
    ("card", "tok_visa_decline_do_not_honor", "visa", "complete", "Pays by Visa card; the bank does not honour it."),
    ("google_pay", "tok_gpay_ok", "visa", "complete", "Pays with Google Pay only; approved without a challenge."),
    ("card", "tok_shopify_card_ok", "visa", "complete", "Pays by card through Shopify's handler; approved."),
)
"""The sandbox PSP's test tokens (carttrace.sandbox.core.psp.TOKENS), spread over the generated personas in this order."""

BUDGETS: Final[tuple[BudgetStyle, ...]] = ("tight", "normal", "generous")
DECISIVENESS: Final[tuple[Decisiveness, ...]] = ("fast", "deliberate")
DISCLOSURES: Final[tuple[Disclosure, ...]] = ("volunteers", "asks", "forgets", "mixed")
DELIVERIES: Final[tuple[DeliveryStyle, ...]] = ("standard", "express", "pickup")
ALLERGENS: Final = ("milk", "peanuts", "gluten", "tree nuts", "soy", "egg", "sesame", "lactose", "nickel", "fragrance")
DIETARY: Final = ("vegan", "vegetarian", "gluten-free", "dairy-free", "kosher", "halal", "organic", "low sugar")
CONDITIONS: Final = ("pregnant", "asthma", "takes warfarin", "breastfeeding", "high blood pressure")
MATERIALS: Final = ("wool", "polyester", "nickel", "latex", "down")
MEASUREMENT_SETS: Final = (
    {"chest": "88 cm", "waist": "74 cm", "foot": "25 cm"},
    {"chest": "102 cm", "waist": "88 cm", "foot": "28 cm"},
    {"bust": "94 cm", "hips": "100 cm", "foot": "24.5 cm"},
    {"chest": "110 cm", "waist": "96 cm", "foot": "29 cm"},
    {"chest": "95 cm", "waist": "80 cm", "height": "176 cm"},
    {"bust": "86 cm", "waist": "68 cm", "hips": "92 cm"},
)

LEGACY: Final = (
    ("card-ok", "tok_visa_ok", "visa", "card", "complete", "Pays by Visa card; the bank approves without a challenge."),
    (
        "card-3ds",
        "tok_visa_3ds_challenge",
        "visa",
        "card",
        "complete",
        "Pays by Visa card; the bank asks for 3-D Secure, the shopper completes it and the bank approves.",
    ),
    (
        "card-3ds-abandons",
        "tok_visa_3ds_challenge",
        "visa",
        "card",
        "abandon",
        "Pays by Visa card; the bank asks for 3-D Secure and the shopper abandons the challenge.",
    ),
    (
        "card-3ds-fails",
        "tok_visa_3ds_fail",
        "visa",
        "card",
        "complete",
        "Pays by Visa card; the shopper completes 3-D Secure and the bank fails the authentication.",
    ),
    (
        "card-declined",
        "tok_visa_decline_insufficient_funds",
        "visa",
        "card",
        "complete",
        "Pays by Visa card; the bank declines it for insufficient funds.",
    ),
    (
        "google-pay",
        "tok_gpay_ok",
        "visa",
        "google_pay",
        "complete",
        "Pays with Google Pay only; the bank approves without a challenge.",
    ),
)
"""The six personas of personas-v1, by slug: their emails, phones (555-0101 to 0106) and tokens are kept."""
SANDBOX_ADDRESSES: Final[Mapping[str, dict[str, str]]] = {
    "US": {
        "street_address": "1 Maple Way",
        "address_locality": "Springfield",
        "address_region": "IL",
        "postal_code": "62701",
        "address_country": "US",
    },
    "CA": {
        "street_address": "1 Maple Way",
        "address_locality": "Ottawa",
        "address_region": "ON",
        "postal_code": "K1P 1J1",
        "address_country": "CA",
    },
    "GB": {
        "street_address": "1 Maple Lane",
        "address_locality": "London",
        "postal_code": "EC1A 1BB",
        "address_country": "GB",
    },
    "EU": {
        "street_address": "Ahornweg 1",
        "address_locality": "Berlin",
        "postal_code": "10115",
        "address_country": "DE",
    },
    "IN": {
        "street_address": "1 Maple Marg",
        "address_locality": "Mumbai",
        "address_region": "MH",
        "postal_code": "400001",
        "address_country": "IN",
    },
    "AU": {
        "street_address": "1 Maple Street",
        "address_locality": "Sydney",
        "address_region": "NSW",
        "postal_code": "2000",
        "address_country": "AU",
    },
    "NZ": {
        "street_address": "1 Maple Street",
        "address_locality": "Wellington",
        "postal_code": "6011",
        "address_country": "NZ",
    },
    "SG": {
        "street_address": "1 Maple Road",
        "address_locality": "Singapore",
        "postal_code": "018989",
        "address_country": "SG",
    },
    "MX": {
        "street_address": "Calle Arce 1",
        "address_locality": "Ciudad de Mexico",
        "address_region": "CDMX",
        "postal_code": "06000",
        "address_country": "MX",
    },
    "JP": {
        "street_address": "1-1 Kaede",
        "address_locality": "Chiyoda",
        "address_region": "Tokyo",
        "postal_code": "100-0001",
        "address_country": "JP",
    },
    "PH": {
        "street_address": "1 Maple Street",
        "address_locality": "Makati",
        "address_region": "Metro Manila",
        "postal_code": "1226",
        "address_country": "PH",
    },
}
"""The one address per market of personas-v1, kept for `card-ok` (the default persona) so older tasks and tests
hold."""
NEW_PERSONAS: Final = 20
"""Personas added to the six of personas-v1."""
AUTHENTICATION_PERSONAS: Final = (
    (
        "card-3ds-frictionless",
        "tok_visa_3ds_frictionless",
        "visa",
        "card",
        "complete",
        "Pays by Visa card; the bank authenticates it with 3-D Secure without a challenge (frictionless) and approves.",
    ),
)
"""Personas after the generated ones (D71), so every 3-D Secure outcome has a persona: challenge passed (card-3ds),
frictionless (here), failed (card-3ds-fails) and abandoned (card-3ds-abandons). Appended, so no earlier persona
moves."""
PERSONA_ROW = tuple[str, str, str | None, str, str, str]
"""slug, token, brand (None for a wallet credential with no card network), kind, 3-D Secure behaviour, description."""
SHOP_PAY_PERSONAS: Final[tuple[PERSONA_ROW, ...]] = (
    ("shop-pay", "tok_shop_pay_ok", None, "shop_pay", "complete", "Pays with Shop Pay only; approved."),
    (
        "shop-pay-declined",
        "tok_shop_pay_declined",
        None,
        "shop_pay",
        "complete",
        "Pays with Shop Pay; the payment is declined (the issuer does not honour it).",
    ),
    (
        "shop-pay-expired",
        "tok_shop_pay_expired",
        None,
        "shop_pay",
        "complete",
        "Pays with Shop Pay; the minted Shop Token's validity has run out before the store processes it.",
    ),
)
"""Personas after the 3-D Secure ones (D73), so the Shop Pay rail's outcomes have personas: approved, declined, and
the expired-token path (the credential's minted token gets no validity window, carttrace.sandbox.core.psp). Appended, so no
earlier persona moves."""

HEADER: Final = """\
# Synthetic shoppers (decisions D13 and D20): who pays, how their bank answers, where their orders ship, and how they
# shop. Written by `carttrace.agents.personas.render_personas()` from word lists; `uv run carttrace agents personas` lists them and
# tests/agents/test_personas.py holds this file to the generator. Do not edit by hand.
#
# Nothing here is a real person or a real card:
# - names are a first name and an invented surname joined by position in two word lists;
# - addresses are fictional, one per delivery market, shaped like that market's real addresses;
# - emails are at example.com and phone numbers are in the 555-0100 to 555-0199 range reserved for fiction;
# - every payment instrument is an opaque test token the sandbox's PSP keys its behaviour on
#   (carttrace.sandbox.core.psp.TOKENS). No card number appears anywhere, test cards included.
#
# carttrace.agents.wallet loads this file and refuses an email outside example.com, a phone number outside the fictional range,
# or any value that looks like a card number. `style` and `constraints` are what task generation turns into hidden-spec
# variety (carttrace.agents.intents): budgets, what is stated or kept for asking, delivery, allergens, sizes and conditions.
"""


def generate_personas() -> dict[str, Any]:
    """The personas file as a mapping, deterministic for this module's lists and version."""
    personas: dict[str, Any] = {}
    for index, (slug, token, brand, kind, three_ds, description) in enumerate(LEGACY):
        personas[slug] = _persona(
            index,
            slug=slug,
            description=description,
            email=f"{slug}@example.com",
            phone=f"+1 202-555-01{index + 1:02d}",
            instruments=[{"kind": kind, "token": token, "brand": brand}],
            three_ds=three_ds,
            addresses=dict(SANDBOX_ADDRESSES) if slug == "card-ok" else None,
            legacy=slug == "card-ok",
        )
    for offset in range(NEW_PERSONAS):
        index = len(LEGACY) + offset
        kind, token, brand, three_ds, description = TOKENS[offset % len(TOKENS)]
        first, last = FIRST_NAMES[index % len(FIRST_NAMES)], LAST_NAMES[index % len(LAST_NAMES)]
        slug = f"{first}-{last}".lower().replace(" ", "-")
        instruments = [{"kind": kind, "token": token, "brand": brand}]
        if kind == "card" and offset % 4 == 1:
            instruments.append({"kind": "google_pay", "token": "tok_gpay_ok", "brand": "visa"})
        personas[slug] = _persona(
            index,
            slug=slug,
            description=description,
            email=f"{first}.{last}@example.com".lower(),
            phone=f"+1 202-555-01{index + 1:02d}",
            instruments=instruments,
            three_ds=three_ds,
            addresses=None,
            legacy=False,
        )
    for offset, (slug, token, brand, kind, three_ds, description) in enumerate(AUTHENTICATION_PERSONAS):
        index = len(LEGACY) + NEW_PERSONAS + offset
        personas[slug] = _persona(
            index,
            slug=slug,
            description=description,
            email=f"{slug}@example.com",
            phone=f"+1 202-555-01{index + 1:02d}",
            instruments=[{"kind": kind, "token": token, "brand": brand}],
            three_ds=three_ds,
            addresses=None,
            legacy=False,
        )
    for offset, (slug, credential, shop_brand, kind, three_ds, description) in enumerate(SHOP_PAY_PERSONAS):
        index = len(LEGACY) + NEW_PERSONAS + len(AUTHENTICATION_PERSONAS) + offset
        personas[slug] = _persona(
            index,
            slug=slug,
            description=description,
            email=f"{slug}@example.com",
            phone=f"+1 202-555-01{index + 1:02d}",
            instruments=[_instrument(kind, credential, shop_brand)],
            three_ds=three_ds,
            addresses=None,
            legacy=False,
            # the index would wrap the word lists onto a name an earlier persona holds
            name=f"{FIRST_NAMES[offset]} {LAST_NAMES[offset + 2]}",
        )
    for persona in personas.values():
        persona["email"] = _neutral_email(persona)
    return {"schema": "mc.personas.v2", "version": PERSONAS_VERSION, "default": "card-ok", "personas": personas}


def _neutral_email(persona: Mapping[str, Any]) -> str:
    """The persona's address from its fictional name, never its slug: the slug names its designed outcome
    (card-declined, card-3ds-fails), and the agent reads the address in the buyer profile before it pays, so a
    designed failure was not blind (the v1 review's #33)."""
    name = persona.get("name")
    if not isinstance(name, str) or not name.strip():
        return str(persona["email"])
    return ".".join(name.lower().split()) + "@example.com"


def _instrument(kind: str, token: str, brand: str | None) -> dict[str, str]:
    out = {"kind": kind, "token": token}
    if brand is not None:
        out["brand"] = brand
    return out


def _persona(
    index: int,
    *,
    slug: str,
    description: str,
    email: str,
    phone: str,
    instruments: list[dict[str, str]],
    three_ds: str,
    addresses: dict[str, dict[str, str]] | None,
    legacy: bool,
    name: str | None = None,
) -> dict[str, Any]:
    rng = seeded_rng("persona", PERSONAS_VERSION, slug)
    if name is None:
        name = f"{FIRST_NAMES[index % len(FIRST_NAMES)]} {LAST_NAMES[index % len(LAST_NAMES)]}"
    # D73: every shopper can pay a store's Shop Pay handler, so a shop_pay rail row is payable whoever the task's
    # persona is; the Shop Pay personas of SHOP_PAY_PERSONAS hold their own variant instead.
    if not any(i["kind"] == "shop_pay" for i in instruments):
        instruments = [*instruments, {"kind": "shop_pay", "token": "tok_shop_pay_ok"}]
    out: dict[str, Any] = {
        "description": description,
        "name": name,
        "email": email,
        "phone": phone,
        "instruments": instruments,
        "three_ds": three_ds,
        "addresses": addresses if addresses is not None else {m: _address(m, rng) for m in MARKET_SHAPES},
        "style": _style(index, legacy=legacy),
        "constraints": _constraints(index, legacy=legacy),
    }
    return out


def _style(index: int, *, legacy: bool) -> dict[str, str]:
    """The first generator's draws for `card-ok`; a spread over every value for the others, each combination once."""
    if legacy:
        return {"budget": "normal", "decisiveness": "fast", "disclosure": "mixed", "delivery": "standard"}
    return {
        "budget": BUDGETS[index % len(BUDGETS)],
        "decisiveness": DECISIVENESS[(index // len(BUDGETS)) % len(DECISIVENESS)],
        "disclosure": DISCLOSURES[index % len(DISCLOSURES)],
        "delivery": DELIVERIES[(index // 2) % len(DELIVERIES)],
    }


def _constraints(index: int, *, legacy: bool) -> dict[str, Any]:
    """One to three needs a persona carries into a category: by position, so every value of every list appears."""
    if legacy:
        return {}
    out: dict[str, Any] = {}
    if index % 2 == 0:
        out["allergens"] = [ALLERGENS[index % len(ALLERGENS)]]
    if index % 3 == 1:
        out["dietary"] = [DIETARY[index % len(DIETARY)]]
    if index % 2 == 1 or index % 5 == 0:
        out["measurements"] = dict(MEASUREMENT_SETS[index % len(MEASUREMENT_SETS)])
    if index % 4 == 3:
        out["conditions"] = [CONDITIONS[index % len(CONDITIONS)]]
    if index % 5 == 2:
        out["avoid_materials"] = [MATERIALS[index % len(MATERIALS)]]
    return out


def _address(market: str, rng: Random) -> dict[str, str]:
    country, patterns, places = MARKET_SHAPES[market]
    locality, region, template = rng.choice(places)
    street = rng.choice(patterns).format(n=rng.randint(2, 180), m=rng.randint(1, 9), street=rng.choice(STREETS))
    out = {"street_address": street, "address_locality": locality}
    if region is not None:
        out["address_region"] = region
    out["postal_code"] = _postal(template, rng)
    out["address_country"] = country
    return out


def _postal(template: str, rng: Random) -> str:
    letters = "ABCDEFGHJKLMNPRSTUWXYZ"
    return "".join(str(rng.randint(0, 9)) if c == "#" else rng.choice(letters) if c == "@" else c for c in template)


def render_personas() -> str:
    """data/agent/personas.yaml's text: the header comment and the generated document, in a fixed key order."""
    body = yaml.safe_dump(generate_personas(), sort_keys=False, allow_unicode=True, width=120)
    return HEADER + body
