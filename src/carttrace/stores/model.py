"""The typed store record: every field of the queue and evidence tabs, plus the few values derived from them.

Why typed. The queue is a people-edited sheet whose cells are free text ("US (likely)", "Unclear (bot wall)",
"None found"). Code downstream needs decisions, not text: which platform throttle applies, which currency a budget is
in, which split a store belongs to. Each label is kept exactly as the sheet shows it, and next to it a typed key read
from it by a small table. An unknown label maps to "other" (or None) and is recorded in `warnings` instead of
failing, because a person adding a new platform name to the sheet must not stop collection.
"""

from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict

from carttrace.stores.text import collapse_ws, fold

StatusKey = Literal["observed", "access_caveat", "scope_note", "review", "fit_review", "other"]
PlatformKey = Literal[
    "shopify",
    "custom",
    "sfcc",
    "unclear",
    "magento",
    "bigcommerce",
    "sap_commerce",
    "woocommerce",
    "godaddy",
    "vtex",
    "square",
    "shopware",
    "other",
    "unknown",
]
"""`unknown` is a store added by hand (carttrace.stores.added) before a preflight confirms its platform; the live throttle
and the rights table read it as a platform that says nothing about whose terms apply."""
Split = Literal["train", "validation", "test"]
MerchantBasis = Literal["parent", "domain"]
AddedMode = Literal["sandbox", "live"]
"""How a store added by hand may be tested: in the sandbox only, or live too once every live gate is open."""

STATUS_KEYS: dict[str, StatusKey] = {
    "observed": "observed",
    "access caveat": "access_caveat",
    "scope note": "scope_note",
    "review": "review",
    "fit review": "fit_review",
}
PLATFORM_KEYS: dict[str, PlatformKey] = {
    "shopify": "shopify",
    "custom / in-house": "custom",
    "salesforce commerce cloud": "sfcc",
    "unclear (bot wall)": "unclear",
    "magento / adobe commerce": "magento",
    "bigcommerce": "bigcommerce",
    "sap commerce": "sap_commerce",
    "woocommerce": "woocommerce",
    "godaddy": "godaddy",
    "vtex": "vtex",
    "square online / weebly": "square",
    "shopware": "shopware",
}
_PLATFORM_PREFIXES: tuple[tuple[str, PlatformKey], ...] = (
    ("shopify", "shopify"),
    ("salesforce", "sfcc"),
    ("magento", "magento"),
    ("adobe commerce", "magento"),
    ("unclear", "unclear"),
    ("custom", "custom"),
)
MARKETS: dict[str, tuple[str | None, str | None]] = {
    "us": ("US", "USD"),
    "uk": ("GB", "GBP"),
    "europe (eur)": ("EU", "EUR"),
    "india": ("IN", "INR"),
    "canada": ("CA", "CAD"),
    "japan": ("JP", "JPY"),
    "australia": ("AU", "AUD"),
    "uae": ("AE", "AED"),
    "singapore": ("SG", "SGD"),
    "global": (None, None),
}
"""Sheet market label to (ISO 3166-1 alpha-2 code, ISO 4217 currency). "EU" stands for the euro area; "Global" has
neither, because the shopper's market is chosen per episode."""


class Store(BaseModel):
    """One row of the store queue joined with its evidence row. Sheet labels are kept verbatim."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    # Store queue tab
    id: str
    """Queue id, e.g. "store-001". Stable across pulls; the sync addresses rows by it."""
    category: str
    name: str
    website: str
    """The queue's link text, a host such as "store-001.example"."""
    platform: str
    """Platform as detected from public page code; "Unclear (bot wall)" when a bot wall blocked the check."""
    market: str
    """Delivery market as the sheet states it, e.g. "UK" or "US (likely)"."""
    parent_company: str | None
    """None when the sheet says "None found", which is not proof the store is independent."""
    status: str
    what_we_saw: str
    what_to_do_next: str
    test_fixture: str | None
    last_observed: date | None

    # Evidence tab (None when the store has no evidence row)
    category_index: int | None
    url: str | None
    source_status: str | None
    checked_at: date | None
    fit_reason: str | None
    observed_evidence: str | None
    parent_group: str | None
    evidence_url: str | None
    canonical_host: str | None
    duplicate_host_count: int | None
    publication_permission: str | None
    eligibility: str | None
    platform_evidence: str | None
    delivery_market: str | None
    sheet_merchant_group_id: str | None
    """What the sheet's "Merchant group ID" cell holds now; `merchant_group_id` is what we compute."""
    browser_preflight_result: str | None
    task_fixture: str | None
    collector_eligibility: str | None
    sheet_split: str | None
    """What the sheet's "Split" cell holds now; `split` is what we compute."""
    gate_approval_reference: str | None

    # Derived, deterministic
    category_slug: str
    status_key: StatusKey
    platform_key: PlatformKey
    market_code: str | None
    market_likely: bool
    """True when the sheet only says the market is likely ("US (likely)")."""
    currency: str | None
    """ISO 4217 currency of the delivery market, for budgets in hidden specs."""
    domain: str
    """Canonical host: lowercase, no scheme, no "www."."""
    registrable_domain: str
    merchant_key: str
    """What the merchant group is keyed by: the normalised parent company, or the registrable domain."""
    merchant_basis: MerchantBasis
    merchant_group_id: str
    split: Split
    warnings: tuple[str, ...] = ()
    """Data-quality notes from loading: tabs that disagree, labels no table knows, values that did not parse."""
    added_mode: AddedMode | None = None
    """None for a store of the team sheet; the mode a store added by hand was added in (carttrace.stores.added)."""


def status_key(label: str) -> StatusKey | None:
    return STATUS_KEYS.get(_key(label))


def platform_key(label: str) -> PlatformKey | None:
    key = _key(label)
    if key in PLATFORM_KEYS:
        return PLATFORM_KEYS[key]
    for prefix, value in _PLATFORM_PREFIXES:
        if key.startswith(prefix):
            return value
    return None


def market_of(label: str) -> tuple[str | None, str | None, bool] | None:
    """(country code, currency, likely) for a sheet market label, or None when the label is not in the table."""
    key = _key(label)
    likely = key.endswith("(likely)")
    key = key.removesuffix("(likely)").strip()
    if key not in MARKETS:
        return None
    code, currency = MARKETS[key]
    return code, currency, likely


def _key(label: str) -> str:
    return collapse_ws(fold(label))
