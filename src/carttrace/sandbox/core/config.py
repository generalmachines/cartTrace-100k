"""What the core needs to know about its environment: the market row, the discount policy, the trust defaults, the
reservation window, the payment handlers it serves, the profile it runs under and the sealed fault rules.

An environment pack (`carttrace.sandbox.envpack`) builds one; the core never reads a file or a protocol to get it, so `apply`
stays a pure function of (state, command, config).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any

from carttrace.core.canon import sha256_hex
from carttrace.sandbox.core.market import Market

FAKE_NAMESPACE = "dev.ucp.sandbox.fake"
"""The sandbox's own handler: it needs no credential and takes every test token."""
FAKE_HANDLERS: Mapping[str, str] = MappingProxyType({FAKE_NAMESPACE: FAKE_NAMESPACE, "fake": FAKE_NAMESPACE})
"""Handler id to namespace for the fake handler, under both names adapters have used."""
DELEGATED_NAMESPACE = "dev.sandbox.psp.delegated"
"""The PSP's handler for credentials it vaulted with an allowance (`vault_credential`): it takes nothing else."""
REQUESTED_NAMESPACE = "dev.sandbox.psp.requested"
"""The PSP's handler for payments that answer a payment request (`request_payment`): every payment names one."""
SHOP_PAY_NAMESPACE = "dev.shopify.shop_pay"
"""Shop Pay's UCP payment handler namespace (vendor/dev.shopify.shop_pay/2026-04-08/spec.md)."""
SHOP_PAY_HANDLER_ID = "shop_pay"
"""The handler id the Shop Pay spec's examples declare, and the id the sandbox's own entry takes when a profile
adopts the handler without the store declaring one (the spec's `handler_id` must match the declared id)."""


def adopted_shop_id(world: str) -> str:
    """The `shop_id` of the sandbox's own Shop Pay handler entry, served where a store adopted the handler (D73); a
    store that declared Shop Pay itself keeps its recorded one (D70). An 11-digit number, the shape Shopify's own shop
    ids take, read from the world's pack id, so no two worlds share one and nothing in it says test or sandbox (the
    v1 release showed every store `sandbox-shop-1`). It is never looked up anywhere: Shop Tokens are minted offline."""
    return str(10_000_000_000 + int(sha256_hex(f"shop_pay:{world}")[:15], 16) % 90_000_000_000)


CLAMPING_PLATFORMS = frozenset({"shopify", "vendure"})
"""Store platforms that add a line over the stock at the stock (with a warning) instead of refusing it: live Shopify
(var/live/delta-01/report.md section 4, store-310 ep-ad499feca1d4). The core's own table, since the core reads no
surface; carttrace.sandbox.platforms reads its Shopify row from it."""

COUPON_KINDS = ("percent", "fixed_cart", "free_shipping")
"""What a coupon takes off: a whole percentage of the merchandise, a fixed amount off the cart, or the shipping (a
free rate of its own). WooCommerce's three kinds as the reference store's coupons use them
(ops/playgrounds/woocommerce/seed.py; captures/diff-woo 62, 64, 70)."""


@dataclass(frozen=True, slots=True)
class Coupon:
    """One discount code's rule. `policy.discount_codes` (code to percent) is shorthand for a percent coupon with no
    conditions; `policy.coupons` holds the rest. The rules are the store's data, honoured on every platform; how
    codes combine and in what order they apply is the platform's (`CoreRules`)."""

    code: str
    kind: str = "percent"
    amount: int = 0
    """Percent off for a percent coupon (0 to 100); minor units off for a fixed_cart one; 0 for free_shipping."""
    individual_use: bool = False
    """Applying it removes every code already applied, and no code may join it afterwards (WooCommerce individual
    use: captures/diff-woo 65 and 66)."""
    min_spend: int | None = None
    """The merchandise subtotal, in minor units, the cart must reach (WooCommerce minimum spend: diff-woo 67)."""
    priority: int = 0
    """Lower applies first on a platform that orders rules (Magento `sort_order`)."""
    stop: bool = False
    """No lower-priority rule applies after this one, on a platform that orders rules (Magento
    `stop_rules_processing`)."""


@dataclass(frozen=True, slots=True)
class CoreRules:
    """What a store's platform changes in the core's own arithmetic and states, as flags. Every default is the
    sandbox's plain behaviour, so a platform with no row here is the sandbox as it always was. The surfaces' side of a
    platform (wire shapes, wording) is carttrace.sandbox.platforms; this is the part `apply` must see, kept in the core
    because the core reads no surface."""

    clamp_stock: bool = False
    """A line over the stock is added at the stock with a warning (live Shopify, see CLAMPING_PLATFORMS)."""
    stack_codes: bool | None = None
    """Whether discount codes combine: True on WooCommerce (WELCOME10 then FIVEOFF both apply, diff-woo 64), False
    where one code is the rule, None to leave it to the pack's `policy.stack_discounts`."""
    ordered_rules: bool = False
    """Codes apply by `Coupon.priority` (then as sent), and a `stop` rule ends the run (Magento SalesRule)."""
    separate_free_rate: bool = False
    """Free shipping is a rate of its own beside the paid ones, never a paid rate zeroed (WooCommerce: "Standard (free
    from 50.00)" at 0.00 listed after Standard 5.99, diff-woo 74 and 75)."""
    free_after_discount: bool = False
    """A free-shipping threshold is compared with the merchandise after discounts (WooCommerce free shipping without
    `ignore_discounts`; Magento OfflineShipping Freeshipping.php#L73-L81)."""
    preselect_first_rate: bool = False
    """The first rate of a group is selected until the shopper picks another (WooCommerce: Standard selected even
    when the free rate is listed, diff-woo 74, 82, 83)."""
    rates_before_address: bool = False
    """Rates are quoted for the store's own country before any address is given (diff-woo 82 and 83)."""
    unshippable_no_rates: bool = False
    """A destination the store does not ship to is kept, with no rates, instead of refused (diff-woo 84: Canada, 200
    and an empty rate list)."""
    refuse_bad_line: bool = False
    """A cart change with a line the store refuses (out of stock, more than the stock) is refused whole and the cart
    keeps its lines (WooCommerce 400 with the cart unchanged, diff-woo 41, 42, 44, 57)."""
    on_hold_methods: frozenset[str] = frozenset()
    """Offline payment methods whose orders wait on the merchant (`on_hold`) instead of being confirmed (WooCommerce
    bank transfer and cheque, diff-woo 112, 117; cash on delivery goes to processing, 104)."""
    options_required: bool = False
    """A line naming a product with options, not one of its variants, is refused as `options_required` (Magento
    ConfigurableProduct Configurable.php#L1022: "You need to choose options for your item.")."""
    free_threshold_with_tax: bool = False
    """A free-shipping threshold counts the tax on the merchandise (Medusa's `item_total` price rule: 49.99 is free
    from 50.00 at 8%, ops/playgrounds/medusa/captures/store-api 87 and 88)."""
    tax_upfront: bool = False
    """Tax is charged from the first line, carts included, at the market's rate, before any address or rate is
    chosen (Medusa taxes by region: store-api 26, a cart of 79.99 taxed 6.40 on create)."""
    tax_on_shipping: bool = False
    """The shipping rate is taxed at the market's rate too (Medusa: Express 14.99 adds 1.20 tax, store-api 73)."""
    codes_case_sensitive: bool = False
    """A code is matched exactly as the store wrote it (Medusa: `welcome10` is invalid where `WELCOME10` applies,
    store-api 44)."""
    silent_repeat_codes: bool = False
    """A code applied again is a quiet no-op, not refused (Medusa: WELCOME10 twice answers 200 with one promotion,
    store-api 43)."""
    default_shipping_group: bool = False
    """A checkout always has one shipping group holding every line, so its rates show before the shopper names a
    method (Medusa's UCP plugin: one `shipping` method and one `default` group whenever there are items,
    ucp-fulfillment.js; captures/ucp-2026-08-25 14)."""
    code_replaces: bool = False
    """One code per order, and a new code takes the place of the one applied rather than being refused (Saleor:
    FIVEOFF after WELCOME10 leaves FIVEOFF, SOLO15 after it leaves SOLO15, ops/playgrounds/saleor/captures/storefront
    36, 42)."""
    compound_codes: bool = False
    """Stacked codes apply one after another, each to what the codes before it left (Medusa: WELCOME10 then BIG20
    take 28% of 159.98, not 30%, store-api 48)."""
    cheapest_on_address: bool = False
    """Once a shipping group has a destination and no chosen rate, the cheapest rate is chosen for it (the plugin's
    update-checkout-session workflow, step 3; ucp-2026-08-25 16: Standard selected after the address)."""


PLAIN_RULES = CoreRules()
OFFLINE_METHODS = ("bank_transfer", "cheque")
"""Payment methods WooCommerce's offline gateways take (direct bank transfer, cheque): credential free, as cash on
delivery is."""
CORE_RULES: Mapping[str, CoreRules] = MappingProxyType(
    {
        "shopify": CoreRules(clamp_stock=True),
        "woocommerce": CoreRules(
            stack_codes=True,
            separate_free_rate=True,
            free_after_discount=True,
            preselect_first_rate=True,
            rates_before_address=True,
            unshippable_no_rates=True,
            refuse_bad_line=True,
            on_hold_methods=frozenset(OFFLINE_METHODS),
        ),
        "magento": CoreRules(stack_codes=False, ordered_rules=True, free_after_discount=True, options_required=True),
        "saleor": CoreRules(
            stack_codes=False,
            code_replaces=True,
            silent_repeat_codes=True,
            codes_case_sensitive=True,
            refuse_bad_line=True,
            tax_upfront=True,
        ),
        "vendure": CoreRules(
            clamp_stock=True,
            stack_codes=True,
            silent_repeat_codes=True,
            codes_case_sensitive=True,
            rates_before_address=True,
        ),
        "prestashop": CoreRules(
            stack_codes=True,
            refuse_bad_line=True,
            free_after_discount=True,
            free_threshold_with_tax=True,
            on_hold_methods=frozenset(OFFLINE_METHODS),
        ),
        "medusa": CoreRules(
            stack_codes=True,
            free_after_discount=True,
            free_threshold_with_tax=True,
            rates_before_address=True,
            refuse_bad_line=True,
            tax_upfront=True,
            tax_on_shipping=True,
            codes_case_sensitive=True,
            silent_repeat_codes=True,
            compound_codes=True,
            default_shipping_group=True,
            cheapest_on_address=True,
        ),
    }
)
"""Each platform's core rules, by the name carttrace.sandbox.platforms gives it. The WooCommerce row cites the reference
store's recorded probes (ops/playgrounds/woocommerce/captures/diff-woo); the Magento row the code reading in
docs/SANDBOX-GAPS.md (not run, so unverified on the wire); the Medusa row the Medusa playground's recorded probes
(ops/playgrounds/medusa/captures, `carttrace.sandbox.platforms.medusa.EVIDENCE`, D68); the Vendure and PrestaShop rows
the code reading at pinned tags (`carttrace.sandbox.platforms.vendure`, `.prestashop`, D68, unverified on the wire); the
Saleor row the Saleor playground's recorded probes (ops/playgrounds/saleor/captures, D68)."""


def core_rules(platform: str | None) -> CoreRules:
    """The core rules of a platform name; the plain sandbox's for any other or none."""
    return CORE_RULES.get((platform or "").lower(), PLAIN_RULES)


def clamps_stock(platform: str | None) -> bool:
    return (platform or "").lower() in CLAMPING_PLATFORMS


@dataclass(frozen=True, slots=True)
class FaultRule:
    """One scheduled fault, as data. `carttrace.sandbox.faults` builds these from envpack knobs; the core interprets them.

    It fires on a command when every trigger it names matches: the command type is in `on`, the command is number
    `at_seq` or later, more than `after` commands have run, and the command touches `variant`. It fires at most
    `times` times (None: every time it matches). An `at_start` rule fires on no command: it changes the world once,
    as the episode opens, before anything can read it (a task's sealed stockout), and names no other trigger.
    """

    id: str
    kind: str
    on: tuple[str, ...] = ()
    at_seq: int | None = None
    after: int | None = None
    variant: str | None = None
    times: int | None = 1
    params: Mapping[str, Any] = field(default_factory=dict)
    at_start: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "kind": self.kind,
            "on": list(self.on),
            "at_seq": self.at_seq,
            "after": self.after,
            "variant": self.variant,
            "times": self.times,
            "params": dict(self.params),
            "at_start": self.at_start,
        }


@dataclass(frozen=True, slots=True)
class ThreeDs:
    """3-D Secure as a plain accept or decline (D111): the issuer authenticates without a challenge (EMV 3DS
    frictionless) and approves, or authenticates and declines (transStatus N), as the token or a due three_ds fault
    scripted it. No challenge page is served and no buyer step is owed."""

    rail: str | None = None
    """The rail the row pays over (ucp_psp, acp_delegated, ap2_mandates, mpp; google_pay for a Google Pay payment
    over UCP), recorded on each decision."""
    abandon: bool = False
    """The buyer would not complete authentication (a persona built to abandon the challenge): a payment that would
    have met a challenge it passes is declined instead, since no frictionless authentication can stand in for it."""


@dataclass(frozen=True, slots=True)
class Config:
    market: Market
    discount_codes: Mapping[str, int] = field(default_factory=dict)
    """Code -> percent off. Codes are upper-case here."""
    stack_discounts: bool = False
    default_tier: str = "anonymous"
    default_permissions: tuple[str, ...] = ()
    reservation_ttl_ms: int = 15 * 60 * 1000
    tick_ms: int = 1000
    handlers: Mapping[str, str] = FAKE_HANDLERS
    """Handler id -> namespace for every payment handler this store serves. Anything else escalates to a human."""
    faults: tuple[FaultRule, ...] = ()
    profile_id: str | None = None
    """The pack's ProtocolProfile id, recorded on every session that does not name its own."""
    clamp_stock: bool = False
    """A line asking for more than the stock is added at the stock (with a `line.clamped` event) rather than refused,
    as live Shopify does (var/live/delta-01/report.md section 4, `merchandise_not_enough_stock`). Set from the
    store's platform (carttrace.sandbox.platforms)."""
    coupons: Mapping[str, Coupon] = field(default_factory=dict)
    """Code to rule, upper-case: the percent shorthand of `discount_codes` and the pack's typed coupons. Empty falls
    back to `discount_codes` alone."""
    rules: CoreRules = PLAIN_RULES
    """The platform's core rules (CORE_RULES), set from the pack's platform."""
    sold_out: frozenset[str] = frozenset()
    psp: str | None = None
    """The payment broker behind the store's PSP (D71, `carttrace.sandbox.core.brokers`): stripe, adyen, braintree or paypal,
    whose words a decline's details carry; None for the sandbox's own PSP."""
    three_ds: ThreeDs | None = None
    """How the PSP answers a payment its issuer wants 3-D Secure for: None parks it on a
    challenge page the buyer answers (D71); a `ThreeDs` decides at once, authenticated and approved or authenticated
    and declined, with no page to complete."""
    shop_id: str | None = None
    """The Shop Pay merchant id (`shop_id` of the store's declared `dev.shopify.shop_pay` handler entry, or the
    sandbox's own where the handler is adopted, D73); None where the store serves no Shop Pay handler. A Shop Token
    mints only for this merchant: the spec scopes a token to its checkout and merchant."""
    """Variants a cart write is refused as sold out (`line.rejected`, reason `sold_out`) though the catalog lists them
    available, as live stores answer (D52): the episode's draw, from the pack's `policy.sold_out`."""

    def coupon_table(self) -> Mapping[str, Coupon]:
        """Every code's rule: the typed coupons, else each percent code of `discount_codes` as a plain coupon."""
        if self.coupons:
            return self.coupons
        return {code: Coupon(code, "percent", int(pct)) for code, pct in self.discount_codes.items()}

    @property
    def stacks(self) -> bool:
        """Whether codes combine: the platform's rule when it has one, else the pack's policy."""
        return self.rules.stack_codes if self.rules.stack_codes is not None else self.stack_discounts
