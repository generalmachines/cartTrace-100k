"""What carts and checkouts share: building lines against stock, repricing, discount codes, buyer details, and making
a new cart.

Two audit bugs live here. A line id the store had returned could not be sent back (the old store deleted the line);
now a returned line id keeps its line, and an id the store never issued is refused out loud with a `line.rejected`
event. And a repeated discount code stacked; now `pricing.resolve_codes` applies a code once.

A variant the episode's draw sold out (D52, `Config.sold_out`) is refused at every cart write with reason
`sold_out`, whatever its shelf holds, as live stores refuse variants their catalog lists available.

Platforms (D42) change two things here, each a `CoreRules` flag: a store that refuses a bad line refuses the whole
change and keeps the cart as it was (WooCommerce answers 400 and leaves the cart alone), and a store whose products
need their options chosen refuses a line naming the product itself (Magento's configurable products).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace

from carttrace.sandbox.core import pricing
from carttrace.sandbox.core.commands import Buyer, LineIn
from carttrace.sandbox.core.context import Ctx
from carttrace.sandbox.core.fault_effects import peek
from carttrace.sandbox.core.state import Cart, Line, Totals


def build_lines(
    ctx: Ctx, existing: tuple[Line, ...], inputs: Sequence[LineIn], owner: str, *, held: Mapping[str, int] | None = None
) -> tuple[Line, ...]:
    """A full replacement of `existing` by `inputs`. A line id the store returned keeps that line; a line id the store
    never issued is refused out loud; stock is checked against what this owner already holds plus what is free."""
    by_id = {ln.id: ln for ln in existing}
    by_variant = {ln.variant_id: ln for ln in existing}
    out: dict[str, Line] = {}
    used_ids: set[str] = set()
    refused = False
    for li in inputs:
        ex = by_id.get(li.line_id) if li.line_id is not None else None
        if li.line_id is not None:
            reason = _line_ref_problem(ex, li)
            if reason is not None:
                ctx.emit("line.rejected", owner=owner, line_id=li.line_id, variant_id=li.variant_id, reason=reason)
                refused = True
                continue
        vid = ex.variant_id if ex else str(li.variant_id)
        if ex is None and vid in by_variant and by_variant[vid].id not in used_ids:
            ex = by_variant[vid]
        if li.quantity == 0:
            continue
        v = ctx.tx.inventory.get(vid)
        if v is None:
            reason = "options_required" if ctx.cfg.rules.options_required and _has_options(ctx, vid) else "not_found"
            ctx.emit("line.rejected", owner=owner, variant_id=vid, reason=reason)
            refused = True
            continue
        qty = li.quantity + (out[vid].quantity if vid in out else 0)
        if vid in ctx.cfg.sold_out:
            # The store's own answer, not the shelf: the catalog still lists the variant available (D52).
            ctx.emit("line.rejected", owner=owner, variant_id=vid, reason="sold_out", requested=qty)
            refused = True
            continue
        free = v.sellable + (held or {}).get(vid, 0) if v.available else 0
        if ctx.cfg.clamp_stock and v.available and 0 < free < qty:
            ctx.emit("line.clamped", owner=owner, variant_id=vid, requested=qty, added=free)
            qty = free
        if not v.available or free < qty:
            ctx.emit(
                "line.rejected",
                owner=owner,
                variant_id=vid,
                reason="out_of_stock",
                requested=qty,
                available=max(0, free),
            )
            refused = True
            continue
        line_id = out[vid].id if vid in out else (ex.id if ex else ctx.new_id("line"))
        used_ids.add(line_id)
        out[vid] = Line(line_id, vid, qty, v.price)
    if refused and ctx.cfg.rules.refuse_bad_line:
        ctx.emit("line.change_refused", owner=owner, kept=len(existing))
        return existing
    for ln in existing:
        if ln.id not in used_ids:
            ctx.emit("line.removed", owner=owner, line_id=ln.id, variant_id=ln.variant_id)
    return tuple(out.values())


def _has_options(ctx: Ctx, product_id: str) -> bool:
    """Whether an id names a product sold as several variants (a configurable product), not a variant."""
    return sum(1 for v in ctx.tx.inventory.all().values() if v.product_id == product_id) > 1


def _line_ref_problem(ex: Line | None, li: LineIn) -> str | None:
    if ex is None:
        return "line_not_found"
    if li.variant_id is not None and li.variant_id != ex.variant_id:
        return "line_variant_mismatch"
    return None


def reprice(ctx: Ctx, lines: tuple[Line, ...]) -> tuple[Line, ...]:
    """The same lines at today's prices."""
    return tuple(replace(ln, unit_price=ctx.variant(ln.variant_id).price) for ln in lines)


def apply_codes(ctx: Ctx, requested: Sequence[str], owner: str, *, lines: Sequence[Line] = ()) -> tuple[str, ...]:
    """The codes that apply to `lines`, each refusal and application logged. A minimum spend is checked against the
    lines' merchandise subtotal when the codes are applied."""
    applied, refusals = pricing.resolve_codes(
        requested,
        ctx.cfg.discount_codes,
        stackable=ctx.cfg.stacks,
        refused=bool(peek(ctx, "refuse_codes")),
        coupons=ctx.cfg.coupon_table(),
        subtotal=pricing.subtotal(lines),
        ordered=ctx.cfg.rules.ordered_rules,
        case_sensitive=ctx.cfg.rules.codes_case_sensitive,
        silent_repeat=ctx.cfg.rules.silent_repeat_codes,
        replaces=ctx.cfg.rules.code_replaces,
    )
    table = ctx.cfg.coupon_table()
    # Facts only a typed coupon can give, so a pack of percent codes logs exactly the events it always did.
    alone = next((c for c in applied if c in table and table[c].individual_use), None)
    for code, why in refusals:
        rule = table.get(code.strip().upper())
        spend = rule.min_spend if rule is not None and why == "minimum_not_met" else None
        blocker = alone if why == "not_combinable" else None
        if spend is None and blocker is None:
            ctx.emit("discount.rejected", owner=owner, code=code, reason=why)
        else:
            ctx.emit("discount.rejected", owner=owner, code=code, reason=why, min_spend=spend, with_code=blocker)
    for code in applied:
        ctx.emit("discount.applied", owner=owner, code=code)
    return applied


def merge_buyer(old: Mapping[str, str] | None, new: Buyer | None) -> dict[str, str]:
    out = dict(old or {})
    if new is not None:
        out.update({k: v for k, v in new.model_dump().items() if v is not None})
    return out


def cart_totals(ctx: Ctx, cart: Cart) -> Cart:
    totals = pricing.totals(
        cart.lines,
        cart.codes,
        ctx.cfg.discount_codes,
        ctx.cfg.market,
        coupons=ctx.cfg.coupon_table(),
        rules=ctx.cfg.rules,
    )
    return replace(cart, totals=totals)


def new_cart(
    ctx: Ctx,
    session: str,
    *,
    lines: Sequence[LineIn],
    codes: Sequence[str] = (),
    buyer: Buyer | None = None,
    country: str | None = None,
) -> Cart:
    """Make, price and store a new active cart for `session`, and name it in the outcome's refs."""
    cid = ctx.new_id("cart")
    built = build_lines(ctx, (), lines, cid)
    cart = Cart(cid, session, "active", built, (), merge_buyer({}, buyer), country, Totals(), ctx.now)
    cart = cart_totals(ctx, replace(cart, codes=apply_codes(ctx, codes, cid, lines=built)))
    ctx.tx.carts.put(cid, cart)
    ctx.emit("cart.created", id=cid, session=session, lines=len(built))
    ctx.refs["cart_id"] = cid
    return cart
