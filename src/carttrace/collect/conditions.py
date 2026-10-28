"""The conditions a manifest row runs under, and how each one changes the episode.

Why conditions are data. The benchmark's claims come from pairs: the same store and task under two conditions that
differ in one thing. A condition is a name and its parameters in the trace header (`carttrace.trace.Condition`), so a reader
can pair episodes without knowing how they were made:
- `baseline`: the store as it is;
- `cross_border`: the shopper's delivery market is another market than the store's own (the first of the plan's
  preferred markets that differs), and a sandbox store is priced, taxed and shipped for that market;
- `version_variant`: the store's protocol profile moved to another UCP release (a preset), with payment handlers, MCP
  and every other family unchanged, so the comparison isolates the UCP release. Only the sandbox can speak a version
  a store does not, so live rows never take it;
- `detail_hidden`: details a category's checklist hypotheses expect (size charts, fabric, care, policies) are removed
  from what our agent reads. A hypothesis that names catalog details (`details`: allergens, size_chart, care...)
  hides exactly those, wherever the agent would read them, so the lift is what that one detail is worth; its
  locations still say whether policy answers are filtered too. A hypothesis that names none hides its locations
  whole, as before. Hypotheses that hide the same details in the same locations are one experiment, named together,
  because hiding the same thing twice measures the same thing twice. Options are never hidden: without them the
  agent cannot choose a variant at all, which measures a broken store rather than a missing detail. A row takes the
  condition only when it hides something from its own task (D47, `hides_decisive`): the experiment names the task's
  decisive detail and at least one of the task's products states it; any other detail_hidden row is a baseline run
  under a wrong label and is not planned;
- `checkout`: the journey goes on from the verified cart to an order (D13), paying as a persona from
  data/agent/personas.yaml after the shopper approves the total. Its parameters `{"checkout": true, "persona": name}`
  are the judge's convention for an episode that authorises an order (carttrace.judge.safety). Sandbox and reference rows
  only: a live journey ends at the cart (D3), so a live row never takes it and a plan that asks for both is refused;
- `detail_added` (D19): detail_hidden with the arms reversed. A store's data inventory (`carttrace live inventory`) holds
  the values its own pages show for products whose agent surface lacks an element (a review app's rating, an
  allergen statement); the row runs on the store's replica with exactly those values added in the UCP field the
  element catalog names (carttrace.sandbox.catalogs.overlay), and its pair is the baseline on the replica as captured.
  Params `{"elements": [...], "inventory": "inv-...", "hypotheses": [...]}`. Replica rows only (a live store cannot
  be given data it lacks), so a live row never takes it. Not yet in CONDITIONS: manifest planning adds it when it
  plans the rows (`added_details` gives one inventory's experiments).
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.protocols.profile import ProtocolProfile
from carttrace.sandbox.catalogs.overlay import add_elements
from carttrace.sandbox.envpack import EnvPack
from carttrace.stores.categories import Category
from carttrace.trace.inventory import StoreDataInventory, find_inventory
from carttrace.trace.schema import Condition

ConditionName = Literal["baseline", "cross_border", "version_variant", "detail_hidden", "checkout"]
CONDITIONS: tuple[ConditionName, ...] = ("baseline", "cross_border", "version_variant", "detail_hidden", "checkout")
CHECKOUT_STEPS = 15
"""Agent steps a checkout row gets beyond collect.step_cap: the checkout, its delivery details, the shopper's approval,
the wallet, 3-D Secure and the order read back (the reference agent needs nine; a model, a few more)."""

HIDE_LOCATIONS: Mapping[str, str] = {
    "get_product.description": "description",
    "get_product.attributes": "attributes",
    "get_product.metafields": "metafields",
    "policies": "policies",
}
"""A hypothesis location (carttrace.stores.categories.Location) to the detail the filter removes to hide it."""
UNHIDEABLE: frozenset[str] = frozenset({"get_product.options"})

MARKET_ALIASES: Mapping[str, str] = {"UK": "GB"}
"""Market codes that name one market: the queue writes GB, the sandbox's own table UK."""


def baseline() -> Condition:
    return Condition(name="baseline")


def same_market(a: str, b: str) -> bool:
    return _market_key(a) == _market_key(b)


def cross_border_market(home: str, preferred: Sequence[str]) -> str | None:
    """The first preferred market that is not the store's own, or None when every preferred market is."""
    return next((market for market in preferred if not same_market(market, home)), None)


def cross_border(home: str, market: str) -> Condition:
    return Condition(name="cross_border", params={"market": market, "home_market": home})


def variant_profile(base: ProtocolProfile, target: ProtocolProfile, *, target_name: str) -> ProtocolProfile | None:
    """`base` moved to the UCP release of `target`: its core and capability versions, everything else unchanged. None
    when that changes nothing, since a variant that speaks the same versions is not an experiment."""
    derived = base.with_versions(
        f"version_variant: UCP from preset {target_name}",
        ucp=target.ucp,
        ucp_capabilities=dict(target.ucp_capabilities),
    )
    return derived if derived.differs_from(base) else None


def version_variant(base: ProtocolProfile, derived: ProtocolProfile, *, target_name: str) -> Condition:
    return Condition(
        name="version_variant",
        params={
            "preset": target_name,
            "from": base.id,
            "to": derived.id,
            "changed": sorted(derived.differs_from(base)),
        },
    )


@dataclass(frozen=True, slots=True)
class HiddenDetail:
    """One detail_hidden experiment: what is hidden, and every hypothesis it tests."""

    hide: tuple[str, ...]
    """The locations of HIDE_LOCATIONS the hypotheses name."""
    hypotheses: tuple[str, ...]
    details: tuple[str, ...] = ()
    """The catalog details hidden one by one, sorted; empty when the locations are hidden whole."""


def hidden_details(category: Category, only: Collection[str] = ()) -> tuple[list[HiddenDetail], list[str]]:
    """The category's detail_hidden experiments, one per distinct pair of catalog details and locations, in the order
    of their first hypothesis; and a note for each hypothesis that cannot be hidden. `only` limits them to those
    hypothesis ids."""
    grouped: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}
    notes: list[str] = []
    for hypothesis in category.hypotheses:
        if only and hypothesis.id not in only:
            continue
        blocked = sorted(set(hypothesis.where) & UNHIDEABLE)
        if blocked:
            notes.append(f"{hypothesis.id}: not hidden, its detail lives in {', '.join(blocked)}")
            continue
        hide = tuple(sorted({HIDE_LOCATIONS[where] for where in hypothesis.where}))
        grouped.setdefault((tuple(sorted(set(hypothesis.details))), hide), []).append(hypothesis.id)
    return [HiddenDetail(hide, tuple(sorted(ids)), details) for (details, hide), ids in grouped.items()], notes


def hides_decisive(spec: Mapping[str, Any], condition: Condition, products: Mapping[str, Mapping[str, Any]]) -> bool:
    """Whether a detail_hidden `condition` hides something from the task whose hidden spec is `spec` (D47): it names
    the task's decisive detail among its `details`, and the detail filter cuts a statement of it from at least one of
    the task's candidate products in `products` (id to product document, the world's fixtures). The judge's own test
    of the row afterwards (judge-v4 `condition_effective`) asks the same two things of the bundle."""
    from carttrace.collect.matchers import HiddenSet
    from carttrace.collect.surfaces import without_details

    decisive = spec.get("decisive") if isinstance(spec.get("decisive"), Mapping) else {}
    assert isinstance(decisive, Mapping)
    detail = decisive.get("detail")
    if not isinstance(detail, str) or detail not in details_of(condition):
        return False
    hidden = HiddenSet((detail,))
    candidates = spec.get("candidates") if isinstance(spec.get("candidates"), list) else []
    assert isinstance(candidates, list)
    for product_id in candidates:
        doc = products.get(str(product_id))
        if doc is None:
            continue
        counts: Counter[str] = Counter()
        without_details(doc, hidden, counts)
        if counts.get(detail, 0) > 0:
            return True
    return False


def detail_hidden(detail: HiddenDetail) -> Condition:
    """The condition of one experiment. `details` appears only when the experiment names catalog details, so an
    experiment that hides locations whole keeps the params (and the episode ids) it always had."""
    params: dict[str, Any] = {"hide": list(detail.hide), "hypotheses": list(detail.hypotheses)}
    if detail.details:
        params["details"] = list(detail.details)
    return Condition(name="detail_hidden", params=params)


ADDED = "detail_added"


@dataclass(frozen=True, slots=True)
class AddedDetail:
    """One detail_added experiment: the elements added, the inventory whose values they are, the hypotheses served."""

    elements: tuple[str, ...]
    inventory: str
    hypotheses: tuple[str, ...] = ()


def added_details(inventory: StoreDataInventory, only: Collection[str] = ()) -> list[AddedDetail]:
    """The detail_added experiments one inventory makes possible: one per element it holds values for (limited to
    `only` element ids when given), in element order, each naming the hypotheses the element serves."""
    out = []
    for addition in sorted(inventory.additions, key=lambda a: a.element):
        if only and addition.element not in only:
            continue
        info = inventory.element(addition.element)
        hypotheses = tuple(sorted(info.hypotheses)) if info is not None else ()
        out.append(AddedDetail((addition.element,), inventory.id, hypotheses))
    return out


def detail_added(detail: AddedDetail) -> Condition:
    params: dict[str, Any] = {"elements": list(detail.elements), "inventory": detail.inventory}
    if detail.hypotheses:
        params["hypotheses"] = list(detail.hypotheses)
    return Condition(name=ADDED, params=params)


def added_of(condition: Condition) -> tuple[str, ...]:
    """The elements a row's condition adds; empty for every condition but detail_added."""
    if condition.name != ADDED:
        return ()
    listed = condition.params.get("elements")
    return tuple(str(item) for item in listed) if isinstance(listed, list) else ()


def added_world(pack: EnvPack, condition: Condition, *, inventory_dir: Path) -> tuple[EnvPack, str]:
    """The pack a detail_added row runs on, and a note saying what was added: the replica `pack` with the condition's
    elements added from its inventory. Raises NotFoundError when the inventory is gone (stripped after the retention
    window) or holds no values for an element, so the row fails instead of running as a baseline in disguise."""
    elements = added_of(condition)
    ident = condition.params.get("inventory")
    if not elements or not isinstance(ident, str):
        raise ValidationError(f"a {ADDED} condition names no elements or no inventory: {condition.params}")
    inventory = find_inventory(inventory_dir, ident)
    additions = [inventory.addition(e) for e in elements]
    missing = [e for e, a in zip(elements, additions, strict=True) if a is None]
    if missing:
        raise NotFoundError(
            f"inventory {ident} holds no values for {', '.join(missing)}",
            hint="take a new inventory (carttrace live inventory) and plan the row again",
        )
    added = add_elements(pack, [a for a in additions if a is not None], inventory_id=ident)
    reached = ", ".join(f"{e} {added.reached.get(e, 0)}" for e in elements)
    return added.pack, f"{ADDED} served on {added.pack.id} (from {pack.id}, inventory {ident}): {reached} products"


def checkout(persona: str) -> Condition:
    return Condition(name="checkout", params={"checkout": True, "persona": persona})


def authorises_checkout(condition: Condition) -> bool:
    """Whether a row's condition authorises an order: the judge's convention, `params["checkout"]` true."""
    return condition.params.get("checkout") is True


def persona_of(condition: Condition) -> str | None:
    """The persona a checkout row pays as; None for a row that does not check out, or names no persona (the default
    one in data/agent/personas.yaml then pays)."""
    persona = condition.params.get("persona") if authorises_checkout(condition) else None
    return persona if isinstance(persona, str) and persona else None


def hidden_of(condition: Condition) -> tuple[str, ...]:
    """The locations a row's condition hides; empty for every condition but detail_hidden."""
    return _listed(condition, "hide")


def details_of(condition: Condition) -> tuple[str, ...]:
    """The catalog details a row's condition hides one by one; empty when it hides locations whole, and for every
    condition but detail_hidden."""
    return _listed(condition, "details")


def _listed(condition: Condition, key: str) -> tuple[str, ...]:
    if condition.name != "detail_hidden":
        return ()
    listed = condition.params.get(key)
    return tuple(str(item) for item in listed) if isinstance(listed, list) else ()


def _market_key(code: str) -> str:
    upper = code.strip().upper()
    return MARKET_ALIASES.get(upper, upper)
