"""The shopper's decisions, mapped when the task is built (sandbox v1 final, D115): the shopper looks them up.

Why. The simulated shopper must not depend on a model, or on the sandbox's answers worked out on
the fly, to decide whether an offered item is the one it wants: "map it beforehand". At build time, on the row's own
sealed world (the same world the D104 build-time replay runs on), every variant the agent could offer is decided once
against the hidden spec, with the need it misses, and stored with the manifest (`Manifest.shopper_maps`, keyed by the
row's envpack and task); the spec's answer for each field, the delivery wish and the substitute policy beside it. The
runtime shopper (carttrace.agents.shopper `use_map`) reads the map: whether the offered item is acceptable, why not, how it
wants the order delivered. The same code decides as before, once and on record; a variant the map does not hold
(none should be: the map covers the whole served catalog) falls back to the same rule.

Schema `mc.shopper-map.v1`, plain JSON:
- `accept`: variant keys the shopper approves (a witness, or a served variant meeting every need, D69's rule);
- `refuse`: field of the need each other variant misses to the sorted variant keys (`stock` for a variant shown out
  of stock, `witness` for one the spec keeps to its witnesses, as a bundle does);
- `fields`: each spec field to its source (stated, on_ask, default) and the clause the shopper says;
- `delivery`: the buyer's delivery wish (value, source, say, deadline_days) or null;
- `substitute`: whether a substitute is allowed, and what the shopper says of it.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Final

from carttrace.agents.spec import BuyerFulfillment, HiddenSpec
from carttrace.trace.schema import TaskRef

SCHEMA: Final = "mc.shopper-map.v1"
STOCK: Final = "stock"
WITNESS: Final = "witness"


@dataclass(frozen=True, slots=True)
class ShopperMap:
    """A task's decisions on one world, as the runtime shopper reads them."""

    accept: frozenset[str]
    refuse: Mapping[str, str] = field(default_factory=dict)
    """Variant key to the field of the need it misses."""
    fields: Mapping[str, Mapping[str, str]] = field(default_factory=dict)
    delivery: BuyerFulfillment | None = None
    substitute: Mapping[str, Any] = field(default_factory=dict)

    def decides(self, key: str) -> bool | None:
        """Whether the shopper approves the variant `key`; None when the map does not hold it."""
        if key in self.accept:
            return True
        return False if key in self.refuse else None

    def missed(self, key: str) -> str | None:
        """The field of the need the variant misses, when the map refuses it."""
        return self.refuse.get(key)

    @classmethod
    def of(cls, doc: Mapping[str, Any]) -> ShopperMap:
        refuse = {key: why for why, keys in (doc.get("refuse") or {}).items() for key in keys}
        delivery = doc.get("delivery")
        return cls(
            accept=frozenset(doc.get("accept") or ()),
            refuse=refuse,
            fields={k: dict(v) for k, v in (doc.get("fields") or {}).items()},
            delivery=BuyerFulfillment.model_validate(delivery) if isinstance(delivery, Mapping) else None,
            substitute=dict(doc.get("substitute") or {}),
        )


def build_map(task: TaskRef, fixtures: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """The shopper's decision map for `task` on the world whose served catalog is `fixtures` (plain JSON)."""
    from carttrace.agents.shopper import DeterministicShopper, variant_key

    shopper = DeterministicShopper(task)
    shopper.use_catalog(fixtures)
    spec = shopper.spec
    accept: list[str] = []
    refuse: dict[str, list[str]] = {}
    for key, (product, variant) in sorted(shopper.served_variants().items()):
        if shopper.fits_variant(variant.id):
            accept.append(key)
            continue
        if not variant.available or variant.id in spec.stocked_out:
            why = STOCK
        else:
            need = shopper.failing_need(product, [variant])
            why = need.field if need is not None else WITNESS
        refuse.setdefault(why, []).append(key)
    for witness in spec.witnesses:
        key = variant_key(witness)
        if key not in accept and not any(key in keys for keys in refuse.values()):
            accept.append(key)
    return {
        "schema": SCHEMA,
        "accept": sorted(accept),
        "refuse": {why: sorted(keys) for why, keys in sorted(refuse.items())},
        "fields": _fields(spec),
        "delivery": _delivery(spec),
        "substitute": _substitute(spec),
    }


def _fields(spec: HiddenSpec) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for p in spec.predicates:
        out[p.field] = {"source": p.source, "say": p.say or ""}
    buyer = spec.buyer
    if buyer is not None:
        for name in buyer.fields():
            found = buyer.field(name)
            if found is not None:
                out[name] = {"source": found.source, "say": getattr(found, "say", "") or ""}
    return dict(sorted(out.items()))


def _delivery(spec: HiddenSpec) -> dict[str, Any] | None:
    held = spec.buyer.fulfillment if spec.buyer is not None else None
    return None if held is None else held.model_dump(mode="json")


def _substitute(spec: HiddenSpec) -> dict[str, Any]:
    found = spec.predicate("substitute_ok")
    if found is None:
        return {"allowed": False, "say": ""}
    return {"allowed": found.value is True, "say": found.say or "", "source": found.source}


def map_key(envpack: str, task_id: str) -> str:
    """A map's key in `Manifest.shopper_maps`: the row's world and task."""
    return f"{envpack}/{task_id}"


__all__ = ["SCHEMA", "ShopperMap", "build_map", "map_key"]
