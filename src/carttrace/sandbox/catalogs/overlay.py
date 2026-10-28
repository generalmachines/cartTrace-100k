"""detail_added on a replica: the store's own replica, served with data elements added to its agent surface from
what the store's own pages show (decision D19).

Why. The live audit (carttrace live inventory) finds elements people see on a store's pages that agents cannot read on
UCP: a review app's rating, an allergen statement in the page text, a return window in the page's markup. The
replica of the store (D12) can say what exposing one of them would be worth: paired runs on the replica as captured
(baseline) and on the same replica with the element added (detail_added) differ in that element and nothing else.
detail_hidden is the same experiment with the arms reversed.

What "added" means, exactly:
- each Addition (carttrace.trace.inventory) names a UCP field and holds the store's own value per product, as its page or
  markup showed it, only for products whose agent surface lacked the element;
- `rating`: the product's `rating` field; `metadata.<key>`: an entry under the product's `metadata`; `policies`: a
  policy served with the product (`*` adds it to every product);
- the views the store gave a product (`_listing`, `_detail`, carttrace.sandbox.surfaces.ucp.catalog) gain the field only
  where the Addition says (`views`), and keep showing exactly what they showed before everywhere else, so a listing
  that never carried metadata does not start carrying the store's other metadata. A product with no recorded view
  is served in its platform's view (D55), which may leave the field out (Shopify's has no rating), so the overlay
  names every field it placed under `_added`, per view, and those are served whatever the view's shape;
- nothing else in the pack changes: variants, prices, recorded answers, the profile; the pack's provenance records
  the inventory, the elements and how many products each reached.

The result is a new pack (its id is its content), built from the old one; the old one is untouched.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from carttrace.core.canon import dumps
from carttrace.sandbox.catalogs.errors import CatalogError
from carttrace.sandbox.envpack import EnvPack
from carttrace.sandbox.surfaces.ucp.catalog import ADDED, DETAIL, LISTING, POLICIES
from carttrace.trace.inventory import Addition

VIEWS = {"listing": LISTING, "detail": DETAIL}
PROVENANCE_KEY = "detail_added"
EVERY_PRODUCT = "*"


@dataclass(frozen=True, slots=True)
class Added:
    """What an overlay did: the new pack, and per element how many of the replica's products it reached."""

    pack: EnvPack
    reached: Mapping[str, int]
    missing: Mapping[str, int]
    """Per element, the products the Addition names that this replica does not hold."""


def add_elements(pack: EnvPack, additions: Sequence[Addition], *, inventory_id: str) -> Added:
    """The replica `pack` with every Addition served on its agent surface (see the module docstring). Raises
    CatalogError for a pack that is not a replica or an Addition naming a field it cannot place."""
    doc = pack.doc
    products: dict[str, dict[str, Any]] = doc["snapshot"]["products"]
    if not products or not all(isinstance(p.get("variants"), list) for p in products.values()):
        raise CatalogError(
            f"pack {pack.id} is not a store's replica, so nothing can be added to it",
            hint="detail_added runs on replicas only: build the manifest with --catalog replica",
        )
    reached: dict[str, int] = {}
    missing: dict[str, int] = {}
    for addition in sorted(additions, key=lambda a: a.element):
        targets = _targets(addition, products)
        missing[addition.element] = sum(1 for pid in addition.values if pid != EVERY_PRODUCT and pid not in products)
        for pid, value in targets:
            _place(products[pid], addition, copy.deepcopy(value))
        reached[addition.element] = len(targets)
    provenance = dict(doc.get("provenance") or {})
    provenance[PROVENANCE_KEY] = {
        "inventory": inventory_id,
        "base": pack.id,
        "elements": {a.element: {"field": a.field, "products": reached[a.element]} for a in additions},
    }
    doc["provenance"] = provenance
    return Added(EnvPack.from_bytes(dumps(doc).encode("utf-8")), reached, missing)


def _targets(addition: Addition, products: Mapping[str, Any]) -> list[tuple[str, Any]]:
    if EVERY_PRODUCT in addition.values:
        value = addition.values[EVERY_PRODUCT]
        return [(pid, value) for pid in sorted(products)]
    return [(pid, value) for pid, value in sorted(addition.values.items()) if pid in products]


def _place(facts: dict[str, Any], addition: Addition, value: Any) -> None:
    if addition.field == "policies":
        policies = facts.setdefault(POLICIES, [])
        if value not in policies:
            policies.append(value)
        return
    if addition.field == "rating":
        _set_top(facts, "rating", value, addition.views)
        return
    parts = addition.field.split(".")
    if parts[0] != "metadata" or len(parts) < 2:
        raise CatalogError(f"{addition.element}: cannot place a value at {addition.field!r}")
    _set_metadata(facts, parts[1:], value, addition.views)


def _set_top(facts: dict[str, Any], key: str, value: Any, views: Sequence[str]) -> None:
    facts[key] = value
    _mark_added(facts, key, views)
    for name, view in VIEWS.items():
        shape = facts.get(view)
        if isinstance(shape, dict) and name in views:
            fields = list(shape.get("fields") or [])
            if key not in fields:
                shape["fields"] = sorted([*fields, key])


def _set_metadata(facts: dict[str, Any], path: Sequence[str], value: Any, views: Sequence[str]) -> None:
    before = copy.deepcopy(facts.get("metadata")) if isinstance(facts.get("metadata"), dict) else None
    metadata = facts.setdefault("metadata", {})
    if not isinstance(metadata, dict):
        raise CatalogError("the product's metadata is not an object, so nothing can be added under it")
    _nested_set(metadata, path, value)
    _mark_added(facts, "metadata", views)
    for name, view in VIEWS.items():
        shape = facts.get(view)
        if not isinstance(shape, dict):
            continue
        fields = list(shape.get("fields") or [])
        values = shape.setdefault("values", {})
        if name in views:
            if "metadata" not in fields:
                shape["fields"] = sorted([*fields, "metadata"])
                values["metadata"] = _nested_set(copy.deepcopy(values.get("metadata") or {}), path, value)
            elif isinstance(values.get("metadata"), dict):
                _nested_set(values["metadata"], path, value)
        elif "metadata" in fields and "metadata" not in values:
            values["metadata"] = before if before is not None else {}
        if not values:
            shape.pop("values")


def _mark_added(facts: dict[str, Any], key: str, views: Sequence[str]) -> None:
    """Name `key` under the product's `_added` for each view the Addition serves it in."""
    added = facts.setdefault(ADDED, {})
    for name in VIEWS:
        if name in views and key not in added.setdefault(name, []):
            added[name] = sorted([*added[name], key])
    for name in [n for n, keys in added.items() if not keys]:
        del added[name]
    if not added:
        del facts[ADDED]


def _nested_set(target: dict[str, Any], path: Sequence[str], value: Any) -> dict[str, Any]:
    current = target
    for key in path[:-1]:
        nxt = current.get(key)
        if not isinstance(nxt, dict):
            nxt = current[key] = {}
        current = nxt
    current[path[-1]] = value
    return target
