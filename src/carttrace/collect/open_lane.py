"""The publishable lane (D85): sandbox traces on twins of catalogs whose licence allows publication, nothing else.

Why a lane of its own. The project (2026-10-16) runs the sandbox in two lanes: replicas built from live stores, for the
internal benchmark (never published, D10), and a lane whose every trace may be released as open data. Since the
always-on loop moved to replicas on 2026-10-13, about 0.2% of new traces allowed publication. So a publishable
manifest is refused at three places, each of which can see something the others cannot:
- the plan (`ManifestPlan.publishable_problem`): no replica, no live half, no hand-off, no store's custom intents;
- the manifest (`Manifest._publishable_sealed`): every row a sandbox or reference row, every world a twin of the
  plan's catalog with no capture and no record a live session left;
- the licence, here: the builder refuses a catalog whose rights do not allow publication before it plans a row
  (`check_open_catalog`), and the runner refuses to start a publishable manifest unless every rights entry it decided
  allows publication (`check_open_rights`).

A catalog is open when its licence allows publication and it is not restricted: `abo` (CC BY 4.0), `wands` (MIT)
and, since the project's decision D90 (amending D11), `lazada` (the ShoppingBench release, Apache-2.0), and `open`,
the open product pools of all three (D91), which the lane draws every stand-in store from: each store's products come
from its category's pool across the three, Lazada's only in a PH store, priced in pesos. Open Food Facts and Open
Beauty Facts (ODbL) say no until a legal review flips their licence records. Training follows the licence as every
sandbox bundle's does (`carttrace.collect.rights`).

`open_stores` lists the queue stores a catalog can stand in for: the twin borrows the store's id, category, market and
protocol profile (D25), and the release pseudonymises it (D74). A store whose category the catalog does not cover, a
store in a market the sandbox lacks, and a catalog publisher's own store (its text names the store, so a pseudonym
would not hide it) are left out. The stores come category by category in turn, so a round's slice of the rotation
spans the categories instead of filling one.
"""

from __future__ import annotations

from collections.abc import Collection, Iterable, Mapping, Sequence
from itertools import zip_longest
from pathlib import Path

from carttrace.collect.bindings import Bindings
from carttrace.collect.manifest import DEFAULT_HOME_MARKET, Manifest, ManifestPlan
from carttrace.collect.rights import sandbox_rights
from carttrace.collect.worlds import REPLICA, sandbox_market
from carttrace.core.canon import loads
from carttrace.core.errors import NotFoundError, SafetyError
from carttrace.stores.model import Store
from carttrace.trace.schema import Rights

COVERAGE_SUFFIX = ".coverage.json"
"""data/catalogs/<source>.coverage.json, written by `carttrace sandbox catalogs build <source> --record`."""


def open_catalog_problem(name: str, rights: Rights, *, restricted: bool) -> str | None:
    """Why catalog `name`, whose sandbox bundles carry `rights`, cannot feed the publishable lane (completing
    "catalog ... "), or None when it can."""
    rules = (
        (name == REPLICA, "is a store's replica, which carries the captured store's rights (D10)"),
        (restricted, "is restricted to internal research (D11)"),
        (rights.platform_terms != "sandbox", f"is under platform terms {rights.platform_terms}"),
        (not rights.publication_allowed, f"does not allow publication: {rights.basis or 'no basis recorded'}"),
    )
    return next((reason for broken, reason in rules if broken), None)


def check_open_catalog(plan: ManifestPlan, bindings: Bindings) -> Rights | None:
    """The rights a publishable plan's bundles will carry; raises SafetyError, before anything is planned, when its
    catalog cannot feed the lane. None for a plan that is not publishable."""
    if not plan.publishable:
        return None
    if plan.catalog == REPLICA:
        raise SafetyError(
            "a publishable plan runs on twins of a licensed catalog, never on replicas (D85)",
            hint="choose an open catalog: open (the pools), abo, wands or lazada",
        )
    restricted = bindings.catalog_restricted(plan.catalog)
    rights = sandbox_rights(bindings.licence_rights(plan.catalog), restricted=restricted)
    problem = open_catalog_problem(plan.catalog, rights, restricted=restricted)
    if problem is not None:
        raise SafetyError(
            f"catalog {plan.catalog!r} {problem}; it cannot feed the publishable lane (D85)",
            hint="choose an open catalog (open, abo, wands, lazada); off and obf wait for the legal review of their "
            "licence",
        )
    return rights


def check_open_rights(manifest: Manifest, rights: Mapping[str, Rights]) -> None:
    """The runner's last check before a publishable manifest's first episode: every rights entry it decided (one per
    catalog, one per replica capture) allows publication. Raises SafetyError naming the first that does not."""
    if not manifest.plan.publishable:
        return
    for key, entry in sorted(rights.items()):
        if ":" in key or not entry.publication_allowed or entry.platform_terms != "sandbox":
            raise SafetyError(
                f"publishable manifest {manifest.id} would write bundles under {key!r}, whose rights do not allow "
                "publication (D85); no episode was started",
                hint="rebuild it with --publishable on an open catalog (open, abo, wands, lazada)",
            )


def covered_categories(data_dir: Path, catalog: str) -> list[str]:
    """The categories `catalog` covers, from its checked-in coverage file: those it offers at least one store in. A
    pool (data/catalogs/<catalog>.json of kind pool, D91) covers what any of its members covers."""
    members = pool_members(data_dir, catalog)
    if members:
        return sorted({c for member in members for c in covered_categories(data_dir, member)})
    path = data_dir / "catalogs" / f"{catalog}{COVERAGE_SUFFIX}"
    if not path.is_file():
        raise NotFoundError(
            f"no coverage file for catalog {catalog!r} at {path}",
            hint=f"run `carttrace sandbox catalogs build {catalog} --record`",
        )
    doc = loads(path.read_bytes())
    table = doc.get("categories") if isinstance(doc, Mapping) else None
    if not isinstance(table, Mapping):
        return []
    return sorted(
        str(slug) for slug, entry in table.items() if isinstance(entry, Mapping) and int(entry.get("stores") or 0) > 0
    )


def pool_members(data_dir: Path, catalog: str) -> list[str]:
    """The members a pool manifest names, or none for a catalog that is not a pool."""
    path = data_dir / "catalogs" / f"{catalog}.json"
    doc = loads(path.read_bytes()) if path.is_file() else None
    if not isinstance(doc, Mapping) or doc.get("kind") != "pool":
        return []
    return [str(m) for m in doc.get("members") or []]


def open_stores(stores: Sequence[Store], covered: Collection[str], publishers: Iterable[str] = ()) -> list[str]:
    """The ids of the queue `stores` a twin of a catalog covering `covered` can stand in for, category by category in
    turn (the first store of each category, then the second of each, ...), each category's in queue order. A catalog
    publisher's domain (release.exclude_store_domains) and a store in a market the sandbox lacks are left out."""
    banned = {_bare(d) for d in publishers}
    by_category: dict[str, list[str]] = {}
    for store in stores:
        if store.category_slug not in covered or _bare(store.domain) in banned:
            continue
        if sandbox_market(store.market_code or DEFAULT_HOME_MARKET) is None:
            continue
        by_category.setdefault(store.category_slug, []).append(store.id)
    turns = zip_longest(*(by_category[slug] for slug in sorted(by_category)))
    return [store_id for turn in turns for store_id in turn if store_id is not None]


def _bare(domain: str) -> str:
    return domain.strip().lower().removeprefix("www.")


__all__ = [
    "check_open_catalog",
    "check_open_rights",
    "covered_categories",
    "open_catalog_problem",
    "open_stores",
    "pool_members",
]
