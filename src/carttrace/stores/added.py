"""Stores added by hand beside the team sheet: a domain, a category, a delivery market and a mode, sandbox or live.

Why a file of its own. The store queue is the team sheet, snapshotted under data/stores/ and never edited here, and a
person who finds a store worth testing should not wait for the sheet and a pull. `add_store` appends one record to
var/stores/added.jsonl (schema mc.stores.added.v1), and the queue the collector plans from (`planning_queue`) is the
snapshot's stores in queue order followed by the added ones in the order they were added (carttrace.stores.queue.merge_added),
so no sheet id changes meaning. Ids are store-a001, store-a002, ... by position in the file, which is append-only: a store
added by hand is never renumbered or removed, so its traces keep their id.

Why so strict on the way in. A store added in live mode becomes real traffic once every live gate opens, and a second
copy of a sheet store would split one merchant across the train and test splits. So the domain must be a public host
name, the category one of the queue's, and a domain whose registrable domain the sheet or an earlier added store
already holds is refused, naming the store that holds it. As a `Store`, an added store's platform is "unknown" until a
preflight (carttrace stores preflight) confirms one, its merchant group is its registrable domain as the sheet's would be,
and its split is "train". When a later snapshot of the sheet lists the same registrable domain, the sheet's store wins
and the added one leaves the planning queue.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path

from pydantic import BaseModel, ConfigDict
from pydantic import ValidationError as PydanticValidationError

from carttrace.config.schema import Settings
from carttrace.core.clock import Clock, SystemClock, iso
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import append_jsonl_locked, read_jsonl
from carttrace.core.ids import stable_id
from carttrace.stores.categories import Category, CategoryCatalog, load_categories
from carttrace.stores.merchant import normalise_host, registrable_domain
from carttrace.stores.model import MARKETS, AddedMode, PlatformKey, Store, market_of, platform_key
from carttrace.stores.queue import load_queue
from carttrace.stores.results import StoreResult
from carttrace.stores.settings import StoresSettings, from_settings
from carttrace.stores.snapshot import latest_snapshot
from carttrace.stores.text import collapse_ws, slugify

ADDED_SCHEMA = "mc.stores.added.v1"
ADDED_FILE = "added.jsonl"
"""Under <paths.var>/stores/, beside the results file the preflight and the collector append to."""
ID_PREFIX = "MC-A"
UNKNOWN_PLATFORM = "unknown"
RESERVED_SUFFIXES = (".invalid", ".test", ".example", ".localhost", ".local")
"""Names that never belong to a real store (RFC 2606 and RFC 6761); sandbox stores live under .invalid."""
MAX_TEXT = 200
_ADDED_ID = re.compile(rf"^{re.escape(ID_PREFIX)}\d{{3,}}$")
_MARKET_CODES: dict[str, str] = {code: currency for code, currency in MARKETS.values() if code and currency}

AnySettings = Settings | StoresSettings


class AddedStore(BaseModel):
    """One store added by hand, as add_store recorded it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    store_id: str
    """store-a001, store-a002, ...: the record's position in the file."""
    domain: str
    """Canonical host: lowercase, no scheme, path, port or "www."."""
    category: str
    """Category slug (data/categories)."""
    name: str | None = None
    market: str = "US"
    """Delivery market code as the queue writes it: US, GB, EU, IN, CA, JP, AU, AE or SG."""
    mode: AddedMode
    """sandbox: tested on its sandbox twin only. live: also tested live, once autopilot.live is true and every live
    gate is open (CARTTRACE_LIVE=1, live.enabled, no pause in the control file)."""
    added_at: str
    """RFC 3339, UTC."""
    added_by: str | None = None


def added_path(settings: AnySettings) -> Path:
    """<paths.var>/stores/added.jsonl."""
    return _view(settings).var_dir / "stores" / ADDED_FILE


def store_id_at(position: int) -> str:
    """The id of the store added `position`-th (from 1): store-a001, ..., store-a999, MC-A1000."""
    return f"{ID_PREFIX}{position:03d}"


def is_added_id(store_id: str) -> bool:
    """Whether an id names a store added by hand, which has no row in the team sheet by design."""
    return _ADDED_ID.match(store_id) is not None


def add_store(
    settings: AnySettings,
    *,
    domain: str,
    category: str,
    name: str | None = None,
    market: str = "US",
    mode: AddedMode = "sandbox",
    added_by: str | None = None,
    clock: Clock | None = None,
) -> AddedStore:
    """Add a store to test and return its record. Raises ValidationError, with a hint saying what to pass instead,
    for a domain that is not a public host, an unknown category or market, an unknown mode, and a store the queue or
    an earlier addition already holds (the hint names that store's id)."""
    view = _view(settings)
    host = normalise_domain(domain)
    categories = load_categories(view.categories_dir)
    slug = category_slug(category, categories)
    code = market_code(market)
    if mode not in ("sandbox", "live"):
        raise ValidationError(f"no mode {mode!r}", hint="a store is added in sandbox or live mode")
    queue = load_queue(latest_snapshot(view.stores_dir), queue_tab=view.queue_tab, evidence_tab=view.evidence_tab)
    path = added_path(view)
    with _exclusive(path):
        _drop_torn_tail(path)
        existing = added_stores(view)
        _refuse_known(host, queue, existing)
        record = AddedStore(
            store_id=store_id_at(len(existing) + 1),
            domain=host,
            category=slug,
            name=_text(name, "name"),
            market=code,
            mode=mode,
            added_at=iso((clock or SystemClock()).now()),
            added_by=_text(added_by, "added_by"),
        )
        append_jsonl_locked(path, {"schema": ADDED_SCHEMA, **record.model_dump(mode="json")})
    return record


def added_stores(settings: AnySettings) -> list[AddedStore]:
    """Every store added by hand, in the order added; [] before the first. Raises ValidationError for a record that
    is not one or ids out of sequence, which only a hand edit makes."""
    path = added_path(settings)
    out: list[AddedStore] = []
    for position, raw in enumerate(read_jsonl(path), start=1):
        if not isinstance(raw, dict) or raw.get("schema") != ADDED_SCHEMA:
            raise ValidationError(
                f"{path} record {position} is not a {ADDED_SCHEMA} record",
                hint="the file is written by `carttrace stores add` only; restore it from a backup",
            )
        try:
            record = AddedStore.model_validate({key: value for key, value in raw.items() if key != "schema"})
        except PydanticValidationError as exc:
            first = exc.errors(include_url=False, include_input=False)[0]
            where = ".".join(str(part) for part in first["loc"]) or "record"
            raise ValidationError(
                f"{path} record {position}: {where}: {first['msg']}",
                hint="the file is written by `carttrace stores add` only; restore it from a backup",
            ) from None
        if record.store_id != store_id_at(position):
            raise ValidationError(
                f"{path} record {position} holds {record.store_id}, not {store_id_at(position)}",
                hint="ids follow the order of the file, which is append-only; never edit it by hand",
            )
        out.append(record)
    return out


def added_queue(
    settings: AnySettings,
    *,
    categories: CategoryCatalog | None = None,
    results: Mapping[str, StoreResult] | None = None,
) -> list[Store]:
    """The stores added by hand as `Store` records, in the order added. `results` (folded preflight results by store
    id, carttrace.stores.load_results) give a store the platform its preflight confirmed."""
    view = _view(settings)
    catalog = categories or load_categories(view.categories_dir)
    found = results or {}
    return [
        added_store_record(added, catalog.get(added.category), found.get(added.store_id))
        for added in added_stores(view)
    ]


def planning_queue(
    settings: AnySettings,
    *,
    snapshot: Path | None = None,
    categories: CategoryCatalog | None = None,
    results: Mapping[str, StoreResult] | None = None,
) -> list[Store]:
    """The queue the collector plans from: the snapshot's stores in queue order (the latest snapshot unless given),
    then the stores added by hand that the snapshot does not hold, in the order added."""
    view = _view(settings)
    chosen = snapshot or latest_snapshot(view.stores_dir)
    added = added_queue(view, categories=categories, results=results)
    return load_queue(chosen, queue_tab=view.queue_tab, evidence_tab=view.evidence_tab, added=added)


def added_store_record(added: AddedStore, category: Category, result: StoreResult | None = None) -> Store:
    """One store added by hand as a `Store`: platform "unknown" until `result` (its folded preflight) names one,
    merchant group by registrable domain, split "train", and its mode in `added_mode`."""
    platform = result.platform if result is not None and result.platform else UNKNOWN_PLATFORM
    key: PlatformKey = "unknown" if platform == UNKNOWN_PLATFORM else (platform_key(platform) or "other")
    registrable = registrable_domain(added.domain)
    return Store(
        id=added.store_id,
        category=category.name,
        name=added.name or added.domain,
        website=added.domain,
        platform=platform,
        market=added.market,
        parent_company=None,
        status=f"Added in {added.mode} mode",
        what_we_saw="",
        what_to_do_next="",
        test_fixture=None,
        last_observed=None,
        category_index=category.index,
        url=None,
        source_status=None,
        checked_at=None,
        fit_reason=None,
        observed_evidence=None,
        parent_group=None,
        evidence_url=None,
        canonical_host=added.domain,
        duplicate_host_count=None,
        publication_permission=None,
        eligibility=None,
        platform_evidence=None,
        delivery_market=None,
        sheet_merchant_group_id=None,
        browser_preflight_result=None,
        task_fixture=None,
        collector_eligibility=None,
        sheet_split=None,
        gate_approval_reference=None,
        category_slug=category.slug,
        status_key="other",
        platform_key=key,
        market_code=added.market,
        market_likely=False,
        currency=_MARKET_CODES.get(added.market),
        domain=added.domain,
        registrable_domain=registrable,
        merchant_key=registrable,
        merchant_basis="domain",
        merchant_group_id=stable_id("mg", registrable),
        split="train",
        added_mode=added.mode,
    )


def normalise_domain(text: str) -> str:
    """ "https://www.Example.com/shop" -> "example.com". Raises ValidationError for anything that is not a public
    host name: a path alone, credentials, a port, an IP address, localhost or a reserved name."""
    from carttrace.live.hosts import host_of  # the one host check every live path uses; imported here, carttrace.live is heavy

    host = normalise_host(host_of(text))
    if host.endswith(RESERVED_SUFFIXES):
        raise ValidationError(f"{host} is a reserved name, never a real store", hint="pass the store's own domain")
    return host


def category_slug(text: str, categories: CategoryCatalog) -> str:
    """The slug of a category given by slug or name ("Clothing (adult)" -> "clothing-adult")."""
    slug = slugify(text)
    try:
        return categories.get(slug).slug
    except NotFoundError:
        raise ValidationError(
            f"no category {text!r}", hint=f"choose one of {[c.slug for c in categories]} (carttrace stores categories)"
        ) from None


def market_code(text: str) -> str:
    """The queue's market code for a code or a sheet label: "us" -> "US", "UK" -> "GB"."""
    found = market_of(text)
    if found is not None and found[0]:
        return found[0]
    code = text.strip().upper()
    if code in _MARKET_CODES:
        return code
    raise ValidationError(f"no market {text!r}", hint=f"name the market the store delivers to: {sorted(_MARKET_CODES)}")


def _refuse_known(host: str, queue: list[Store], added: list[AddedStore]) -> None:
    registrable = registrable_domain(host)
    for store in queue:
        if store.domain == host:
            raise ValidationError(
                f"{host} is already in the store queue as {store.id}", hint=f"test {store.id}; it is planned already"
            )
        if store.registrable_domain == registrable:
            raise ValidationError(
                f"{host} is the same merchant as {store.id} ({store.domain}): both are under {registrable}",
                hint=f"test {store.id}; one merchant is one store, so its traces never straddle the splits",
            )
    for earlier in added:
        if earlier.domain == host or registrable_domain(earlier.domain) == registrable:
            raise ValidationError(
                f"{host} was already added as {earlier.store_id} ({earlier.domain})",
                hint=f"test {earlier.store_id}; see `carttrace stores added`",
            )


def _text(value: str | None, label: str) -> str | None:
    if value is None:
        return None
    text = collapse_ws(value)
    if len(text) > MAX_TEXT:
        raise ValidationError(f"{label} is longer than {MAX_TEXT} characters", hint=f"shorten {label}")
    return text or None


def _view(settings: AnySettings) -> StoresSettings:
    return settings if isinstance(settings, StoresSettings) else from_settings(settings)


@contextmanager
def _exclusive(path: Path) -> Iterator[None]:
    """Hold the added file's sidecar lock while its next id is decided and its record appended, so two people adding
    at once never take one id. The append itself still goes through append_jsonl_locked, which locks the file."""
    import fcntl  # POSIX only, as carttrace.core.files

    path.parent.mkdir(parents=True, exist_ok=True)
    with (path.parent / f"{path.name}.lock").open("a", encoding="utf-8") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        yield


def _drop_torn_tail(path: Path) -> None:
    """Cut a last line that a crashed append left without its newline, so the next record starts a line of its own.
    That record was never acknowledged (add_store raised, or its process died) and readers already skip it."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        return
    if not data or data.endswith(b"\n"):
        return
    with path.open("r+b") as fh:
        fh.truncate(data.rfind(b"\n") + 1)
