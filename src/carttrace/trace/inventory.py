"""A store's data inventory, `mc.inventory.v1`: what one store gives an AI shopping agent, element by element, on
every surface it serves, measured read-only on one day (decision D19).

Why a record of its own. A capture (D12) keeps what the store's agent protocol answered, as the ground truth of its
replica. The inventory asks a different question: of the data a shopper's agent needs (reviews, size charts, the
decisive details of a category, policies, search), which does the store publish, and where? People see a store
through its HTML pages (and search engines through their schema.org markup); agents see it through UCP over MCP and
Shopify's Storefront MCP. An element that people can see but agents cannot is the cheapest, most concrete fix a store
can make, so the inventory holds, per store and day:
- `cells`: the element by surface matrix, each cell present, partial, absent or unknown, with the share of sampled
  products that carry it and at most two short evidence quotes naming their source and detector version;
- `search`: the search probe's scores on each agent search surface (definitions in carttrace.live.search_probe);
- `gaps`: the flags a store acts on, the first of them `people_only` (seen by people, missing for agents);
- `additions`: for the detail_added knob, the values the store's own pages show for products whose agent surface
  lacks them, and the UCP field each one belongs in (carttrace.sandbox.catalogs.overlay serves them on the store's replica);
- the surfaces read and how each read ended, the sample of products, the element catalog's hash, the budget, and a
  live store's rights (D10: training and publication false).

Why here and not in carttrace.live. carttrace.live writes inventories, the sandbox overlay and the benchmark's census read them, and
the retention purge strips them; the first two sit side by side on layer 4, so the schema lives one layer down, as the
capture's does.

Files and ids. `save` writes `<inventory dir>/<store>/<YYYY-MM-DD>.json` holding exactly the canonical bytes
(carttrace.core.canon); the id is `inv-<sha12>` of the content without the id, so the same replies give the same bytes. One
inventory per store and day: a second run the same day replaces the file. Everything is redacted with the trace's one
table before it is sealed (`seal_inventory`), and reviewer names never reach it at all: review elements keep counts
and ratings, never who wrote them. After `live.retention_days_raw` days, `strip_inventory` (run by `carttrace trace purge`)
removes what is store content (quotes, sampled titles and URLs, additions) and keeps the states, shares, scores and
gaps, as expired scans keep their verdicts.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.canon import digest, dumps, loads
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import atomic_write_bytes
from carttrace.trace.capture import store_dir
from carttrace.trace.redact import redact_json
from carttrace.trace.schema import Rights, StoreRef

INVENTORY_SCHEMA: Final = "mc.inventory.v1"
INVENTORY_DIR: Final = "inventory"
"""Inventories live under <paths.var>/inventory/<store>/<YYYY-MM-DD>.json."""
QUOTE_CHARS: Final = 200
MAX_EVIDENCE: Final = 2
_ID = re.compile(r"^inv-[0-9a-f]{12}$")
_FILE = re.compile(r"^\d{4}-\d{2}-\d{2}\.json$")
_DAY = re.compile(r"^\d{4}-\d{2}-\d{2}$")

Surface = Literal["ucp_mcp", "store_mcp", "jsonld", "microdata", "html", "store_files"]
SURFACES: Final[tuple[Surface, ...]] = ("ucp_mcp", "store_mcp", "jsonld", "microdata", "html", "store_files")
AGENT_SURFACES: Final[tuple[Surface, ...]] = ("ucp_mcp", "store_mcp")
"""What an agent reads: UCP over MCP first (the protocol the benchmark's agent speaks), Shopify's Storefront MCP."""
PEOPLE_SURFACES: Final[tuple[Surface, ...]] = ("jsonld", "microdata", "html")
"""What a person (and a search engine) reads on the store's product pages."""
State = Literal["present", "partial", "absent", "unknown"]
STATE_RANK: Final[Mapping[str, int]] = {"present": 3, "partial": 2, "absent": 1, "unknown": 0}
ReadStatus = Literal[
    "read", "not_served", "robots_disallowed", "blocked", "throttled", "budget_spent", "error", "skipped"
]
"""How reading one surface ended: read; not_served (the store offers no such surface); robots_disallowed; blocked (a
401, a 403 or a bot wall); throttled (a 429, the store was paused and the inventory stopped there); budget_spent;
error (a network failure or a store error); skipped (not attempted: the inventory stopped earlier, or a dry run)."""
Ended = Literal["complete", "budget_spent", "throttled", "blocked"]
GapKind = Literal["people_only", "store_mcp_only", "partial_on_agent", "absent_everywhere", "agent_unread"]
"""people_only: present or partial for people (HTML, JSON-LD, microdata) and absent on UCP: the cheap, concrete fix;
store_mcp_only: Shopify's Storefront MCP serves it and UCP does not; partial_on_agent: UCP carries it for fewer
products than the store's pages do; absent_everywhere: no surface carries it, so the store needs a data provider;
agent_unread: UCP could not be read (robots, no UCP, a wall) while people see the element."""
Scope = Literal["product", "store"]
View = Literal["listing", "detail"]
DETAIL_ONLY: Final[tuple[View, ...]] = ("detail",)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class InventoryBudget(_Frozen):
    """How much one inventory may ask a store for. Every bound is checked before the request it would exceed."""

    requests: int = Field(default=150, ge=10, le=1000)
    """HTTP requests to the store in all, robots.txt, discovery, every session and every page included."""
    pages: int = Field(default=20, ge=1, le=100)
    """HTML pages (product pages and policy pages) read at most; a sitemap is not counted as a page."""
    products: int = Field(default=40, ge=1, le=500)
    """Products read through the agent protocol (a capture's get_product) at most."""
    sample: int = Field(default=16, ge=1, le=100)
    """Products compared across every surface: the same products, read on UCP, Storefront MCP and their pages."""
    probe_queries: int = Field(default=17, ge=0, le=60)
    """Search probe requests per search surface (6 known-item, 3 attribute, 3 typo, 1 nonsense, 2 filters, and 2 for
    pagination); 0 turns the probe off."""


class ElementInfo(_Frozen):
    """An element as the inventory was measured with it (from data/elements)."""

    id: str
    name: str
    group: str
    scope: Scope
    version: int = Field(ge=1)
    hypotheses: list[str] = Field(default_factory=list)
    """Category checklist hypotheses (data/categories) the element stands for or serves."""
    expose: dict[str, str] = Field(default_factory=dict)
    """Where a store should expose it: ucp, shopify (metafields), schema_org, policy_page."""


class Evidence(_Frozen):
    quote: str = Field(max_length=QUOTE_CHARS)
    source: str
    """A page URL or a tool name (`ucp_mcp get_product`), as redacted."""
    detector: str
    """`<element>@v<version>` and the probe that fired."""
    product_id: str | None = None


class Cell(_Frozen):
    """One element on one surface."""

    element: str
    surface: Surface
    state: State
    products: int = Field(default=0, ge=0)
    """Sampled products this surface was read for and the element applies to (0 for a store-level element)."""
    present: int = Field(default=0, ge=0)
    partial: int = Field(default=0, ge=0)
    coverage: float | None = None
    """present / products for a product element, rounded to four places; None for a store element or no products."""
    reason: str = ""
    """Why a cell is unknown, or a short note on how it was decided."""
    evidence: list[Evidence] = Field(default_factory=list, max_length=MAX_EVIDENCE)


class SurfaceRead(_Frozen):
    surface: Surface
    status: ReadStatus
    source: str = ""
    """The endpoint, the capture the replies came from, or the pages' host."""
    products: int = Field(default=0, ge=0)
    requests: int = Field(default=0, ge=0)
    note: str = ""


class SampledProduct(_Frozen):
    id: str
    title: str = ""
    url: str = ""
    surfaces: list[Surface] = Field(default_factory=list)
    """The surfaces this product was read on."""


class KnownItem(_Frozen):
    asked: int = 0
    found: int = 0
    recall_at_k: float | None = None
    mrr: float | None = None


class AttributeQueries(_Frozen):
    asked: int = 0
    with_results: int = 0
    precision_at_k: float | None = None


class TypoQueries(_Frozen):
    asked: int = 0
    found: int = 0
    tolerance: float | None = None


class FilterCheck(_Frozen):
    offered: bool = False
    """The search tool takes this filter."""
    accepted: bool | None = None
    """The store answered a filtered search without an error; None when not tried."""
    honoured: bool | None = None
    """Every result of the filtered search satisfies the filter; None when not tried or nothing came back."""


class Pagination(_Frozen):
    offered: bool | None = None
    """The first page offered a next page; None when not tried."""
    distinct: bool | None = None
    """The second page held products the first did not; None when not tried."""


class SearchScores(_Frozen):
    """The search probe on one agent search surface. Exact definitions: carttrace.live.search_probe."""

    surface: Surface
    tool: str
    k: int = Field(ge=1)
    queries: int = Field(ge=0)
    known_item: KnownItem = Field(default_factory=KnownItem)
    attribute: AttributeQueries = Field(default_factory=AttributeQueries)
    typo: TypoQueries = Field(default_factory=TypoQueries)
    zero_result_rate: float | None = None
    nonsense_results: bool | None = None
    filters: dict[str, FilterCheck] = Field(default_factory=dict)
    pagination: Pagination = Field(default_factory=Pagination)
    errors: int = Field(default=0, ge=0)
    notes: list[str] = Field(default_factory=list)


class Gap(_Frozen):
    element: str
    kind: GapKind
    agent_state: State
    """The element's state on UCP over MCP."""
    people_state: State
    """Its best state on the store's pages (HTML, JSON-LD, microdata)."""
    seen_on: list[Surface] = Field(default_factory=list)
    """The surfaces where it is present or partial."""
    fix: str
    """What to do, in one plain sentence."""
    addable: bool = False
    """The inventory holds values the detail_added knob can serve on the store's replica."""


class Addition(_Frozen):
    """Values the store's own pages show for products whose agent surface lacks the element."""

    element: str
    field: str
    """Where it goes on the replica's UCP product: a top-level field (`rating`), a path under `metadata`
    (`metadata.size_chart`), or `policies` for a store-level policy served with every product."""
    views: list[View] = Field(default_factory=lambda: list(DETAIL_ONLY))
    source: Surface
    """The surface the first value came from; each product's value is taken from the first surface, in the
    element's `add.from` order, that shows the element present for that product."""
    values: dict[str, Any]
    """Product id to the value; `*` for a store-level value."""


class InventoryBody(_Frozen):
    """Everything an inventory holds but its id: what the id hashes, every default included."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.inventory.v1"] = Field(default=INVENTORY_SCHEMA, alias="schema")
    store: StoreRef
    category: str
    day: str = Field(pattern=_DAY.pattern)
    observed_at: str
    finished_at: str
    ended: Ended
    budget: InventoryBudget
    requests: int = Field(ge=0)
    """HTTP requests sent to the store, all surfaces included."""
    elements_sha: str
    """Hash of the element catalog the cells were detected with."""
    elements: list[ElementInfo]
    surfaces: list[SurfaceRead]
    sample: list[SampledProduct] = Field(default_factory=list)
    capture: str | None = None
    """The capture (cap-<sha12>) whose replies stood for UCP over MCP, when one did."""
    cells: list[Cell]
    search: list[SearchScores] = Field(default_factory=list)
    gaps: list[Gap] = Field(default_factory=list)
    additions: list[Addition] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    rights: Rights
    stripped: bool = False
    """Store content (quotes, sampled titles and URLs, additions, notes) was removed after the retention window."""


class StoreDataInventory(InventoryBody):
    """A sealed inventory. Build one with `seal_inventory`; validation refuses one whose id is not its content."""

    id: str = Field(pattern=_ID.pattern)

    @model_validator(mode="after")
    def _sealed(self) -> Self:
        expected = inventory_id(self.body())
        if self.id != expected:
            raise ValueError(f"inventory id {self.id} does not match its content ({expected})")
        known = {e.id for e in self.elements}
        stray = sorted({c.element for c in self.cells} - known)
        if stray:
            raise ValueError(f"cells name elements the inventory does not list: {stray}")
        return self

    def body(self) -> dict[str, Any]:
        return self.model_dump(mode="json", by_alias=True, exclude={"id"})

    def to_bytes(self) -> bytes:
        return dumps(self.model_dump(mode="json", by_alias=True)).encode("utf-8")

    def cell(self, element: str, surface: Surface) -> Cell | None:
        return next((c for c in self.cells if c.element == element and c.surface == surface), None)

    def element(self, element: str) -> ElementInfo | None:
        return next((e for e in self.elements if e.id == element), None)

    def addition(self, element: str) -> Addition | None:
        return next((a for a in self.additions if a.element == element), None)

    def path(self, root: Path) -> Path:
        return inventory_path(root, self.store.store_id, self.day)

    def save(self, root: Path) -> Path:
        """Write `<root>/<store>/<day>.json`, replacing an earlier inventory of the same store and day."""
        path = self.path(root)
        atomic_write_bytes(path, self.to_bytes())
        return path


def inventory_id(body: Mapping[str, Any]) -> str:
    return f"inv-{digest(body, 12)}"


def inventory_path(root: Path, store_id: str, day: str) -> Path:
    if not _DAY.match(day):
        raise ValidationError(f"{day!r} is not a YYYY-MM-DD day")
    return root / store_dir(store_id) / f"{day}.json"


def seal_inventory(*, salt: str = "", **parts: Any) -> StoreDataInventory:
    """A StoreDataInventory from its parts (plain JSON or models), redacted with the trace's one table and keyed by
    `salt` (trace.redaction_salt_env), with the id computed from its whole body, defaults included."""
    draft = {"schema": INVENTORY_SCHEMA, **{k: _plain(v) for k, v in parts.items()}}
    try:
        plain = loads(dumps(draft))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"an inventory holds something JSON cannot: {exc}") from None
    redacted, _ = redact_json(plain, salt=salt)
    try:
        body = InventoryBody.model_validate(redacted).model_dump(mode="json", by_alias=True)
    except PydanticValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        place = ".".join(str(p) for p in first["loc"]) or "document"
        raise ValidationError(f"the inventory is not a valid {INVENTORY_SCHEMA}: {place}: {first['msg']}") from None
    return _validated({**body, "id": inventory_id(body)}, where="the inventory")


def load_inventory(path: Path) -> StoreDataInventory:
    """An inventory from its file. Refuses one that is not canonical or does not validate."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise NotFoundError(f"no inventory at {path}", hint="take one with `carttrace live inventory`") from None
    try:
        raw = loads(data)
    except ValueError as exc:
        raise ValidationError(f"{path} is not JSON: {exc}", hint="inventories are written by carttrace live") from None
    if not isinstance(raw, Mapping) or raw.get("schema") != INVENTORY_SCHEMA:
        found = raw.get("schema") if isinstance(raw, Mapping) else type(raw).__name__
        raise ValidationError(f"{path} declares schema {found!r}; this code reads {INVENTORY_SCHEMA}")
    inventory = _validated(raw, where=str(path))
    if inventory.to_bytes() != data:
        raise ValidationError(f"{path} is not the canonical form of {inventory.id}", hint="never edit one by hand")
    return inventory


def inventory_files(root: Path, store_ids: Sequence[str] = ()) -> list[Path]:
    """Inventory files under `root`, in the given stores' directories (every store's when none is given), sorted."""
    if not root.is_dir():
        return []
    dirs = [root / store_dir(s) for s in dict.fromkeys(store_ids)] if store_ids else sorted(root.iterdir())
    return sorted(p for d in dirs if d.is_dir() for p in d.iterdir() if _FILE.match(p.name))


def latest_inventories(root: Path, *, category: str | None = None) -> dict[str, StoreDataInventory]:
    """Store id to its newest inventory (of one category when given). A file that does not load is skipped:
    `load_inventory` says why when asked for it."""
    out: dict[str, StoreDataInventory] = {}
    for path in inventory_files(root):
        try:
            inventory = load_inventory(path)
        except (ValidationError, NotFoundError):
            continue
        if category is not None and inventory.category != category:
            continue
        key = inventory.store.store_id
        if key not in out or (inventory.day, inventory.id) > (out[key].day, out[key].id):
            out[key] = inventory
    return dict(sorted(out.items()))


def find_inventory(root: Path, ident: str) -> StoreDataInventory:
    """The inventory with this id (inv-<sha12>) under `root`."""
    if not _ID.match(ident):
        raise ValidationError(f"{ident!r} is not an inventory id", hint="inventory ids look like inv-<sha12>")
    for path in reversed(inventory_files(root)):
        try:
            inventory = load_inventory(path)
        except (ValidationError, NotFoundError):
            continue
        if inventory.id == ident:
            return inventory
    raise NotFoundError(
        f"no inventory {ident} under {root}",
        hint="inventories are stripped of their values after live.retention_days_raw days; take a new one",
    )


def strip_inventory(inventory: StoreDataInventory) -> StoreDataInventory:
    """The inventory without store content: no quotes, sampled titles or URLs, additions or notes; states, shares,
    search scores and gaps stay. Stripping a stripped inventory changes nothing."""
    if inventory.stripped:
        return inventory
    body = inventory.body()
    body["cells"] = [{**cell, "evidence": []} for cell in body["cells"]]
    body["sample"] = [{"id": p["id"], "surfaces": p["surfaces"]} for p in body["sample"]]
    body["additions"] = []
    body["notes"] = []
    body["search"] = [{**s, "notes": []} for s in body["search"]]
    body["gaps"] = [{**g, "addable": False} for g in body["gaps"]]
    body["stripped"] = True
    validated = InventoryBody.model_validate(body).model_dump(mode="json", by_alias=True)
    return _validated({**validated, "id": inventory_id(validated)}, where="the stripped inventory")


def expired(inventory: StoreDataInventory, *, now: datetime, retention_days: int) -> bool:
    """Whether the inventory is older than the retention window."""
    return datetime.fromisoformat(inventory.observed_at) < now - timedelta(days=retention_days)


def iter_cells(inventories: Sequence[StoreDataInventory], element: str) -> Iterator[tuple[str, Cell]]:
    """(store id, cell) for every cell of `element` across inventories."""
    for inventory in inventories:
        for cell in inventory.cells:
            if cell.element == element:
                yield inventory.store.store_id, cell


def _validated(raw: Mapping[str, Any], *, where: str) -> StoreDataInventory:
    try:
        return StoreDataInventory.model_validate(raw)
    except PydanticValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        place = ".".join(str(p) for p in first["loc"]) or "document"
        raise ValidationError(
            f"{where} is not a valid {INVENTORY_SCHEMA}: {place}: {first['msg']}",
            hint="inventories are written by `carttrace live inventory` and never edited",
        ) from None


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return value
