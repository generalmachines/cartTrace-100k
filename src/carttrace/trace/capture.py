"""A store capture, `mc.capture.v1`: everything one polite, read-only live session saw of a store, kept as the hidden
ground truth its replica is built from and checked against (decision D12).

What a capture holds:
- `store`, `category`, `captured_at` and `finished_at`, and why it ended (`ended`);
- the budget it ran under and the search queries it sent;
- the store's `/.well-known/ucp` document, its ProtocolProfile and its payment handlers as declared;
- its tool list with schemas, each marked allowed or refused by the live allowlist;
- every tool call it made (`calls`: arguments without the client's `meta`, the store's reply, the error code);
- every HTTP exchange of the session (`wire`: robots.txt, discovery, the MCP handshake, every tools/call), redacted
  at write time with the trace's one redaction table;
- the products and variants it saw, normalised (prices in minor units of the store's currency) beside the store's own
  document for each (get_product's reply completed with what only search results listed);
- the data rights (D10): a live store's, so training and publication are false, and raw content of a Shopify store
  expires after live.retention_days_raw days (`carttrace trace purge`).

Why here and not in carttrace.live. The live client writes captures, the sandbox builds replicas from them and the retention
purge expires them; the first two sit side by side on layer 4 and cannot import each other, so the schema lives one
layer down, beside the trace bundle schema it shares its wire records and rights with.

Files. `save` writes `<captures dir>/<store>/<id>.json` holding exactly the canonical bytes (carttrace.core.canon), and the id
is `cap-<sha12>` of the content without the id, so the same session against the same replies gives the same bytes,
and `load` refuses a file that is not the canonical form of a valid capture or whose content does not hash to its
name.
"""

from __future__ import annotations

import re
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any, Final, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator
from pydantic import ValidationError as PydanticValidationError

from carttrace.core.canon import digest, dumps, loads
from carttrace.core.errors import NotFoundError, ValidationError
from carttrace.core.files import atomic_write_bytes
from carttrace.protocols.profile import ProtocolProfile
from carttrace.trace.schema import Rights, StoreRef, WireRecord

CAPTURE_SCHEMA: Final = "mc.capture.v1"
CAPTURES_DIR: Final = "captures"
"""Captures live under <paths.var>/captures/<store>/<capture id>.json."""
_ID = re.compile(r"^cap-[0-9a-f]{12}$")
_FILE = re.compile(r"^cap-[0-9a-f]{12}\.json$")
_STORE_DIR = re.compile(r"[^A-Za-z0-9._-]")

CaptureEnd = Literal["complete", "request_budget", "throttled", "blocked"]
"""complete: every query was paged as far as the budget allowed; request_budget: the request budget cut it short;
throttled: the store answered 429 and was paused; blocked: a 401, a 403 or a bot wall stopped it."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CaptureBudget(_Frozen):
    """How much one capture may ask a store for. Every bound is checked before the request it would exceed."""

    products: int = Field(default=40, ge=1, le=500)
    """Products to read in detail (get_product). Paging stops once this many distinct products were found."""
    pages: int = Field(default=3, ge=1, le=20)
    """search_catalog pages read per query at most."""
    page_size: int = Field(default=25, ge=1, le=50)
    """The `pagination.limit` each search asks for; UCP lets the store return fewer."""
    requests: int = Field(default=200, ge=1, le=2000)
    """Tool calls in all (searches, product reads and policy lookups); the handshake and discovery are not counted."""


class CapturedTool(_Frozen):
    """A tool as the store listed it, with the client-filled `meta` argument removed from its schema."""

    name: str
    description: str = ""
    input_schema: dict[str, Any] = Field(default_factory=dict)
    output_schema: dict[str, Any] | None = None
    annotations: dict[str, Any] = Field(default_factory=dict)
    allowed: bool
    """Whether the live allowlist lets an agent call it."""


class CapturedCall(_Frozen):
    """One tool call the capture made, and what the store answered."""

    index: int = Field(ge=0)
    tool: str
    arguments: dict[str, Any]
    """As the capture asked, without `meta` (the client's, never part of the request a replica answers)."""
    is_error: bool = False
    error_code: str | None = None
    reply: Any = None
    """The UCP document the store answered with (MCP structuredContent, else the JSON of the first text block)."""
    wire: list[int] = Field(default_factory=list)
    """The `seq` of each wire record the call sent."""


class OptionPick(_Frozen):
    name: str
    label: str


class CapturedVariant(_Frozen):
    id: str = Field(min_length=1)
    title: str = ""
    price: int = Field(ge=0)
    """Minor units of `currency`, as the store showed it."""
    currency: str = Field(pattern=r"^[A-Z]{3}$")
    available: bool = True
    options: list[OptionPick] = Field(default_factory=list)
    """In the order the store listed them."""


class CapturedProduct(_Frozen):
    id: str = Field(min_length=1)
    title: str
    variants: list[CapturedVariant] = Field(min_length=1)
    """Every variant the store showed with a price, in the store's order."""
    document: dict[str, Any]
    """The store's own product document: get_product's reply, completed with the fields and variants only search
    results carried (carttrace.trace.products.merge_documents)."""
    detailed: bool
    """get_product answered for it; otherwise it is known from search results only."""
    seen_in: list[int] = Field(default_factory=list)
    """Indexes of the calls whose replies held it."""


class CaptureBody(_Frozen):
    """Everything a capture holds but its id: what the id hashes, every default included."""

    model_config = ConfigDict(frozen=True, extra="forbid", populate_by_name=True)

    schema_: Literal["mc.capture.v1"] = Field(default=CAPTURE_SCHEMA, alias="schema")
    store: StoreRef
    """The store as captured: its queue id (or its host when it has none), the host it served from, its category
    and its platform."""
    category: str
    captured_at: str
    finished_at: str
    ended: CaptureEnd
    budget: CaptureBudget
    queries: list[str]
    policy_queries: list[str] = Field(default_factory=list)
    market: str | None = None
    """The `context.address_country` every request named, when the capture was asked for one market."""
    profile: ProtocolProfile
    discovery: dict[str, Any] | None = None
    """The /.well-known/ucp document as the store served it."""
    payment_handlers: dict[str, Any] = Field(default_factory=dict)
    """The handlers the discovery document declared, as declared (namespace to entries)."""
    tools: list[CapturedTool] = Field(default_factory=list)
    calls: list[CapturedCall] = Field(default_factory=list)
    wire: list[WireRecord] = Field(default_factory=list)
    products: list[CapturedProduct] = Field(default_factory=list)
    currency: str | None = Field(default=None, pattern=r"^[A-Z]{3}$")
    """The one currency every captured price is in; None when no product was captured."""
    notes: list[str] = Field(default_factory=list)
    rights: Rights


class StoreCapture(CaptureBody):
    """A sealed capture. Build one with `seal_capture`; validation refuses one whose id is not its content."""

    id: str = Field(pattern=_ID.pattern)

    @model_validator(mode="after")
    def _sealed(self) -> Self:
        expected = capture_id(self.body())
        if self.id != expected:
            raise ValueError(f"capture id {self.id} does not match its content ({expected})")
        currencies = {v.currency for p in self.products for v in p.variants}
        if len(currencies) > 1:
            raise ValueError(f"captured prices are in more than one currency: {sorted(currencies)}")
        if currencies and self.currency not in currencies:
            raise ValueError(f"capture currency {self.currency} is not the currency of its prices")
        return self

    def body(self) -> dict[str, Any]:
        """Everything but the id: what the id hashes."""
        return self.model_dump(mode="json", by_alias=True, exclude={"id"})

    def to_bytes(self) -> bytes:
        return dumps(self.model_dump(mode="json", by_alias=True)).encode("utf-8")

    @property
    def store_dir(self) -> str:
        """The directory under the captures root this capture is saved in: its store id, made file-safe."""
        return store_dir(self.store.store_id)

    def save(self, root: Path) -> Path:
        """Write `<root>/<store>/<id>.json`. Never overwrites a different file: the name is the content."""
        path = root / self.store_dir / f"{self.id}.json"
        data = self.to_bytes()
        if path.exists():
            if path.read_bytes() != data:
                raise ValidationError(
                    f"{path} exists and does not hold {self.id}", hint="the file was edited; remove it"
                )
            return path
        atomic_write_bytes(path, data)
        return path

    def successful(self, *tools: str) -> Iterator[CapturedCall]:
        """The calls that got an answer, in order, of the given tools (every tool when none is given)."""
        for call in self.calls:
            if not call.is_error and (not tools or call.tool in tools):
                yield call


def capture_id(body: Mapping[str, Any]) -> str:
    return f"cap-{digest(body, 12)}"


def seal_capture(**parts: Any) -> StoreCapture:
    """A StoreCapture from its parts (plain JSON or models), with the id computed from its whole body, defaults
    included, so the same parts always give the same id however they were spelled."""
    draft = {"schema": CAPTURE_SCHEMA, **{k: _plain(v) for k, v in parts.items()}}
    try:
        plain = loads(dumps(draft))
    except (TypeError, ValueError) as exc:
        raise ValidationError(f"a capture holds something JSON cannot: {exc}") from None
    try:
        body = CaptureBody.model_validate(plain).model_dump(mode="json", by_alias=True)
    except PydanticValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        place = ".".join(str(p) for p in first["loc"]) or "document"
        raise ValidationError(f"the capture is not a valid {CAPTURE_SCHEMA}: {place}: {first['msg']}") from None
    return _validated({**body, "id": capture_id(body)}, where="the capture")


def store_dir(store_id: str) -> str:
    """A store id (store-004, or a host) as a directory name."""
    name = _STORE_DIR.sub("_", store_id.strip())
    if not name.strip("._"):
        raise ValidationError(f"{store_id!r} cannot name a capture directory", hint="pass a queue id or a host")
    return name


def load_capture(path: Path) -> StoreCapture:
    """A capture from its file. Refuses one that is not canonical, does not validate, or is not the capture its name
    says."""
    try:
        data = path.read_bytes()
    except FileNotFoundError:
        raise NotFoundError(f"no capture at {path}", hint="capture the store with `carttrace live capture`") from None
    try:
        raw = loads(data)
    except ValueError as exc:
        raise ValidationError(f"{path} is not JSON: {exc}", hint="captures are written by `carttrace live capture`") from None
    if not isinstance(raw, Mapping) or raw.get("schema") != CAPTURE_SCHEMA:
        found = raw.get("schema") if isinstance(raw, Mapping) else type(raw).__name__
        raise ValidationError(
            f"{path} declares schema {found!r}; this code reads {CAPTURE_SCHEMA}",
            hint="a purged capture leaves only its purge record; capture the store again",
        )
    capture = _validated(raw, where=str(path))
    if capture.to_bytes() != data:
        raise ValidationError(f"{path} is not the canonical form of {capture.id}", hint="never edit a capture by hand")
    if path.stem != capture.id:
        raise ValidationError(f"{path.name} holds {capture.id}", hint="load captures only under their own names")
    return capture


def find_capture(root: Path, ident: str | Path) -> Path:
    """The file of a capture named by its path, or by its id under `root` (any store directory)."""
    candidate = Path(ident)
    if candidate.is_file():
        return candidate
    text = str(ident)
    if _ID.match(text):
        found = sorted(root.glob(f"*/{text}.json")) if root.is_dir() else []
        if found:
            return found[0]
        raise NotFoundError(f"no capture {text} under {root}", hint="list them with `ls var/captures/*/`")
    raise NotFoundError(f"{text} is neither a capture file nor a capture id", hint="pass a path or a cap-<sha12> id")


def capture_files(root: Path, store_ids: Sequence[str] = ()) -> list[Path]:
    """Capture files under `root`, in the given stores' directories (every store's when none is given), sorted.
    Purge records and other files are not captures."""
    if not root.is_dir():
        return []
    dirs = [root / store_dir(s) for s in dict.fromkeys(store_ids)] if store_ids else sorted(root.iterdir())
    return sorted(p for d in dirs if d.is_dir() for p in d.iterdir() if _FILE.match(p.name))


def is_capture_file(path: Path) -> bool:
    return bool(_FILE.match(path.name))


def newest_captures(root: Path, *, store_ids: Sequence[str] = (), category: str | None = None) -> dict[str, Path]:
    """Store id to its newest capture that holds products (of one category when given), under `root`, in the given
    stores' directories (every store's when none is given). A file that does not load is not a capture and is
    skipped; `load_capture` says why when it is asked for that file."""
    return {store: found[2] for store, found in sorted(_newest(root, store_ids, category).items())}


def latest_capture(root: Path, store_ids: Sequence[str], category: str) -> Path | None:
    """The newest capture with products of one store in a category, looked up under each of its ids (its queue id,
    its host), or None when it has none."""
    found = _newest(root, store_ids, category).values()
    return max(found)[2] if found else None


def _newest(root: Path, store_ids: Sequence[str], category: str | None) -> dict[str, tuple[str, str, Path]]:
    newest: dict[str, tuple[str, str, Path]] = {}
    for path in capture_files(root, store_ids):
        try:
            capture = load_capture(path)
        except (ValidationError, NotFoundError):
            continue
        if not capture.products or (category is not None and capture.category != category):
            continue
        found = (capture.captured_at, capture.id, path)
        store = capture.store.store_id
        if store not in newest or found > newest[store]:
            newest[store] = found
    return newest


def _validated(raw: Mapping[str, Any], *, where: str) -> StoreCapture:
    try:
        return StoreCapture.model_validate(raw)
    except PydanticValidationError as exc:
        first = exc.errors(include_url=False, include_input=False)[0]
        place = ".".join(str(p) for p in first["loc"]) or "document"
        raise ValidationError(
            f"{where} is not a valid {CAPTURE_SCHEMA}: {place}: {first['msg']}",
            hint="captures are written by `carttrace live capture` and never edited",
        ) from None


def _plain(value: Any) -> Any:
    if isinstance(value, BaseModel):
        return value.model_dump(mode="json", by_alias=True)
    if isinstance(value, list | tuple):
        return [_plain(v) for v in value]
    if isinstance(value, Mapping):
        return {str(k): _plain(v) for k, v in value.items()}
    return value
