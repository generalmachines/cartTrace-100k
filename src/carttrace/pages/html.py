"""A store's HTML page as data: its schema.org JSON-LD and microdata, its visible text, links, images, attributes and
widget markers, read with a plain HTML parser and no script ever run (decision D19).

Why parse and never drive. The data audit asks what a store shows people on its product pages, so it can tell the
store what agents miss, and D2 keeps browsers away from live stores. A page as served already holds most of it:
- **JSON-LD** (`<script type="application/ld+json">`): Shopify themes describe the product, its offers and brand;
  review apps (Judge.me, Yotpo, Okendo) often add an AggregateRating and reviews; `@graph`, lists and nested nodes are
  flattened into one list of typed nodes, property names without their `schema:` or `https://schema.org/` prefix;
- **microdata** (`itemscope`, `itemtype`, `itemprop`): the same vocabulary inline, read into the same node shape;
- **visible text**: the page's main region (`<main>`, `role="main"` or `#MainContent`) as lines, table cells joined
  by " | " as carttrace.bench's readers expect, and apart from it the chrome around it (announcement bars, footers);
- **links, product images with their alt text, attributes** (`data-*`, `aria-label`, `itemprop`, `name`), the class
  and id tokens a widget's markup carries, the option pickers (`name="options[Size]"`), tables, and spec rows
  (`<dt>`/`<dd>` pairs and two-cell table rows);
- for the website surface (decision D31), which shops a store without an agent protocol through its own pages: each
  link's area (nav, header, footer or main), the `<meta>` tags (OpenGraph's og: and product: properties), the forms
  with their method, action and inputs (a form is read for its address, never submitted), the OpenSearch
  description links, and the embedded JSON a theme renders its variants from (`application/json` and Magento's
  `text/x-magento-init` scripts), kept as text and never run.

Reviewer names never leave this module in text. The subtree of any element whose class or id names a list of
reviews, or whose itemtype is a Review, is left out of the text, so a quote taken from the page can never carry a
reviewer's name; review elements are counted, never quoted. Everything is bounded: at most `MAX_HTML` characters are
parsed and every list is capped.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from html.parser import HTMLParser
from typing import Any
from urllib.parse import urljoin, urlsplit

from carttrace.trace.jsonld import parse_block

MAX_HTML = 2_000_000
"""Characters of a page parsed; the rest is ignored, and the facts say so."""
MAX_LINKS = 600
"""Links kept outside the main region, and again inside it."""
MAX_IMAGES = 120
MAX_ATTRS = 4000
MAX_MARKERS = 6000
MAX_NODES = 400
ATTR_CHARS = 200
TEXT_CHARS = 200_000
MAX_META = 200
MAX_FORMS = 40
MAX_INPUTS = 40
MAX_DATA_SCRIPTS = 120
DATA_SCRIPT_CHARS = MAX_HTML
"""A data script is kept whole up to the page's own bound: Store 084's swatch config is one 475,070-character
`text/x-magento-init` script (2026-10-13 smoke), and a truncated one is not JSON, so a cut lost every variant."""
DATA_SCRIPT_TYPES = frozenset({"application/json", "text/x-magento-init"})
"""Script types whose text is data a theme renders from (Shopify's variant JSON, Magento's swatch config)."""
CODE_SCRIPT_TYPES = frozenset({"", "text/javascript", "application/javascript", "module"})
"""Script types whose text is code: read only for the JSON a page assigns to its app state, never run."""
MAX_STATE_SCRIPTS = 12
MAX_PRICE_MARKS = 200
MAX_PICKERS = 30
MAX_PICKER_VALUES = 120
NEAR_CART = 400
"""Elements between a price and an add-to-cart control for the price to count as the one sold there."""
STATE_NAMES = frozenset({"__NEXT_DATA__", "__NUXT_DATA__", "__APOLLO_STATE__"})
"""Ids of application/json scripts that hold an app's whole state (Next.js, Nuxt, Apollo)."""
_STATE_ASSIGNMENT = re.compile(
    r"(?:window\.|self\.|globalThis\.|\bvar\s+|\blet\s+|\bconst\s+)"
    r"(?P<name>__[A-Z][A-Z0-9_]{1,40}__|spConfig|productJson|productJSON|product_json)\s*=\s*"
    r"(?:new\s+Product\.Config\(\s*|JSON\.parse\(\s*)?(?=[\[{\"])"
)
"""`window.__INITIAL_STATE__ = {...}`, `var spConfig = new Product.Config({...})` (Magento 1), and the same assigned
through `JSON.parse("...")`: the JSON a page hands its scripts, read as data."""
_CURRENCY_MARK = re.compile(
    r"[$€£¥₹₩₪₫₱฿]|\b(?:USD|EUR|GBP|CAD|AUD|NZD|INR|JPY|CHF|SEK|NOK|DKK|PLN|AED|SGD|HKD|MXN|BRL|ZAR|Rs\.?|kr)\b"
)
_CART_TOKEN = re.compile(r"add-?to-?(?:cart|bag|basket)|addtocart|addtobag|product-addtocart|btn-cart", re.IGNORECASE)
_CART_TEXT = re.compile(r"^\s*(?:add to (?:cart|bag|basket)|add to shopping (?:cart|bag))\b", re.IGNORECASE)
_OLD_PRICE = re.compile(r"old-?price|price-?old|strike|was-?price|compare|msrp|list-?price|crossed", re.IGNORECASE)
_PLACEHOLDER = re.compile(r"^(?:select|choose|pick|please select|--)", re.IGNORECASE)
_NOT_OPTION = re.compile(
    r"^(?:sort|show|limit|per page|view|qty|quantity|country|region|language|currency|state|province)\b|"
    r"\b(?:sort by|items per page|results per page)\b",
    re.IGNORECASE,
)
"""Picker names that are a page's toolbar or a form's address fields, never a product option."""
_DWVAR = re.compile(r"^dwvar_[^_]+_(?P<name>.+)$")
_AREAS = {"nav": "nav", "header": "header", "footer": "footer"}
_ROLE_AREAS = {"navigation": "nav", "banner": "header", "contentinfo": "footer"}

_SKIPPED = frozenset({"script", "style", "noscript", "template", "svg", "iframe", "head", "object", "canvas"})
_VOID = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)
_BLOCK = frozenset(
    {
        "address", "article", "aside", "blockquote", "br", "caption", "dd", "div", "dl", "dt", "figcaption", "figure",
        "footer", "form", "h1", "h2", "h3", "h4", "h5", "h6", "header", "hr", "li", "nav", "ol", "p", "pre", "section",
        "table", "tbody", "tfoot", "thead", "tr", "ul", "details", "summary", "fieldset", "legend", "label", "option",
    }
)  # fmt: skip
_CELLS = frozenset({"td", "th"})
_CELL = "\x1f"
_KEPT_ATTRS = frozenset({"aria-label", "itemprop", "name", "content", "property"})
_REVIEW_SCOPE = re.compile(
    r"(?:^|[-_])(?:reviews?|testimonials?)(?:$|[-_])|^jdgm-rev|^jdgm-widget$|^yotpo-review|^stamped-review|"
    r"^oke-review|^okereviews|^spr-review|^loox-review|^junip-review|^ruk_review",
    re.IGNORECASE,
)
"""Class or id tokens of an element that holds reviews: its text is never collected."""
_SCHEMA_PREFIX = re.compile(r"^(?:https?://schema\.org/|schema:)", re.IGNORECASE)
_OPTION_NAME = re.compile(r"^options\[(?P<name>[^\]]{1,60})\]$")
_SPACES = re.compile(r"[ \t\f\v\r\xa0]+")
PRODUCT_TYPES = frozenset({"Product", "ProductGroup", "IndividualProduct", "ProductModel"})


@dataclass(frozen=True, slots=True)
class Link:
    text: str
    href: str
    area: str = ""
    """Where the link sits: nav, header, footer, main, or "" elsewhere."""


@dataclass(frozen=True, slots=True)
class FormInput:
    name: str
    type: str
    value: str = ""


@dataclass(frozen=True, slots=True)
class Form:
    """A form as served: where it goes, how, and the inputs it carries. Read for its address only; never sent."""

    action: str
    method: str
    """Lowercase; "get" when the form names none, as HTML does."""
    inputs: tuple[FormInput, ...] = ()
    search: bool = False
    """The form says it searches: role="search", a type="search" input, or "search" in its action, id or class."""
    sku: str = ""
    """The sku an add-to-cart form names (Magento's `data-product-sku`)."""


@dataclass(frozen=True, slots=True)
class Image:
    src: str
    alt: str


@dataclass(frozen=True, slots=True)
class PriceMark:
    """A price the markup shows outside schema.org (decision D36): Magento's price box (`data-price-amount` with its
    `data-price-type`), SFCC's sales value (`content` on `.sales .value`, inside the `data-pid` it prices), or the
    visible text of a price element that carries a currency marker."""

    kind: str
    """"magento_price_box", "sfcc_price" or "page_text"."""
    value: str
    """The amount attribute, or the element's visible text."""
    product: str = ""
    """The nearest `data-product-id` or `data-pid` around it: the product the price is shown for."""
    price_type: str = ""
    """Magento's data-price-type ("finalPrice", "oldPrice"), SFCC's "sales" or "list", "old" for a struck price."""
    main: bool = False
    near_cart: bool = False
    """An add-to-cart control is within NEAR_CART elements of it."""


@dataclass(frozen=True, slots=True)
class Picker:
    """An option picker as served: a <select> (its label and its options' texts) or a group of swatches."""

    name: str
    values: tuple[str, ...]
    kind: str
    """"select" or "swatch"."""
    field: str = ""
    """The control's own name (`super_attribute[306]`, `options[2617]`, `dwvar_A123_color`)."""


@dataclass(frozen=True, slots=True)
class PageFacts:
    """Everything the audit reads from one page. `jsonld` and `microdata` hold schema.org nodes (dicts whose `@type`
    is a list of type names); every other field is what a person sees or what the markup says."""

    url: str
    title: str = ""
    canonical: str = ""
    jsonld: tuple[dict[str, Any], ...] = ()
    jsonld_errors: int = 0
    """JSON-LD blocks that were not JSON."""
    microdata: tuple[dict[str, Any], ...] = ()
    main_text: str = ""
    chrome_text: str = ""
    links: tuple[Link, ...] = ()
    images: tuple[Image, ...] = ()
    """Images in the main region."""
    attrs: tuple[tuple[str, str], ...] = ()
    markers: frozenset[str] = frozenset()
    """Class and id tokens, lowercase."""
    options: tuple[tuple[str, tuple[str, ...]], ...] = ()
    """Option pickers: (name, values) from name="options[Name]" selects and inputs."""
    tables: tuple[str, ...] = ()
    """Each table of the main region as lines, cells joined by " | "."""
    specs: tuple[tuple[str, str], ...] = ()
    """(name, value) rows: <dt>/<dd> pairs and two-cell table rows of the main region."""
    meta: tuple[tuple[str, str], ...] = ()
    """(property or name, lowercase; content) of every <meta> tag with both, in page order."""
    forms: tuple[Form, ...] = ()
    search_links: tuple[str, ...] = ()
    """OpenSearch description URLs (<link rel="search" type="application/opensearchdescription+xml">)."""
    data_scripts: tuple[str, ...] = ()
    """The text of application/json and text/x-magento-init scripts, and a WooCommerce variations form's
    `data-product_variations` JSON, as data; never run."""
    state_scripts: tuple[tuple[str, str], ...] = ()
    """(name, JSON text) of the app state a page embeds: `__NEXT_DATA__`, `window.__INITIAL_STATE__ = {...}`,
    Magento 1's `spConfig`. Read as data; never run."""
    prices: tuple[PriceMark, ...] = ()
    pickers: tuple[Picker, ...] = ()
    cart_controls: int = 0
    """Add-to-cart controls on the page (forms that add to a cart, buttons named or labelled so)."""
    truncated: bool = False

    def nodes(self, source: str) -> tuple[dict[str, Any], ...]:
        return self.jsonld if source == "jsonld" else self.microdata

    def products(self, source: str) -> list[dict[str, Any]]:
        """The schema.org Product (or ProductGroup) nodes of one source ("jsonld" or "microdata"), in page order."""
        return [n for n in self.nodes(source) if is_product(n)]


def parse_page(html: str, url: str) -> PageFacts:
    """The facts of one page as served."""
    parser = _Page(url)
    parser.feed(html[:MAX_HTML])
    parser.close()
    if parser.form is not None:
        parser.close_form()
    jsonld, errors = _jsonld(parser.scripts)
    main, chrome = _lines("".join(parser.main)), _lines("".join(parser.chrome))
    if not parser.main_found:
        main, chrome = chrome, ""
    return PageFacts(
        url=url,
        title=parser.h1 or " ".join(parser.title.split()),
        canonical=parser.canonical,
        jsonld=tuple(jsonld[:MAX_NODES]),
        jsonld_errors=errors,
        microdata=tuple(parser.items[:MAX_NODES]),
        main_text=main[:TEXT_CHARS],
        chrome_text=chrome[:TEXT_CHARS],
        links=tuple(parser.links),
        images=tuple(parser.images),
        attrs=tuple(parser.attrs),
        markers=frozenset(parser.markers),
        options=tuple((name, tuple(values)) for name, values in parser.options.items()),
        tables=tuple(t for t in (_lines("".join(parts)) for parts in parser.tables) if t),
        specs=tuple(parser.specs),
        meta=tuple(parser.meta),
        forms=tuple(parser.done_forms),
        search_links=tuple(parser.search_links),
        data_scripts=tuple(parser.data_scripts),
        state_scripts=tuple(parser.states),
        prices=tuple(_near_cart(parser.marks, parser.cart_positions)),
        pickers=tuple(_pickers(parser.selects, parser.swatches, parser.labels)),
        cart_controls=len(parser.cart_positions),
        truncated=len(html) > MAX_HTML,
    )


def state_json(code: str) -> list[tuple[str, str]]:
    """The JSON a script assigns to its app state, by name: each `<name> = {...}` (or `new Product.Config({...})`,
    or `JSON.parse("...")`) whose value parses as JSON. Nothing is run; a value that is not JSON is left out."""
    out: list[tuple[str, str]] = []
    decoder = json.JSONDecoder()
    for found in _STATE_ASSIGNMENT.finditer(code):
        try:
            value, end = decoder.raw_decode(code, found.end())
        except ValueError:
            continue
        if isinstance(value, str):  # JSON.parse("..."): the string is the JSON
            try:
                json.loads(value)
            except ValueError:
                continue
            text = value
        else:
            text = code[found.end() : end]
        out.append((found.group("name"), text))
        if len(out) >= MAX_STATE_SCRIPTS:
            break
    return out


def _near_cart(marks: list[tuple[int, PriceMark]], carts: list[int]) -> Iterator[PriceMark]:
    for position, mark in marks:
        near = any(abs(position - cart) <= NEAR_CART for cart in carts)
        yield PriceMark(mark.kind, mark.value, mark.product, mark.price_type, mark.main, near)


def _pickers(
    selects: list[tuple[dict[str, str], list[str]]], swatches: dict[str, list[str]], labels: Mapping[str, str]
) -> Iterator[Picker]:
    """Selects named by their <label for>, aria-label or own name; swatch groups by their data-attr. A picker whose
    only name is a number or whose values are all placeholders is left out."""
    count = 0
    for values, options in selects:
        field = values.get("name", "")
        dwvar = _DWVAR.match(field)
        bracket = _OPTION_NAME.match(field)
        name = (
            labels.get(values.get("id", ""), "")
            or values.get("aria-label", "")
            or values.get("data-attribute-name", "")
            or values.get("data-attr", "")
            or (dwvar.group("name") if dwvar else "")
            or (bracket.group("name") if bracket and not bracket.group("name").isdigit() else "")
            or (field if field and not re.search(r"[\d\[\]]", field) else "")
        )
        name = " ".join(name.replace("*", " ").split()).strip(" :")[:60]
        kept = tuple(dict.fromkeys(o for o in options if o and not _PLACEHOLDER.match(o)))[:MAX_PICKER_VALUES]
        if name and kept and count < MAX_PICKERS and not _NOT_OPTION.search(name) and not _NOT_OPTION.search(field):
            count += 1
            yield Picker(name=name, values=kept, kind="select", field=field[:80])
    for name, options in swatches.items():
        if options and count < MAX_PICKERS:
            count += 1
            yield Picker(name=name[:60], values=tuple(options[:MAX_PICKER_VALUES]), kind="swatch", field=name[:80])


def schema_name(value: str) -> str:
    """A schema.org type or property name without its prefix: "https://schema.org/InStock" -> "InStock"."""
    text = _SCHEMA_PREFIX.sub("", value.strip())
    return text.rsplit("/", 1)[-1] if "://" in text else text


def is_product(node: Mapping[str, Any]) -> bool:
    types = node.get("@type")
    return isinstance(types, list) and any(t in PRODUCT_TYPES for t in types)


def walk_nodes(value: Any) -> Iterator[dict[str, Any]]:
    """Every typed node in a JSON-LD value, depth first: @graph members, list items and nested objects."""
    if isinstance(value, list):
        for item in value:
            yield from walk_nodes(item)
    elif isinstance(value, Mapping):
        if "@type" in value:
            yield dict(value)
        for key, item in value.items():
            if key != "@context":
                yield from walk_nodes(item)


def same_store(url: str, host: str) -> bool:
    """Whether a URL is on `host` or its www twin."""
    other = (urlsplit(url).hostname or "").lower()
    return other.removeprefix("www.") == host.lower().removeprefix("www.")


def _jsonld(scripts: list[str]) -> tuple[list[dict[str, Any]], int]:
    nodes: list[dict[str, Any]] = []
    errors = 0
    for raw in scripts:
        if not raw.strip():
            continue
        value = parse_block(raw)
        if value is None:
            errors += 1
            continue
        nodes.extend(walk_nodes(_normal_value(value)))
    return nodes, errors


def _normal_value(value: Any) -> Any:
    """JSON-LD with prefixed property and type names made plain, and every @type a list, recursively."""
    if isinstance(value, list):
        return [_normal_value(v) for v in value]
    if isinstance(value, Mapping):
        out: dict[str, Any] = {}
        for key, item in value.items():
            if key == "@type":
                types = item if isinstance(item, list) else [item]
                out["@type"] = [schema_name(t) for t in types if isinstance(t, str)]
            else:
                out[str(key) if str(key).startswith("@") else schema_name(str(key))] = _normal_value(item)
        return out
    return value


def _lines(text: str) -> str:
    out = []
    for raw in text.split("\n"):
        cells = [_SPACES.sub(" ", cell).strip() for cell in raw.split(_CELL)]
        line = " | ".join(c for c in cells if c)
        if line:
            out.append(line)
    return "\n".join(out)


@dataclass(slots=True)
class _Open:
    """One open element and what closing it must undo."""

    tag: str
    skipped: bool = False
    excluded: bool = False
    main: bool = False
    item: dict[str, Any] | None = None
    """The microdata item this element opened."""
    capture: tuple[list[str], dict[str, Any], list[str]] | None = None
    """(property names, their item, the text collected) for an itemprop whose value is its text."""
    link: list[str] | None = None
    href: str = ""
    table: list[str] | None = None
    """This table's text, when the element is a table of the main region."""
    cells: list[str] | None = None
    """A table row's cell texts."""
    text: list[str] | None = None
    """Text collected for a cell, a <dt> or a <dd>."""
    select: str | None = None
    h1: bool = False
    area: str = ""
    """nav, header or footer when this element is one (by tag or ARIA role)."""
    pid: str = ""
    """The product id this element names (`data-product-id`, `data-pid`), for the prices inside it."""
    price_scope: str = ""
    """"sales" or "list" on an SFCC price wrapper; "old" on a struck or former price."""
    price_text: list[str] | None = None
    """The visible text of a price element, the outermost one."""
    label_for: str | None = None
    label_text: list[str] | None = None
    picker: tuple[dict[str, str], list[str]] | None = None
    """A <select>'s attributes and its options' texts."""
    option_text: list[str] | None = None
    swatch_group: str = ""
    cart_text: list[str] | None = None
    """A button's text, checked for "add to cart" when it closes."""


class _Page(HTMLParser):
    def __init__(self, url: str) -> None:
        super().__init__(convert_charrefs=True)
        self.url = url
        self.stack: list[_Open] = []
        self.scripts: list[str] = []
        self.script: list[str] | None = None
        self.main: list[str] = []
        self.chrome: list[str] = []
        self.main_found = False
        self.title = ""
        self.in_title = False
        self.h1 = ""
        self.canonical = ""
        self.links: list[Link] = []
        self.main_links = 0
        """Links of the main region, which have a cap of their own: a mega menu never crowds out a results grid."""
        self.images: list[Image] = []
        self.attrs: list[tuple[str, str]] = []
        self.markers: set[str] = set()
        self.options: dict[str, list[str]] = {}
        self.tables: list[list[str]] = []
        self.specs: list[tuple[str, str]] = []
        self.pending_dt = ""
        self.items: list[dict[str, Any]] = []
        self.item_stack: list[dict[str, Any]] = []
        self.meta: list[tuple[str, str]] = []
        self.done_forms: list[Form] = []
        self.form: tuple[dict[str, str], list[FormInput]] | None = None
        self.search_links: list[str] = []
        self.data_scripts: list[str] = []
        self.data_script: list[str] | None = None
        self.data_script_id = ""
        self.code_script: list[str] | None = None
        self.code_chars = 0
        self.states: list[tuple[str, str]] = []
        self.position = 0
        """Elements started so far: how far a price sits from an add-to-cart control."""
        self.marks: list[tuple[int, PriceMark]] = []
        self.cart_positions: list[int] = []
        self.labels: dict[str, str] = {}
        self.selects: list[tuple[dict[str, str], list[str]]] = []
        self.swatches: dict[str, list[str]] = {}

    # -- state ---------------------------------------------------------------------------------------------------

    def _any(self, flag: str) -> bool:
        return any(getattr(o, flag) for o in self.stack)

    def _region(self) -> list[str] | None:
        """Where text goes now: None inside a skipped or a review element, else the main region or the chrome."""
        if self._any("skipped") or self._any("excluded"):
            return None
        return self.main if self._any("main") else self.chrome

    def _area(self) -> str:
        """The innermost nav, header or footer around the current element, else main inside the main region."""
        for opened in reversed(self.stack):
            if opened.area:
                return opened.area
        return "main" if self._any("main") else ""

    def _innermost(self, attribute: str) -> _Open | None:
        return next((o for o in reversed(self.stack) if getattr(o, attribute) is not None), None)

    # -- tags ------------------------------------------------------------------------------------------------------

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        values = {k.lower(): (v or "") for k, v in attrs}
        opened = _Open(tag)
        self.position += 1
        self._markers_and_attrs(values)
        self._woo_variations(tag, values)
        self._head_and_forms(tag, values)
        if tag == "script":
            opened.skipped = True
            self._open_script(values)
        elif tag in _SKIPPED:
            opened.skipped = True
        if tag == "title" and not any(o.tag == "body" for o in self.stack):
            self.in_title = True
        if not self.main_found and (
            tag == "main" or values.get("role", "").lower() == "main" or values.get("id") == "MainContent"
        ):
            opened.main = True
            self.main_found = True
        if _review_scope(values):
            opened.excluded = True
        opened.area = _AREAS.get(tag) or _ROLE_AREAS.get(values.get("role", "").strip().lower(), "")
        if tag == "link" and "canonical" in values.get("rel", "").lower().split() and values.get("href"):
            self.canonical = urljoin(self.url, values["href"])
        self._microdata(tag, values, opened)
        if not opened.skipped and not self._any("skipped"):
            self._commerce(tag, values, opened)
        self.stack.append(opened)
        self._content(tag, values, opened)
        if tag in _VOID:
            self.stack.pop()
            self._close(opened)

    def _open_script(self, values: Mapping[str, str]) -> None:
        """Which kept text a script holds: JSON-LD, data (a theme's JSON), or inline code read for its app state."""
        kind = values.get("type", "").strip().lower()
        if kind == "application/ld+json":
            self.script = []
        elif kind in DATA_SCRIPT_TYPES and len(self.data_scripts) < MAX_DATA_SCRIPTS:
            self.data_script = []
            self.data_script_id = values.get("id", "")
        elif kind in CODE_SCRIPT_TYPES and not values.get("src") and len(self.states) < MAX_STATE_SCRIPTS:
            self.code_script = []
            self.code_chars = 0

    def _close_script(self) -> None:
        if self.script is not None:
            self.scripts.append("".join(self.script))
            self.script = None
        if self.data_script is not None:
            text = "".join(self.data_script)[:DATA_SCRIPT_CHARS]
            self.data_scripts.append(text)
            if self.data_script_id in STATE_NAMES and len(self.states) < MAX_STATE_SCRIPTS:
                self.states.append((self.data_script_id, text))
            self.data_script = None
        if self.code_script is not None:
            code = "".join(self.code_script)
            if "=" in code and ("__" in code or "Config" in code or "product" in code):
                self.states.extend(state_json(code)[: MAX_STATE_SCRIPTS - len(self.states)])
            self.code_script = None

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self.in_title = False
        if not any(o.tag == tag for o in self.stack):
            return
        while self.stack:
            opened = self.stack.pop()
            self._close(opened)
            if opened.tag == tag:
                break

    def _script_data(self, data: str) -> bool:
        """Text inside a script this parser keeps (JSON-LD, data, code read for its app state): kept, and True."""
        if self.script is not None:
            self.script.append(data)
        elif self.data_script is not None:
            self.data_script.append(data)
        elif self.code_script is not None:
            if self.code_chars < DATA_SCRIPT_CHARS:
                self.code_script.append(data)
                self.code_chars += len(data)
        else:
            return False
        return True

    def handle_data(self, data: str) -> None:
        if self._script_data(data):
            return
        if self.in_title:
            self.title += data
        if self._any("skipped"):
            return
        clean = data.replace(_CELL, " ")
        struck = max((i for i, o in enumerate(self.stack) if o.price_scope == "old"), default=-1)
        for index, opened in enumerate(self.stack):
            if opened.capture is not None:
                opened.capture[2].append(clean)
            if opened.link is not None:
                opened.link.append(clean)
            if opened.price_text is not None and index >= struck and len(opened.price_text) < 40:
                opened.price_text.append(clean)  # a struck price inside a price element is not its price
            for kept in (opened.label_text, opened.option_text, opened.cart_text):
                if kept is not None and len(kept) < 40:
                    kept.append(clean)
        holder = self._innermost("text")
        if holder is not None and holder.text is not None:
            holder.text.append(clean)
        region = self._region()
        if region is None:
            return
        region.append(clean)
        table = self._innermost("table")
        if table is not None and table.table is not None:
            table.table.append(clean)

    # -- pieces ----------------------------------------------------------------------------------------------------

    def _head_and_forms(self, tag: str, values: Mapping[str, str]) -> None:
        """<meta> tags, OpenSearch links, forms and their inputs, wherever they sit (the head is otherwise skipped)."""
        if tag == "meta":
            key = (values.get("property") or values.get("name") or values.get("itemprop") or "").strip().lower()
            if key and "content" in values and len(self.meta) < MAX_META:
                self.meta.append((key[:80], " ".join(values["content"].split())[: ATTR_CHARS * 5]))
        elif tag == "link":
            rels = values.get("rel", "").lower().split()
            kind = values.get("type", "").lower()
            if "search" in rels and "opensearchdescription" in kind and values.get("href"):
                self.search_links.append(urljoin(self.url, values["href"])[:500])
        elif tag == "form":
            if self.form is not None:
                self.close_form()
            if len(self.done_forms) < MAX_FORMS:
                self.form = (dict(values), [])
        elif tag in ("input", "select", "textarea", "button") and self.form is not None:
            inputs = self.form[1]
            name = values.get("name", "").strip()
            if name and len(inputs) < MAX_INPUTS:
                kind = values.get("type", "text" if tag == "input" else tag).strip().lower()
                inputs.append(FormInput(name=name[:80], type=kind[:20], value=values.get("value", "")[:200]))

    def close_form(self) -> None:
        """The open form, read into a Form."""
        assert self.form is not None
        values, inputs = self.form
        self.form = None
        action = urljoin(self.url, values.get("action", "").strip() or self.url)
        method = (values.get("method", "").strip().lower() or "get")[:10]
        tokens = " ".join((values.get("id", ""), values.get("class", ""), action)).lower()
        search = (
            values.get("role", "").lower() == "search" or any(i.type == "search" for i in inputs) or "search" in tokens
        )
        sku = values.get("data-product-sku", "").strip()[:80]
        self.done_forms.append(Form(action=action[:500], method=method, inputs=tuple(inputs), search=search, sku=sku))

    def _commerce(self, tag: str, values: Mapping[str, str], opened: _Open) -> None:
        """What a page shows of buying outside schema.org (D36): the product ids elements name, price boxes and price
        texts, add-to-cart controls, labels, selects and swatches."""
        opened.pid = (values.get("data-pid") or values.get("data-product-id") or "").strip()[:80]
        tokens = values.get("class", "").lower().split()
        self._prices(tag, values, opened, tokens)
        if self._is_cart(tag, values, tokens):
            self.cart_positions.append(self.position)
        elif tag in ("button", "a"):
            opened.cart_text = []
        self._controls(tag, values, opened)

    def _prices(self, tag: str, values: Mapping[str, str], opened: _Open, tokens: list[str]) -> None:
        classes = values.get("class", "")
        if "sales" in tokens:
            opened.price_scope = "sales"
        elif "list" in tokens or "strike-through" in tokens:
            opened.price_scope = "list"
        if tag in ("del", "s", "strike") or _OLD_PRICE.search(classes):
            opened.price_scope = "old"
        pid = next((o.pid for o in reversed(self.stack) if o.pid), "")
        main = self._any("main")
        if values.get("data-price-amount", "").strip() and len(self.marks) < MAX_PRICE_MARKS:
            own = re.search(r"product-price-(\d+)", values.get("id", ""))
            mark = PriceMark(
                "magento_price_box",
                values["data-price-amount"].strip()[:40],
                product=pid or (own.group(1) if own else ""),
                price_type=values.get("data-price-type", "").strip()[:40],
                main=main,
            )
            self.marks.append((self.position, mark))
        scope = next((o.price_scope for o in reversed(self.stack) if o.price_scope), "")
        if "value" in tokens and values.get("content", "").strip() and scope and len(self.marks) < MAX_PRICE_MARKS:
            mark = PriceMark("sfcc_price", values["content"].strip()[:40], product=pid, price_type=scope, main=main)
            self.marks.append((self.position, mark))
        named = f"{classes} {values.get('id', '')} {values.get('itemprop', '')}".lower()
        if (
            "price" in named
            and self._region() is not None
            and not any(o.price_text is not None for o in self.stack)
            and len(self.marks) < MAX_PRICE_MARKS
        ):
            opened.price_text = []

    def _controls(self, tag: str, values: Mapping[str, str], opened: _Open) -> None:
        if tag == "label" and values.get("for"):
            opened.label_for = values["for"]
            opened.label_text = []
        if tag == "select" and len(self.selects) < MAX_PICKERS * 2:
            in_cart = self.form is not None and "cart" in self.form[0].get("action", "").lower()
            area = self._area()
            if in_cart or area == "main" or (not self.main_found and area not in ("nav", "header", "footer")):
                opened.picker = (dict(values), [])
        elif tag == "option" and any(o.picker is not None for o in self.stack):
            opened.option_text = []
        if values.get("data-attr") and not values.get("data-attr-value"):
            opened.swatch_group = values["data-attr"].strip()[:60]
        group = next((o.swatch_group for o in reversed(self.stack) if o.swatch_group), "")
        value = values.get("data-attr-value") or values.get("data-option-label") or ""
        if group and value.strip():
            kept = self.swatches.setdefault(group, [])
            if value.strip() not in kept and len(kept) < MAX_PICKER_VALUES:
                kept.append(value.strip()[:80])

    def _is_cart(self, tag: str, values: Mapping[str, str], tokens: list[str]) -> bool:
        """An add-to-cart control: a form that posts to the cart, or a control whose id, class, name or label says
        add to cart (a button's text is checked when it closes)."""
        if tag == "form":
            action = values.get("action", "").lower()
            return "cart/add" in action or action.rstrip("/").endswith("/cart/add")
        if tag not in ("button", "input", "a"):
            return False
        words = " ".join(
            (values.get("id", ""), " ".join(tokens), values.get("name", ""), values.get("data-action", ""))
        )
        if _CART_TOKEN.search(words):
            return True
        label = values.get("aria-label", "") or (values.get("value", "") if tag == "input" else "")
        return bool(_CART_TEXT.match(label))

    def _close_commerce(self, opened: _Open) -> None:
        if opened.price_text is not None:
            text = " ".join("".join(opened.price_text).split())[:120]
            if text and _CURRENCY_MARK.search(text) and re.search(r"\d", text):
                pid = opened.pid or next((o.pid for o in reversed(self.stack) if o.pid), "")
                old = opened.price_scope == "old" or any(o.price_scope == "old" for o in self.stack)
                mark = PriceMark(
                    "page_text", text, product=pid, price_type="old" if old else "", main=self._any("main")
                )
                self.marks.append((self.position, mark))
        if opened.cart_text is not None and _CART_TEXT.match(" ".join("".join(opened.cart_text).split())):
            self.cart_positions.append(self.position)
        if opened.label_for is not None and opened.label_text is not None:
            text = " ".join("".join(opened.label_text).split())[:80]
            if text:
                self.labels.setdefault(opened.label_for, text)
        if opened.option_text is not None:
            holder = next((o.picker for o in reversed(self.stack) if o.picker is not None), None)
            text = " ".join("".join(opened.option_text).split())[:80]
            if holder is not None and text and len(holder[1]) < MAX_PICKER_VALUES:
                holder[1].append(text)
        if opened.picker is not None:
            self.selects.append(opened.picker)

    def _woo_variations(self, tag: str, values: Mapping[str, str]) -> None:
        """A WooCommerce variable product's variations, which its add-to-cart form carries as JSON in the
        `data-product_variations` attribute (the parser has unescaped it), kept as a data script: too long for
        `attrs`, and the only place the page names each variation's id, price and stock. A form whose variations
        load by script says "false" there, and is left alone."""
        raw = values.get("data-product_variations", "") if tag == "form" else ""
        if raw.lstrip().startswith("[") and len(self.data_scripts) < MAX_DATA_SCRIPTS:
            self.data_scripts.append(raw[:DATA_SCRIPT_CHARS])

    def _markers_and_attrs(self, values: Mapping[str, str]) -> None:
        if len(self.markers) < MAX_MARKERS:
            self.markers.update(token.lower()[:80] for token in values.get("class", "").split())
            if values.get("id"):
                self.markers.add(values["id"].lower()[:80])
        for name, value in values.items():
            if len(self.attrs) >= MAX_ATTRS:
                break
            if (name.startswith("data-") or name in _KEPT_ATTRS) and value.strip() and len(value) <= ATTR_CHARS * 5:
                self.attrs.append((name, " ".join(value.split())[:ATTR_CHARS]))

    def _content(self, tag: str, values: Mapping[str, str], opened: _Open) -> None:
        if self._any("skipped"):
            return
        region = self._region()
        if region is not None and (tag in _BLOCK or tag in _CELLS):
            region.append(_CELL if tag in _CELLS else "\n")
        self._inline(tag, values, opened)
        if self._any("main") and region is not None:
            self._table(tag, opened)
        self._options(tag, values, opened)

    def _inline(self, tag: str, values: Mapping[str, str], opened: _Open) -> None:
        """The first heading, links, and the main region's images."""
        if tag == "h1" and not self.h1:
            opened.h1 = True
            opened.text = []
        elif tag == "a" and values.get("href"):
            opened.link = []
            opened.href = urljoin(self.url, values["href"])
        elif tag == "img" and self._any("main") and len(self.images) < MAX_IMAGES:
            src = values.get("src") or values.get("data-src") or values.get("data-srcset", "").split(" ")[0]
            if src:
                self.images.append(Image(src=urljoin(self.url, src), alt=" ".join(values.get("alt", "").split())))

    def _table(self, tag: str, opened: _Open) -> None:
        """Tables, their rows and cells, and definition lists, in the main region."""
        table = self._innermost("table")
        if tag == "table":
            opened.table = []
            self.tables.append(opened.table)
        elif tag == "tr":
            opened.cells = []
            if table is not None and table.table is not None:
                table.table.append("\n")
        elif tag in _CELLS:
            opened.text = []
            if table is not None and table.table is not None:
                table.table.append(_CELL)
        elif tag in ("dt", "dd"):
            opened.text = []

    def _options(self, tag: str, values: Mapping[str, str], opened: _Open) -> None:
        name = _OPTION_NAME.match(values.get("name", ""))
        if name is not None:
            option = self.options.setdefault(name.group("name").strip(), [])
            if tag == "select":
                opened.select = name.group("name").strip()
            elif tag == "input" and values.get("value") and values["value"] not in option:
                option.append(values["value"][:80])
        elif tag == "option" and values.get("value"):
            select = next((o.select for o in reversed(self.stack) if o.select), None)
            if select is not None and values["value"] not in self.options.setdefault(select, []):
                self.options[select].append(values["value"][:80])

    def _microdata(self, tag: str, values: Mapping[str, str], opened: _Open) -> None:
        props = values.get("itemprop", "").split()
        parent = self.item_stack[-1] if self.item_stack else None
        if "itemscope" in values:
            node: dict[str, Any] = {"@type": [schema_name(t) for t in values.get("itemtype", "").split()]}
            if props and parent is not None:
                for prop in props:
                    _add(parent, prop, node)
            elif len(self.items) < MAX_NODES:
                self.items.append(node)
            opened.item = node
            self.item_stack.append(node)
            return
        if not props or parent is None:
            return
        value = _attribute_value(tag, values, self.url)
        if value is not None:
            for prop in props:
                _add(parent, prop, value)
        else:
            opened.capture = (props, parent, [])

    def _close(self, opened: _Open) -> None:
        if opened.tag == "script":
            self._close_script()
        self._close_commerce(opened)
        if opened.tag == "form" and self.form is not None:
            self.close_form()
        if not opened.skipped and (opened.tag in _BLOCK or opened.tag in _CELLS):
            region = self.main if (opened.main or self._any("main")) else self.chrome
            if not opened.excluded and not self._any("excluded") and not self._any("skipped"):
                region.append("\n" if opened.tag not in _CELLS else "")
        if opened.capture is not None:
            props, item, parts = opened.capture
            text = " ".join("".join(parts).split())
            for prop in props if text else ():
                _add(item, prop, text)
        if opened.item is not None and self.item_stack and self.item_stack[-1] is opened.item:
            self.item_stack.pop()
        if opened.link is not None:
            area = self._area()
            main = area == "main"
            if (self.main_links if main else len(self.links) - self.main_links) < MAX_LINKS:
                text = " ".join("".join(opened.link).split())[:200]
                self.links.append(Link(text=text, href=opened.href[:500], area=area))
                self.main_links += int(main)
        if opened.text is not None or opened.cells is not None:
            self._close_text(opened, " ".join("".join(opened.text or ()).split()))

    def _close_text(self, opened: _Open, text: str) -> None:
        """The heading, a table cell into its row, a two-cell row or a <dt>/<dd> pair into the spec rows."""
        if opened.h1:
            self.h1 = text[:300]
        elif opened.tag in _CELLS:
            row = next((o for o in reversed(self.stack) if o.cells is not None), None)
            if row is not None and row.cells is not None:
                row.cells.append(text)
        elif opened.tag == "dt":
            self.pending_dt = text[:100]
        elif opened.tag == "dd" and self.pending_dt:
            if text:
                self.specs.append((self.pending_dt, text[:500]))
            self.pending_dt = ""
        if opened.tag == "tr" and opened.cells is not None and len(opened.cells) == 2 and all(opened.cells):
            self.specs.append((opened.cells[0][:100], opened.cells[1][:500]))


def _review_scope(values: Mapping[str, str]) -> bool:
    if values.get("itemtype", "").lower().rsplit("/", 1)[-1] in ("review", "userreview", "criticreview"):
        return True
    tokens = [*values.get("class", "").split(), values.get("id", "")]
    return any(t and _REVIEW_SCOPE.search(t) for t in tokens)


_VALUE_ATTRIBUTES: tuple[tuple[frozenset[str], str, bool], ...] = (
    (frozenset({"a", "area", "link"}), "href", True),
    (frozenset({"img", "audio", "video", "source", "embed", "iframe", "track"}), "src", True),
    (frozenset({"data", "meter"}), "value", False),
    (frozenset({"time"}), "datetime", False),
    (frozenset({"object"}), "data", True),
)
"""Where an itemprop's value lives on each element (the HTML microdata rules): (tags, attribute, is a URL)."""


def _attribute_value(tag: str, values: Mapping[str, str], base: str) -> str | None:
    """An itemprop's value from the element's attributes, or None when its value is the element's text."""
    if "content" in values:
        return values["content"].strip()
    for tags, attribute, is_url in _VALUE_ATTRIBUTES:
        if tag in tags and attribute in values:
            return urljoin(base, values[attribute]) if is_url else values[attribute]
    return "" if tag in _VOID else None


def _add(item: dict[str, Any], prop: str, value: Any) -> None:
    name = schema_name(prop)
    if name in item:
        current = item[name]
        item[name] = [*current, value] if isinstance(current, list) else [current, value]
    else:
        item[name] = value
