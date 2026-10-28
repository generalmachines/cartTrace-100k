"""How a product document reads as observed keys: the key scheme of the category data-type union (D62), in one place.

Why here. The union (carttrace.bench.union_read) reads every live agent-view reply and every pack (D76) through these
functions, and the sandbox's exposure sheet (carttrace.collect.exposure_sheet, D93) reads what each surface serves through
the same ones, so a key a live reply shows and a key a replica serves resolve to the same type id. carttrace.bench and
carttrace.collect are siblings on one layer and may not import each other, so the reader lives a layer below both. The code
moved here unchanged from carttrace.bench.union_read, which still re-exports every name.
"""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any, Final

from carttrace.agents.details.size_chart import read_chart
from carttrace.agents.products import Product, read_product
from carttrace.core.errors import MCError
from carttrace.pages.html import schema_name

STRUCTURED: Final = "structured"
"""A key the store's data names itself: a column, a schema.org property, a document field."""
LABEL: Final = "label"
"""A free label the store wrote: a spec row's name, a "Label: value" line, a "key: value" tag, an app-state field."""
HEADING: Final = "heading"
"""A short line that recurs on the store's product pages and nowhere on its homepage: a section title."""
OPTION: Final = "option"
"""A variant option's name (Size, Colour, Inseam)."""
FREE_KINDS: Final = frozenset({LABEL, HEADING, OPTION})

Marker = Callable[[str, str, str, Any], None]
"""(key, kind, field path, value): records one key with its example."""

MAX_DEPTH: Final = 8
MAX_LIST: Final = 40

IGNORED_LEAVES: Final = frozenset(
    {
        "id", "gid", "handle", "slug", "position", "created_at", "updated_at", "createdat", "updatedat",
        "admin_graphql_api_id", "product_id", "variant_id", "variant_ids", "src", "url", "href", "width", "height",
        "aspect_ratio", "option1", "option2", "option3", "uid", "uuid", "loading", "error", "success", "cursor",
        "__typename", "typename", "has_next_page", "page", "logo", "sameas", "mainentityofpage", "potentialaction",
        "@context", "@id", "@type", "context", "type_code", "typecode", "modifiedtime", "synchronizationblocked",
        "container", "stylecode", "csrf", "token", "key", "nonce", "seo", "template", "theme", "styleclass",
    }
)  # fmt: skip
"""Leaf field names that are plumbing in every platform's documents (ids, positions, timestamps, sizes of images):
not data a shopper is shown. A field under these names is never a type; the union says so in its docstring."""
IGNORED_PREFIXES: Final = ("preview_image", "featured_image.preview", "featured_media.preview_image", "ucp.")
ENUMERATED: Final = frozenset(
    {
        "media[].media_type", "media[].type", "variants[].media[].type", "Offer.availability", "Offer.itemCondition",
        "AggregateOffer.availability", "Product.itemCondition", "links[].type", "totals[].type", "messages[].code",
        "messages[].type", "variants[].availability.status", "variants[].inventory_policy",
        "MerchantReturnPolicy.returnPolicyCategory", "MerchantReturnPolicy.returnFees",
    }
)  # fmt: skip
"""Fields whose value says which kind of thing it is (a video among images, a preorder among stock states): each
value becomes its own key, `path=value`, beside the field itself."""
AVAILABILITY_LEAVES: Final = frozenset({"available", "availability", "is_in_stock", "instock", "in_stock"})
"""Boolean fields where false is data (sold out), not absence."""
COUNT_WORDS: Final = ("inventory", "stock", "quantity")
DEFAULT_ONE: Final = frozenset({"min", "increment"})
"""Quantity rules whose value 1 is the platform's default (buy one, in ones), not a limit the store set."""
"""Numeric fields where zero is data (none left), not absence."""
_LABEL_LINE: Final = re.compile(r"^\s*([^\W\d_][^:|]{0,40}?)\s*:\s*(.*)$")
_TAG_LABEL: Final = re.compile(r"^\s*([^\W\d_][^:]{1,30}?)\s*[:=]\s*(\S.*)$")
_DIGIT: Final = re.compile(r"\d")
_CAMEL: Final = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
NOT_LABELS: Final = frozenset({"http", "https", "www", "mailto", "tel"})
"""What a "Label:" pattern catches that is a URL's scheme, not a label."""


def slug(text: str) -> str:
    """A label as a stable id segment: accents folded, camelCase split, lower case, runs of anything else as one
    hyphen. "Fabric & Care" and "fabric_care" are both `fabric-care`; "numberOfReviews" is `number-of-reviews`."""
    plain = unicodedata.normalize("NFKD", _CAMEL.sub(" ", text)).encode("ascii", "ignore").decode().lower()
    plain = re.sub(r"['’`]", "", plain)
    return re.sub(r"[^a-z0-9]+", "-", plain).strip("-")


def label_slug(text: str) -> str | None:
    """A free label's slug, or None when it is not a label: empty, a number or size, longer than 40 characters or
    five words, or not starting with a letter."""
    words = text.strip().split()
    if not words or len(words) > 5 or len(text.strip()) > 40:
        return None
    out = slug(text)
    if len(out) < 2 or not out[0].isalpha() or out.replace("-", "").isdigit() or out in NOT_LABELS:
        return None
    return out


def heading_slug(line: str) -> str | None:
    """A line that may be a section heading: short, no digits (prices, sizes, counts), at most five words."""
    text = line.strip()
    if not 3 <= len(text) <= 40 or _DIGIT.search(text) or text.endswith((".", "!", "?", ",")):
        return None
    return label_slug(text)


def shown(path: str, value: Any) -> bool:
    """Whether a field shows something: not empty; false only where false is data (sold out), zero only where zero is
    data (none in stock)."""
    leaf = path.rsplit(".", 1)[-1].removesuffix("[]").lower()
    if value is None or value in ("", [], {}):
        return False
    if isinstance(value, bool):
        return value or leaf in AVAILABILITY_LEAVES
    if leaf in DEFAULT_ONE and value in (1, "1"):
        return False
    if isinstance(value, int | float) and value == 0:
        return any(word in leaf for word in COUNT_WORDS)
    return not (isinstance(value, str) and not value.strip())


def flatten(value: Any, path: str = "", depth: int = 0) -> Iterator[tuple[str, Any]]:
    """(path, value) for every leaf of a JSON document, lists marked `[]`; a list of scalars is one leaf."""
    if depth > MAX_DEPTH:
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = str(key)
            yield from flatten(item, f"{path}.{name}" if path else name, depth + 1)
    elif isinstance(value, list):
        if value and all(not isinstance(v, Mapping | list) for v in value):
            yield path, value
            return
        for item in value[:MAX_LIST]:
            yield from flatten(item, f"{path}[]", depth + 1)
    else:
        yield path, value


def ignored(path: str) -> bool:
    if path.startswith(IGNORED_PREFIXES):
        return True
    leaf = path.rsplit(".", 1)[-1].removesuffix("[]").lower()
    return leaf in IGNORED_LEAVES or leaf.startswith("@")


def enum_value(value: Any) -> str | None:
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip().rsplit("/", 1)[-1]
    out = slug(text)
    return out or None


def _read_fields(doc: Mapping[str, Any], prefix: str, mark: Marker) -> None:
    """Every field with a value, once per document, and an enumerated field's value as its own key."""
    seen: set[str] = set()
    for path, value in flatten(doc):
        if path.startswith("metadata.") or ignored(path) or "compare_at" in path or "list_price" in path:
            continue
        if not shown(path, value):
            continue
        key = f"{prefix}:{path}"
        if key not in seen:
            seen.add(key)
            mark(key, STRUCTURED, path, value)
        if path in ENUMERATED and (value_slug := enum_value(value)) is not None:
            qualified = f"{key}={value_slug}"
            if qualified not in seen:
                seen.add(qualified)
                mark(qualified, STRUCTURED, path, value)


def _read_doc_labels(doc: Mapping[str, Any], prefix: str, mark: Marker) -> None:
    """A document's option names, metafield keys and "key: value" tags: labels the store chose."""
    for option in doc.get("options") or () if isinstance(doc.get("options"), list) else ():
        name = option.get("name") if isinstance(option, Mapping) else None
        label = label_slug(str(name)) if isinstance(name, str) else None
        if label is not None:
            mark(f"option:{label}", OPTION, "options[].name", name)
    metadata = doc.get("metadata")
    if isinstance(metadata, Mapping):
        for name, value in sorted(metadata.items(), key=lambda kv: str(kv[0])):
            label = label_slug(str(name).replace("_", " ").replace(".", " "))
            if label is not None and shown(str(name), value):
                mark(f"{prefix}_meta:{label}", LABEL, f"metadata.{name}", value)
    for raw_tag in doc.get("tags") or () if isinstance(doc.get("tags"), list) else ():
        match = _TAG_LABEL.match(str(raw_tag))
        label = label_slug(match.group(1)) if match else None
        if match and label is not None:
            mark(f"tag:{label}", LABEL, "tags[]", str(raw_tag))


def read_text(lines: Sequence[str], mark_label: Marker) -> None:
    """A text's "Label: value" lines (a label alone on its line takes the next line as its value) and its heading
    candidates. Headings are kept or dropped per store later, once it is known which recur (`heading_ok`)."""
    for index, line in enumerate(lines):
        match = _LABEL_LINE.match(line)
        if match:
            label = label_slug(match.group(1))
            value = match.group(2).strip() or (lines[index + 1].strip() if index + 1 < len(lines) else "")
            if label is not None and value and len(match.group(1)) <= 40:
                mark_label(f"label:{label}", LABEL, "text", f"{match.group(1).strip()}: {value}")
                continue
        head = heading_slug(line)
        if head is not None:
            mark_label(f"heading:{head}", HEADING, "text", line.strip())


def compare_at(doc: Mapping[str, Any]) -> tuple[str, str] | None:
    """(field, "was X, now Y") when a variant's (or the document's own) compare-at or list price is above its price:
    a zero or equal compare-at price is the platform's default, not a sale."""
    variants = [v for v in doc.get("variants") or () if isinstance(v, Mapping)]
    for variant, prefix in [*((v, "variants[].") for v in variants), (doc, "")]:
        price = _amount(variant.get("price"))
        for name in ("compare_at_price", "list_price"):
            was = _amount(variant.get(name))
            if price is not None and was is not None and was > price:
                return f"{prefix}{name}", f"was {_shown_amount(was)}, now {_shown_amount(price)}"
    low, listed = _amount(get_in(doc, "price_range", "min")), _amount(get_in(doc, "list_price_range", "min"))
    if low is not None and listed is not None and listed > low:
        return "list_price_range", f"list {listed}, price {low}"
    return None


def _shown_amount(value: float) -> str:
    return str(int(value)) if value == int(value) else f"{value:.2f}"


def get_in(doc: Mapping[str, Any], *keys: str) -> Any:
    value: Any = doc
    for key in keys:
        value = value.get(key) if isinstance(value, Mapping) else None
    return value


def _amount(value: Any) -> float | None:
    if isinstance(value, Mapping):
        value = value.get("amount")
    if isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.replace(",", ""))
        except ValueError:
            return None
    return None


def product_of(doc: Mapping[str, Any], product_id: str) -> Product | None:
    """The product reader's view of a document (its description as lines and tables), or None when it cannot read
    it; a feed's or .js document's HTML description is read the way a UCP description is."""
    raw = {k: v for k, v in doc.items() if k in ("title", "description", "body_html", "metadata")}
    try:
        return read_product({**raw, "id": product_id})
    except (MCError, ValueError, TypeError):
        return None


def mark_document(doc: Mapping[str, Any], prefix: str, product_id: str, mark: Marker) -> None:
    """Every key one product document shows, its compare-at price against its price, and its description's labels,
    headings, size chart and "key: value" tags, each passed to `mark` as (key, kind, field path, value)."""
    _read_fields(doc, prefix, mark)
    found = compare_at(doc)
    if found is not None:
        mark(f"{prefix}:{found[0]}", STRUCTURED, found[0], found[1])
    _read_doc_labels(doc, prefix, mark)
    product = product_of(doc, product_id)
    if product is not None:
        read_text(product.lines, mark)
        chart = read_chart(product)
        if chart is not None:
            mark("text:size_chart", STRUCTURED, "description table", "size chart: " + ", ".join(chart.measures()))


SKIPPED_FIELDS: Final = frozenset({"provenance"})
"""A pack product's bookkeeping, never a data type (D76)."""


def _flat_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    """A twin's metadata groups its attributes (`attributes`, `details`); each attribute is a label of its own."""
    out: dict[str, Any] = {}
    for name, value in metadata.items():
        if isinstance(value, Mapping) and value:
            for inner, item in value.items():
                out.setdefault(str(inner), item)
        else:
            out.setdefault(str(name), value)
    return out


def served_document(doc: Mapping[str, Any]) -> dict[str, Any]:
    """A pack's product as data: its private fields (`_policies`, `_listing`) and bookkeeping left out, metadata
    groups opened up."""
    out = {k: v for k, v in doc.items() if not str(k).startswith("_") and k not in SKIPPED_FIELDS}
    metadata = out.get("metadata")
    if isinstance(metadata, Mapping):
        out["metadata"] = _flat_metadata(metadata)
    return out


# -- schema.org nodes -------------------------------------------------------------------------------------------------

SKIPPED_SCHEMA: Final = frozenset(
    {
        "BreadcrumbList", "ListItem", "WebSite", "WebPage", "ItemPage", "CollectionPage", "SearchAction",
        "EntryPoint", "SiteNavigationElement", "Organization", "Corporation", "OnlineStore", "Store",
        "LocalBusiness", "ContactPoint", "PostalAddress", "GeoCoordinates", "OpeningHoursSpecification", "Person",
        "ReadAction", "Thing",
    }
)  # fmt: skip
"""schema.org node types on a product page that describe the site, not the product (the homepage's own nodes are read
as store-level keys), and Person, whose values are reviewers'."""
REVIEW_SCHEMA: Final = frozenset({"Review", "Comment", "UserComments"})
SKIPPED_SCHEMA_PROPS: Final = frozenset({"url", "sameAs", "mainEntityOfPage", "potentialAction", "logo", "position"})


def schema_type(node: Mapping[str, Any]) -> str | None:
    raw = node.get("@type")
    first = raw[0] if isinstance(raw, list) and raw else raw
    return schema_name(str(first)) if isinstance(first, str) and first else None


def mark_schema_nodes(nodes: Sequence[Mapping[str, Any]], mark: Callable[..., None]) -> None:
    """The page's schema.org nodes about the product: each property with a value, an enumerated one's value as its
    own key, a PropertyValue's name as a label; a review's values withheld. `mark` takes (key, kind, field
    path, value, withheld=False)."""
    for node in nodes:
        node_type = schema_type(node)
        if node_type is None or node_type in SKIPPED_SCHEMA:
            continue
        if node_type == "PropertyValue":
            label = label_slug(str(node.get("name") or ""))
            if label is not None and shown("value", node.get("value")):
                mark(f"prop:{label}", LABEL, "schema.org PropertyValue", f"{node.get('name')}: {node.get('value')}")
            continue
        for prop, value in sorted(node.items()):
            if prop.startswith("@") or prop in SKIPPED_SCHEMA_PROPS or not shown(prop, value):
                continue
            path = f"{node_type}.{prop}"
            mark(f"schema:{path}", STRUCTURED, f"schema.org {path}", value, withheld=node_type in REVIEW_SCHEMA)
            if path in ENUMERATED and (value_slug := enum_value(value)) is not None:
                mark(f"schema:{path}={value_slug}", STRUCTURED, f"schema.org {path}", value)
