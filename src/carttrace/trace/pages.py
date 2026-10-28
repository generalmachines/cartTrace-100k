"""The product region of a store page, kept so a website verdict can be replayed (D69).

Why keep anything, and why only this. Until D69 a website episode's page bodies were dropped at write time (only
their length was kept), because page text can name reviewers and no redaction rule can find a name in free HTML. That
left a website verdict that could not be replayed: the judge's evidence was the facts the surface read, with nothing
to read them from again. So the writer now keeps, per page, the part a verdict rests on, cleaned so that what could
name a person is gone before the first byte is staged:

- **The product region only:** the page's `<main>`, else the element marked up as a schema.org Product, else the
  body without its header, footer, navigation and asides. Navigation, menus and footers are site chrome, not
  evidence.
- **Review blocks stripped whole:** any element whose id, class, itemprop, itemtype, role or data attributes name a
  review, a rating widget, testimonials, questions and answers or comments (Yotpo, Judge.me, Okendo, Stamped, Loox,
  Bazaarvoice, Trustpilot, Junip, Reviews.io, Feefo, Shopify product reviews), with everything inside it. A reviewer's
  name and a review's text never appear outside such a block on the platforms we shop.
- **No scripts, styles, forms or embeds:** script, style, noscript, template, iframe, svg, canvas, form, input,
  button, select and textarea elements are dropped, and so are comments. JSON-LD is the exception: it is the store's
  own structured product data, which the data audit reads, so it is kept as parsed JSON beside the HTML with every
  review and author removed (and then passes the trace's JSON redaction, whose review rule also applies).
- **Only plain attributes:** href, src, alt, title, id, class, itemprop, itemtype, itemscope, content, datetime,
  value, name, lang, and data attributes naming a product, variant, price or availability. Event handlers, styles and
  everything else go.
- **Bounded:** the cleaned HTML is cut at `MAX_HTML_CHARS` at a tag boundary, and says it was cut.

The kept HTML's sha256 is recorded beside it, so a replay can show it read the very bytes the verdict was made from.
The cleaner's version is recorded too: a change to these rules is a new version, and old bundles keep what they kept.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from html import escape
from html.parser import HTMLParser
from typing import Any, Final

from carttrace.core.canon import loads, sha256_hex

CLEANER: Final = "mc.trace.pages/1"
MAX_HTML_CHARS: Final = 60_000
"""What a kept product region may hold, in characters; a region longer than this is cut, and says so. Not yet
measured on live pages (bodies were never kept before D69): the first website rounds under D69 should check how
often the cut fires and move the bound if verdict evidence is cut off."""
MAX_JSON_LD: Final = 8
"""JSON-LD blocks kept per page; a page with more is a theme repeating itself."""
MAX_JSON_LD_CHARS: Final = 20_000

_DROP: Final = frozenset(
    {
        "script", "style", "noscript", "template", "iframe", "svg", "canvas", "form", "input", "button", "select",
        "textarea", "object", "embed", "video", "audio", "picture", "source", "link", "meta", "title",
    }
)  # fmt: skip
"""Elements dropped with everything inside them (JSON-LD scripts are read before they go)."""
_CHROME: Final = frozenset({"header", "footer", "nav", "aside", "dialog"})
"""Site chrome, dropped when the region is the body."""
_VOID: Final = frozenset(
    {"area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "param", "source", "track", "wbr"}
)
_KEEP_ATTRS: Final = frozenset(
    {
        "href", "src", "alt", "title", "id", "class", "itemprop", "itemtype", "itemscope", "content", "datetime",
        "value", "name", "lang",
    }
)  # fmt: skip
_DATA_ATTR: Final = re.compile(r"data-(?:product|variant|price|sku|availability|stock|option)[a-z0-9_-]*")
_REVIEW: Final = re.compile(
    r"review|rating|testimonial|yotpo|jdgm|judge-?me|okendo|stamped|loox|bazaarvoice|\bbv[-_]|trustpilot|junip"
    r"|reviews-?io|feefo|spr-|question|\bqa\b|q-?and-?a|comment|ugc|author|customer-?photos",
    re.IGNORECASE,
)
"""What marks an element as a review block, read over its id, class, itemprop, itemtype, role and data attributes."""
_JSON_LD_DROP: Final = frozenset({"review", "reviews", "author", "creator", "comment", "comments"})
"""JSON-LD keys removed with their values: reviews and the people who wrote them."""
_SPACE: Final = re.compile(r"\s+")


@dataclass(slots=True)
class _Node:
    tag: str
    attrs: list[tuple[str, str]]
    children: list[_Node | str] = field(default_factory=list)


class _Tree(HTMLParser):
    """A forgiving tree of the page, built with the standard library parser: unclosed tags close with their parent,
    and nothing is fetched or run. Dropped elements never enter the tree; JSON-LD is collected on the way past."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = _Node("#root", [])
        self.stack: list[_Node] = [self.root]
        self.skipping: list[str] = []
        self.json_ld: list[str] = []
        self._ld: list[str] | None = None
        self.removed: dict[str, int] = {"reviews": 0, "dropped": 0}

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if self.skipping:
            if tag not in _VOID:
                self.skipping.append(tag)
            return
        pairs = [(k.lower(), v or "") for k, v in attrs]
        if tag == "script" and any(k == "type" and "ld+json" in v.lower() for k, v in pairs):
            self._ld = []
        if tag in _DROP or _is_review(pairs):
            self.removed["reviews" if tag not in _DROP else "dropped"] += 1
            if tag not in _VOID:
                self.skipping.append(tag)
            return
        node = _Node(tag, pairs)
        self.stack[-1].children.append(node)
        if tag not in _VOID:
            self.stack.append(node)

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        if tag not in _VOID:
            self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        if self.skipping:
            if tag in self.skipping:
                while self.skipping and self.skipping.pop() != tag:
                    pass
            if not self.skipping and tag == "script" and self._ld is not None:
                self.json_ld.append("".join(self._ld))
                self._ld = None
            return
        for depth in range(len(self.stack) - 1, 0, -1):
            if self.stack[depth].tag == tag:
                del self.stack[depth:]
                return

    def handle_data(self, data: str) -> None:
        if self._ld is not None:
            self._ld.append(data)
            return
        if not self.skipping and data:
            self.stack[-1].children.append(data)


def _is_review(attrs: list[tuple[str, str]]) -> bool:
    for name, value in attrs:
        data = name.startswith("data-")
        if (name in ("id", "class", "itemprop", "itemtype", "role") or data) and (
            _REVIEW.search(value) or (data and _REVIEW.search(name))
        ):
            return True
    return False


@dataclass(frozen=True, slots=True)
class KeptPage:
    """A page's kept product region: what goes into a website page's wire record in place of its body."""

    html: str
    json_ld: list[Any]
    region: str
    """main, product, or body: where the region was found."""
    chars: int
    """Characters in the page as served."""
    truncated: bool
    removed_reviews: int
    """Review blocks stripped from the region."""

    def record(self) -> dict[str, Any]:
        """The wire record's response body for this page."""
        return {
            "kept": "product_region",
            "cleaner": CLEANER,
            "region": self.region,
            "html": self.html,
            "sha256": sha256_hex(self.html),
            "json_ld": self.json_ld,
            "chars": self.chars,
            "kept_chars": len(self.html),
            "truncated": self.truncated,
            "removed_reviews": self.removed_reviews,
        }


def keep_page(html: str) -> KeptPage:
    """The product region of `html`, cleaned of review blocks, scripts and chrome, bounded; deterministic."""
    tree = _Tree()
    tree.feed(html)
    tree.close()
    region, node = _region(tree.root)
    parts: list[str] = []
    _render(node, parts, chrome=region == "body", top=True)
    text = "".join(parts).strip()
    truncated = len(text) > MAX_HTML_CHARS
    if truncated:
        text = text[: _cut_at(text)] + "<!-- cut -->"
    return KeptPage(
        html=text,
        json_ld=_json_ld(tree.json_ld),
        region=region,
        chars=len(html),
        truncated=truncated,
        removed_reviews=tree.removed["reviews"],
    )


def _cut_at(text: str) -> int:
    """Where a too-long region is cut: after the last closing tag that ends within the bound, so no element is cut
    in half; at the bound itself when there is none."""
    start = text.rfind("</", 0, MAX_HTML_CHARS)
    while start > 0:
        end = text.find(">", start) + 1
        if 0 < end <= MAX_HTML_CHARS:
            return end
        start = text.rfind("</", 0, start)
    return MAX_HTML_CHARS


def _region(root: _Node) -> tuple[str, _Node]:
    main = _find(root, lambda n: n.tag == "main" or ("role", "main") in n.attrs)
    if main is not None:
        return "main", main
    product = _find(root, lambda n: any(k == "itemtype" and v.rstrip("/").endswith("/Product") for k, v in n.attrs))
    if product is not None:
        return "product", product
    body = _find(root, lambda n: n.tag == "body")
    return "body", body if body is not None else root


def _find(node: _Node, match: Callable[[_Node], bool]) -> _Node | None:
    stack = [node]
    while stack:
        current = stack.pop(0)
        if match(current):
            return current
        stack[:0] = [child for child in current.children if isinstance(child, _Node)]
    return None


def _render(node: _Node, out: list[str], *, chrome: bool, top: bool = False) -> None:
    if not top:
        if chrome and node.tag in _CHROME:
            return
        out.append(f"<{node.tag}{_attrs(node.attrs)}>")
    for child in node.children:
        if isinstance(child, str):
            text = _SPACE.sub(" ", child)
            if text.strip():
                out.append(escape(text, quote=False))
        else:
            _render(child, out, chrome=chrome)
    if not top and node.tag not in _VOID:
        out.append(f"</{node.tag}>")


def _attrs(attrs: list[tuple[str, str]]) -> str:
    kept = [
        (name, value)
        for name, value in attrs
        if (name in _KEEP_ATTRS or _DATA_ATTR.fullmatch(name)) and not value.lower().lstrip().startswith("javascript:")
    ]
    return "".join(f' {name}="{escape(value)}"' if value else f" {name}" for name, value in kept)


def _json_ld(blocks: list[str]) -> list[Any]:
    out: list[Any] = []
    for block in blocks[:MAX_JSON_LD]:
        if len(block) > MAX_JSON_LD_CHARS:
            continue
        try:
            value = loads(block)
        except ValueError:
            continue
        out.append(_without_reviews(value))
    return out


def _without_reviews(value: Any) -> Any:
    if isinstance(value, Mapping):
        if str(value.get("@type", "")).lower() in ("review", "person", "comment", "question", "answer"):
            return None
        return {
            str(k): _without_reviews(v) for k, v in value.items() if str(k).lower().lstrip("@") not in _JSON_LD_DROP
        }
    if isinstance(value, list):
        kept = [_without_reviews(item) for item in value]
        return [item for item in kept if item is not None]
    return value
