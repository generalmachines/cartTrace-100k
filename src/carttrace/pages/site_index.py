"""A store's own product index, for an agent on its website: the product URLs its robots-declared sitemaps list,
searched by plain word match (decision D34).

Why. The first website smoke (2026-10-13) found nothing on 8 of 8 episodes: Store 084 (Magento) draws its search
results with script, so the results page held no product link, and Store 177 (custom) shows no search form
and its collection pages repeat the menu. Both stores publish every product in their sitemaps, the index a store
writes for crawlers. So the website surface reads it, once per session, and answers word queries over it:
- **Only addresses the store declares**: the sitemaps robots.txt names, else /sitemap.xml, and the child sitemaps an
  index names (carttrace.live.generic.sitemap_products, the capture pass's walker, capped at `MAX_FILES` files here). No
  other path is guessed; every file goes through the one road (robots per path, throttle, recorder, walls named).
- **Plain word match**: a query's words against each URL's slug and the entry's title (its sitemap `<image:title>`,
  or the capture's title); a word matches its plural. No model, no fuzzy search: a result names the words it matched.
- **Capture hints are marked.** When the run seeds the index with our latest capture's product URLs (config
  `collect.website_capture_hints`), an entry only the capture holds has `source` "capture", never "sitemap", so the
  judge can tell what the agent found from what we handed it. An entry both hold is the store's ("sitemap").
- **The shopper's market first** (D38). A multi-locale sitemap lists every locale's copy of a product, and a US
  shopper landed on Store 454's /th-th/ pages, which show no price. Each entry's locale is read from its sitemap
  entry's hreflang alternates when it has them, else from its path prefix (/en-us/, /us/, /us/en/, /en/), else from
  the sitemap file it came from (/sitemap_us.xml, /en-us/sitemap.xml); an entry with a hreflang alternate in the
  market is replaced by that alternate. When any entry is in the market, entries of other locales are left out (and
  counted); when none is, every locale stays. Each result says its `locale` and `market_match` (None: unknown).
"""

from __future__ import annotations

import itertools
import re
import unicodedata
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final, Literal
from urllib.parse import unquote, urlsplit

IndexSource = Literal["sitemap", "capture"]
MAX_FILES = 8
"""Sitemap files one session reads: an index and its first product files (the walker orders products first)."""
MAX_HINTS = 50_000
STOP_WORDS = frozenset({"a", "an", "and", "the", "of", "for", "with", "in", "on", "to", "by", "my", "pair", "html"})
_WORD = re.compile(r"[a-z0-9]+")
IDENTITY_PARAMS: Final = frozenset(
    {"pid", "product_id", "productid", "prod_id", "prodid", "item_id", "itemid", "id", "sku", "cid", "category_id"}
)
"""Query parameters that name the page itself: Gap's `product.do?pid=`, a script's `?product_id=` or `?id=`, a
category's `?cid=`. Round 10 (D58) found every one of Gap's products under one key, because the query was dropped."""
MARKET_LANGUAGES: Final[Mapping[str, tuple[str, ...]]] = {
    "US": ("en",),
    "CA": ("en", "fr"),
    "GB": ("en",),
    "IE": ("en",),
    "AU": ("en",),
    "NZ": ("en",),
    "SG": ("en",),
    "IN": ("en",),
    "AE": ("en", "ar"),
    "DE": ("de",),
    "FR": ("fr",),
    "IT": ("it",),
    "ES": ("es",),
    "NL": ("nl",),
    "MX": ("es",),
    "JP": ("ja",),
}
"""A market's languages: a language-only locale (/en/, hreflang "en") in one of them is the market's."""
LANGUAGES: Final = frozenset(
    {
        "ar",
        "cs",
        "da",
        "de",
        "el",
        "en",
        "es",
        "fi",
        "fr",
        "he",
        "hu",
        "it",
        "ja",
        "ko",
        "nb",
        "nl",
        "no",
        "pl",
        "pt",
        "ro",
        "ru",
        "sk",
        "sv",
        "th",
        "tr",
        "uk",
        "vi",
        "zh",
    }
)
REGIONS: Final = frozenset(
    {
        "ae",
        "ar",
        "at",
        "au",
        "be",
        "br",
        "ca",
        "ch",
        "cl",
        "cn",
        "co",
        "cz",
        "de",
        "dk",
        "es",
        "fi",
        "fr",
        "gb",
        "gr",
        "hk",
        "ie",
        "il",
        "in",
        "it",
        "jp",
        "kr",
        "mx",
        "my",
        "nl",
        "no",
        "nz",
        "ph",
        "pl",
        "pt",
        "ro",
        "ru",
        "sa",
        "se",
        "sg",
        "th",
        "tr",
        "tw",
        "uk",
        "us",
        "vn",
        "za",
    }
)
"""Two-letter codes a lone path segment is read as a locale by; anything else is a path, not a locale."""
_TAG = re.compile(r"^([a-z]{2})[-_]([a-z]{2})$")
_TOKEN = re.compile(r"[a-z]+")


@dataclass(frozen=True, slots=True)
class Locale:
    """Where an address says it sells: `text` as the store wrote it (lowercased: "en-us", "us/en", "th-th", "en"),
    with its language and region; `either` holds a lone two-letter segment that may be a language or a region."""

    text: str
    language: str | None = None
    region: str | None = None
    either: str | None = None

    def score(self, market: str) -> int:
        """2 in the market's region (in one of its languages, when the locale names one), 1 in one of its languages
        with no region said, or in its region in another language (Store 041's /us/es/ for a US shopper, D58), -1 another
        locale."""
        code = market.lower()
        languages = MARKET_LANGUAGES.get(market.upper(), ())
        if self.either is not None:
            return 2 if self.either == code else 1 if self.either in languages else -1
        if self.region is not None:
            if self.region != code and not (code == "gb" and self.region == "uk"):
                return -1
            return 2 if self.language is None or not languages or self.language in languages else 1
        return 1 if self.language in languages else -1


@dataclass(frozen=True, slots=True)
class IndexEntry:
    url: str
    title: str = ""
    lastmod: str = ""
    source: IndexSource = "sitemap"
    locale: str | None = None
    """The locale the address is in (`Locale.text`), when anything says."""
    market_match: bool | None = None
    """Whether that locale is the shopper's market; None when no market was given or the locale is unknown."""

    def words(self) -> frozenset[str]:
        return frozenset(stems(f"{slug_text(self.url)} {self.title}"))


@dataclass(frozen=True, slots=True)
class Hit:
    entry: IndexEntry
    matched: tuple[str, ...]


@dataclass
class SiteIndex:
    """The product URLs of one store, each address once, with how they came to be known."""

    entries: list[IndexEntry] = field(default_factory=list)
    sitemaps: tuple[str, ...] = ()
    """Sitemap files read, in order."""
    statuses: dict[str, str] = field(default_factory=dict)
    edition: str | None = None
    """The capture edition the hints came from, when any were seeded."""
    market: str | None = None
    other_locales: int = 0
    """Entries left out because they are in another locale and the market's own exist."""
    _words: dict[str, frozenset[str]] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        sitemap: Iterable[IndexEntry],
        hints: Iterable[IndexEntry] = (),
        *,
        sitemaps: Sequence[str] = (),
        statuses: dict[str, str] | None = None,
        edition: str | None = None,
        market: str | None = None,
        alternates: Mapping[str, Mapping[str, str]] | None = None,
        files_of: Mapping[str, str] | None = None,
    ) -> SiteIndex:
        """The store's entries first; a hint whose address the sitemap also lists only lends it a missing title.
        With `market`, each entry's locale is read (its `alternates`, its path, the sitemap file `files_of` names)
        and the market's locale is preferred, as the module docstring says."""
        alts = alternates or {}
        files = files_of or {}
        by_key: dict[str, IndexEntry] = {}
        for entry in sitemap:
            found = _localised(entry, alts.get(entry.url) or {}, files.get(entry.url), market)
            by_key.setdefault(url_key(found.url), found)
        for count, hint in enumerate(hints):
            if count >= MAX_HINTS:
                break
            key = url_key(hint.url)
            have = by_key.get(key)
            if have is None:
                by_key[key] = _localised(IndexEntry(hint.url, hint.title, hint.lastmod, "capture"), {}, None, market)
            elif not have.title and hint.title:
                by_key[key] = IndexEntry(
                    have.url, hint.title, have.lastmod, have.source, have.locale, have.market_match
                )
        entries, other = _in_market(list(by_key.values()), market)
        index = cls(entries, tuple(sitemaps), dict(statuses or {}), edition, market, other)
        index._words = {e.url: e.words() for e in index.entries}
        return index

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {"sitemap": 0, "capture": 0}
        for entry in self.entries:
            out[entry.source] += 1
        return out

    def search(self, query: str | None, *, skip: frozenset[str] = frozenset(), limit: int = 24) -> list[Hit]:
        """Entries matching the query's words, the most words first, then the newest; with no query, the newest.
        `skip` holds url_keys left out (pages that answered 404 this session)."""
        live = [e for e in self.entries if url_key(e.url) not in skip]
        wanted = sorted(set(stems(query or "")))
        if not wanted:
            ranked = sorted(live, key=lambda e: (_newest(e.lastmod), e.url))
            return [Hit(e, ()) for e in ranked[:limit]]
        hits: list[tuple[int, str, str, Hit]] = []
        for entry in live:
            have = self._words.get(entry.url) or entry.words()
            matched = tuple(w for w in wanted if w in have)
            if matched:
                hits.append((-len(matched), _newest(entry.lastmod), entry.url, Hit(entry, matched)))
        hits.sort(key=lambda h: (h[0], h[1], h[2]))
        return [h[3] for h in hits[:limit]]


def locale_of_tag(tag: str) -> Locale | None:
    """A hreflang language tag ("en-US", "zh-Hant-TW", "en") as a Locale; None for x-default or nothing."""
    parts = [p for p in re.split(r"[-_]", tag.strip().lower()) if p]
    if not parts or parts[0] == "x" or not parts[0].isalpha() or len(parts[0]) not in (2, 3):
        return None
    region = parts[-1] if len(parts) > 1 and len(parts[-1]) == 2 and parts[-1].isalpha() else None
    return Locale("-".join(parts), language=parts[0], region=region)


def locale_of_path(url: str) -> Locale | None:
    """The locale an address's first path segments name: /en-us/, /th_th/, /us/en/ (a region, then a language),
    or a lone /us/ or /en/; None when the path starts with anything else."""
    segments = [unquote(p).lower() for p in urlsplit(url).path.split("/") if p][:2]
    if not segments:
        return None
    tag = _TAG.match(segments[0])
    if tag is not None:
        return Locale(segments[0].replace("_", "-"), language=tag.group(1), region=tag.group(2))
    first = segments[0]
    if len(first) != 2 or first not in LANGUAGES | REGIONS:
        return None
    second = segments[1] if len(segments) > 1 else ""
    if first in REGIONS and len(second) == 2 and second in LANGUAGES:
        return Locale(f"{first}/{second}", language=second, region=first)
    return Locale(first, either=first)


def locale_of_file(url: str) -> Locale | None:
    """The locale a sitemap file's address names: a folder (/en-us/sitemap.xml, /us/sitemap.xml) or its name
    (sitemap_en-us.xml, sitemap-us.xml); None when it names none."""
    path = urlsplit(url).path.lower()
    for segment in [p for p in path.split("/") if p][:-1]:
        found = locale_of_path(f"/{segment}/")
        if found is not None:
            return found
    tokens = _TOKEN.findall(path.rsplit("/", 1)[-1].rsplit(".", 1)[0])
    for left, right in itertools.pairwise(tokens):
        if len(left) == 2 and len(right) == 2 and left in LANGUAGES and right in REGIONS:
            return Locale(f"{left}-{right}", language=left, region=right)
    for token in tokens:
        if len(token) == 2 and token in LANGUAGES | REGIONS:
            return Locale(token, either=token)
    return None


def stems(text: str) -> list[str]:
    """A text's words, folded (case and accents), stop words out, each reduced to a plain singular."""
    folded = unicodedata.normalize("NFKD", text.lower()).encode("ascii", "ignore").decode("ascii")
    return [_stem(w) for w in _WORD.findall(folded) if w not in STOP_WORDS]


def slug_text(url: str) -> str:
    """The words of a URL's last two path segments ("/rings/aquamarine-well-ring.html" -> "rings aquamarine ...")."""
    parts = [unquote(p) for p in urlsplit(url).path.split("/") if p]
    return " ".join(re.sub(r"\.html?$", "", p) for p in parts[-2:]).replace("-", " ").replace("_", " ")


def url_key(url: str) -> str:
    """An address as the session knows a page: host without `www.`, path without a trailing slash, and only the query
    parameters that name the page (`identity_query`, D58), so tracking and variant parameters never split a page in
    two and a store that names its products by a parameter (`product.do?pid=`) never merges them into one."""
    parts = urlsplit(url)
    host = (parts.hostname or "").lower().removeprefix("www.")
    identity = identity_query(parts.query)
    return f"{host}{parts.path.rstrip('/')}".lower() + (f"?{identity}" if identity else "")


def identity_query(query: str) -> str:
    """The parameters of a query string that name a page rather than decorate it (IDENTITY_PARAMS), sorted by name
    and kept as written; empty when there are none. `?variant=`, `?cgid=`, `utm_*` and the like are left out."""
    kept = []
    for pair in query.split("&"):
        name, _, value = pair.partition("=")
        if value and unquote(name).lower() in IDENTITY_PARAMS:
            kept.append((unquote(name).lower(), value))
    return "&".join(f"{name}={value}" for name, value in sorted(kept))


def _localised(entry: IndexEntry, alternates: Mapping[str, str], file: str | None, market: str | None) -> IndexEntry:
    """The entry with its locale, or its alternate in the market when the market has a better one."""
    own_tag = next((t for t, href in alternates.items() if url_key(href) == url_key(entry.url)), None)
    locale = (locale_of_tag(own_tag) if own_tag else None) or locale_of_path(entry.url)
    if locale is None and file:
        locale = locale_of_file(file)
    if market is None:
        return _with(entry, entry.url, locale, None)
    own = locale.score(market) if locale is not None else 0
    best: tuple[int, str, Locale] | None = None
    for tag, href in sorted(alternates.items()):
        found = locale_of_tag(tag)
        score = found.score(market) if found is not None else 0
        if found is not None and score > max(own, 0) and (best is None or score > best[0]):
            best = (score, href, found)
    if best is not None:
        return _with(entry, best[1], best[2], True)
    return _with(entry, entry.url, locale, None if locale is None else own > 0)


def _with(entry: IndexEntry, url: str, locale: Locale | None, match: bool | None) -> IndexEntry:
    return IndexEntry(url, entry.title, entry.lastmod, entry.source, locale.text if locale else None, match)


def _in_market(entries: list[IndexEntry], market: str | None) -> tuple[list[IndexEntry], int]:
    """The entries a shopper in `market` sees: when any is in the market's locale, the market's best (a region
    before a language alone) and those whose locale is unknown; else all. And how many were left out."""
    if market is None or not any(e.market_match for e in entries):
        return entries, 0
    scores = {e.url: _score(e, market) for e in entries}
    best = max(scores.values())
    kept = [e for e in entries if scores[e.url] == best or (e.locale is None)]
    return kept, len(entries) - len(kept)


def _score(entry: IndexEntry, market: str) -> int:
    if entry.locale is None:
        return 0
    locale = locale_of_tag(entry.locale) if "-" in entry.locale else locale_of_path(f"/{entry.locale}/")
    return locale.score(market) if locale is not None else 0


def _stem(word: str) -> str:
    if len(word) > 4 and word.endswith("ies"):
        return word[:-3] + "y"
    if len(word) > 4 and word.endswith(("ches", "shes", "sses", "xes")):
        return word[:-2]
    if len(word) > 3 and word.endswith("s") and not word.endswith("ss"):
        return word[:-1]
    return word


def _newest(lastmod: str) -> str:
    """A sort key that puts the newest lastmod first (ISO dates sort as text) and entries without one last."""
    return "".join(chr(0x10FFFF - ord(c)) for c in lastmod) if lastmod else "\U0010ffff"
