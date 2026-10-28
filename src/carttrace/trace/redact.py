"""Redaction at write time: one table of named rules that every trace record passes through before it is staged.

Why at write time, and why one table. A trace is copied into judge runs, training exports, Slack posts and datasets,
and no copy can be cleaned after the fact, so nothing sensitive may reach the first byte on disk. CLAUDE.md names what
never does: cookies, auth headers, the secret `?key=` part of Shopify cart ids, signatures and personal names. The lab
learned the rest when a credential leaked through a transcript. Every rule lives in `REDACTION_RULES` with its
reason, so "what do we strip?" has one answer, and a new rule is one entry.

What the rules promise:
- Deterministic: the same record always becomes the same bytes, so replays and re-exports are byte-identical.
- Joinable: emails, phone numbers and reviewer names become placeholders derived from sha256 of the value, so the
  address in a tool result still matches the one in the next call. A cart id keeps its token and loses only its key.
- Idempotent: a redacted record passes through unchanged, and no rule fires on it again.
- Narrow where the data is evidence: a payment token (`tok_visa_3ds_challenge`), a cart token, a metafield `key`, an
  `Idempotency-Key` header and an SK-II product handle are all read by the judge, so a field named "token" or "key"
  is never redacted for its name alone, and credential shapes are matched tightly.
"""

from __future__ import annotations

import functools
import math
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from typing import Any, Final

from carttrace.core.canon import dumps, loads
from carttrace.core.cards import is_identifier_key, looks_like_card_number
from carttrace.core.errors import ValidationError
from carttrace.core.ids import stable_id
from carttrace.trace.schema import WireRecord

REDACTED = "redacted"
"""What a secret becomes: plain letters, so it stays valid inside a URL, a header value and a cookie."""

PLACEHOLDER_DOMAIN = "redacted.invalid"
"""Emails become `email-<sha>@redacted.invalid`: still shaped like an email, never deliverable (RFC 2606)."""

SECRET_QUERY_PARAMS: tuple[str, ...] = (
    "key",
    "token",
    "access_token",
    "signature",
    "sig",
    "api_key",
    "apikey",
    "client_secret",
    "refresh_token",
    "id_token",
    "password",
    "x-amz-signature",
    "x-amz-security-token",
    "x-amz-credential",
)
"""Query parameters whose value is always a secret, matched case-insensitively."""


# Replacements: (secret, salt) -> what replaces it. A salt keys the stable placeholders; see TraceWriter.open.


def _fixed(secret: str, salt: str) -> str:
    return REDACTED


def _email_placeholder(secret: str, salt: str) -> str:
    return f"{stable_id('email', salt, secret.replace('%40', '@').lower())}@{PLACEHOLDER_DOMAIN}"


def _phone_placeholder(secret: str, salt: str) -> str:
    digits = re.sub(r"[^0-9]", "", secret)
    if len(digits) == 10 and not secret.lstrip().startswith("+"):
        digits = f"1{digits}"  # a North American number written without its country code joins its +1 form
    return stable_id("phone", salt, digits)


def _name_placeholder(secret: str, salt: str) -> str:
    return stable_id("reviewer", salt, " ".join(secret.split()).casefold())


_PLACEHOLDER = re.compile(
    rf"{REDACTED}|(?:email|phone|reviewer|jws|jwp|jwd)-[0-9a-f]{{12}}(?:@{re.escape(PLACEHOLDER_DOMAIN)})?"
)
"""Everything a rule writes. A value that already is one is left alone, which is what makes redaction idempotent."""


# Last checks on a candidate, for what a pattern cannot pin down.

_FILE_SUFFIX = re.compile(
    r"png|jpe?g|gif|webp|svg|avif|bmp|ico|tiff?|heic|mp4|webm|mov|m?js|css|map|json|html?|pdf|woff2?|ttf"
)


def _is_email(candidate: str) -> bool:
    """Shopify CDN names retina images `logo@2x.png`; a "TLD" that is a file suffix means a file, not an address."""
    return not _FILE_SUFFIX.fullmatch(candidate.rpartition(".")[2].lower())


_PHONE_SHAPE = re.compile(r"\+?[0-9\s().-]+")


def _is_phone(candidate: str) -> bool:
    """Only phone characters, and as many digits as a real number has. Keeps "optional" or "true" under a phone key."""
    digits = sum(ch in "0123456789" for ch in candidate)
    return bool(_PHONE_SHAPE.fullmatch(candidate.strip())) and 7 <= digits <= 15


# Patterns. Every text pattern names the part that goes as the group `secret`.

_URL_CREDENTIALS = re.compile(r"(?<![A-Za-z0-9+.-])[A-Za-z][A-Za-z0-9+.-]*://(?P<secret>[^\s/?#@\"'<>\\]+)@")
"""The lookbehind starts a scheme only where a run of scheme characters starts. Without it every letter of a long
base64 image in a body starts a match that backtracks to the end of the run, which is quadratic: minutes per MB."""

_CART_KEY = re.compile(
    r"(?:gid(?::|%3A)(?://|%2F%2F)shopify(?:/|%2F)Cart(?:/|%2F)|/cart/c/)[A-Za-z0-9_-]+"
    r"(?:\?|%3F)key(?:=|%3D)"
    r"(?P<secret>(?:[A-Za-z0-9_~-]|%(?!2[36])[0-9A-Fa-f]{2}|\.(?=[A-Za-z0-9_~-]))+)",
    re.IGNORECASE,
)
"""gid://shopify/Cart/<token>?key=<secret> and /cart/c/<token>?key=<secret>, plain or percent-encoded. Seen on live
stores: 24 or 26 character tokens, keys of 32 hex digits or 12 letters and digits. A dot ends the key only when
nothing follows it, so prose like "...?key=abc." keeps its full stop."""

_SECRET_QUERY = re.compile(
    r"(?:(?<=[?&;\s])|^|%3F|%26)"
    r"(?:" + "|".join(re.escape(p) for p in sorted(SECRET_QUERY_PARAMS, key=len, reverse=True)) + r")"
    r"(?:=|%3D)(?P<secret>(?:(?!%2[36])[^&#\s\"'<>\\])*(?:(?!%2[36])[^&#\s\"'<>\\.,;:!?)\]]))",
    re.IGNORECASE,
)
"""Values run to the next delimiter, so a base64 signature keeps no tail, but never end on sentence punctuation,
which base64 does not use: "...?token=abc." keeps its full stop. Also matches form bodies and `name=value` quoted
in error text."""

_CREDENTIAL_SHAPE = re.compile(
    r"(?P<secret>"
    r"\bsk-(?=[A-Za-z0-9_-]*[A-Z])[A-Za-z0-9_-]{20,}"  # OpenAI and Anthropic keys: mixed case, unlike sk-ii handles
    r"|\bgh[pousr]_[A-Za-z0-9]{30,}"  # GitHub tokens
    r"|\bhf_(?=[A-Za-z0-9]*[A-Z])[A-Za-z0-9]{30,}"  # Hugging Face tokens
    r"|\b(?:AKIA|ASIA)[0-9A-Z]{16}\b"  # AWS access key ids
    r"|\bxox[abprs]-[A-Za-z0-9-]{10,}"  # Slack tokens
    r"|(?<=hooks\.slack\.com/services/)[A-Za-z0-9/_-]{20,}"  # Slack webhook paths
    r"|\bAIza[0-9A-Za-z_-]{35}"  # Google API keys
    r"|\bshp(?:at|ca|pa|ss)_[0-9A-Fa-f]{32}"  # Shopify admin and app tokens
    r"|(?<=[Bb]earer )(?=[A-Za-z._~+/=-]*[0-9])[A-Za-z0-9._~+/=-]{15,}[A-Za-z0-9_~+/=-]"  # bearer tokens in text
    r")"
)
_CREDENTIAL_GATE = (
    "sk-",
    "ghp_",
    "gho_",
    "ghu_",
    "ghs_",
    "ghr_",
    "hf_",
    "AKIA",
    "ASIA",
    "xox",
    "hooks.slack.com/services/",
    "AIza",
    "shpat_",
    "shpca_",
    "shppa_",
    "shpss_",
    "earer ",
)
"""One literal from each alternative of _CREDENTIAL_SHAPE, which is case-sensitive."""

_JWS = re.compile(
    r"(?P<secret>\beyJ[A-Za-z0-9_-]{10,}\.(?:eyJ[A-Za-z0-9_-]{10,})?\."
    r"(?!jw[spd]-[0-9a-f]{12}(?![A-Za-z0-9_-]))[A-Za-z0-9_-]{10,}(?:~[A-Za-z0-9_-]*)*)"
)
"""A JWT or JWS: a JSON header, a JSON payload or none (a detached JWS, `<header>..<signature>`, UCP's AP2
merchant_authorization), a signature, and an SD-JWT's disclosures after `~`."""


def _jws_placeholder(secret: str, salt: str) -> str:
    """The JWS with its structure kept: the header as it was (alg, kid, typ: what was signed how), the payload, the
    signature and each SD-JWT disclosure as a stable placeholder, so a forwarded checkout_jwt joins the merchant's
    and a mandate's parts can be told apart, while no claim (an address, an email in base64) and no signature is
    kept."""
    token, *disclosures = secret.split("~")
    header, payload, signature = token.split(".")
    parts = [header, stable_id("jwp", salt, payload) if payload else "", stable_id("jws", salt, signature)]
    tail = [stable_id("jwd", salt, d) if d else "" for d in disclosures]
    return "~".join([".".join(parts), *tail])


_EMAIL = re.compile(
    r"(?<![A-Za-z0-9._%+-])(?<!//)"
    r"(?P<secret>[A-Za-z0-9._%+-]+(?:@|%40)(?:[A-Za-z0-9](?:[A-Za-z0-9-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,24})"
    r"(?![A-Za-z0-9-]|\.[A-Za-z0-9]|/[0-9])"
)
"""`(?<!//)` keeps the host of a URL whose credentials were just replaced (https://redacted@host) from reading as an
address; `%40` catches addresses inside query strings. The last lookahead keeps a versioned id: a task template is
written `<template>@<version>` and the version may be a module path with its own number
(`compare.named.0@mc.agents.tasks/1`), which read as an address and was redacted in 18,505 episode headers of
2026-10-11 and 12. No address is followed by a slash and a digit, and a domain followed by a dot and a letter is not
cut short to get around that; an address followed by anything else is still redacted."""

_PHONE = re.compile(
    r"(?<![\w+/.-])"
    r"(?P<secret>"
    r"\+[0-9]{1,3}(?:[ -]?\(?[0-9]{1,4}\)?){2,5}"  # international, with its +: +44 20 7946 0958, +14155550100
    r"|(?:1[ .-])?(?:\([0-9]{3}\)[ .-]?|[0-9]{3}[ .-])[0-9]{3}[ .-][0-9]{4}"  # North American, with separators
    r")"
    r"(?![\w-])"
)
"""Deliberately narrow: a bare run of digits is a variant id, a GTIN or a timestamp far more often than a phone
number, so only a leading + or the separators of a written number count. Phone-named fields catch the rest."""

_CREDENTIAL_KEYS = re.compile(
    r"secret|password|passwd|passphrase|(?:access|refresh|bearer|session|private|auth|csrf|xsrf)token|^idtoken$|apikey"
)
_CARD = re.compile(r"(?<![\w/.=&?#%-])(?P<secret>\d(?:[ -]?\d){12,18})(?![\w/.&#%-])")
"""A digit run of card length, single spaces or dashes allowed between digits, not inside a word, a path, a query
value or a decimal number, so a gid:// id or a product URL's ?variant= keeps its digits."""
_PHONE_KEYS = re.compile(r"phone|mobile|^tel$|^fax")
_TOKEN_HEADERS = re.compile(r"token|secret|passw(?:or)?d|api-?key|csrf|xsrf")
_AUTH_SCHEME = re.compile(r"[A-Za-z][A-Za-z0-9!#$%&'*+.^_`|~-]*")


@dataclass(frozen=True, slots=True)
class RedactionRule:
    """One kind of sensitive value, and every place in a record where it can appear."""

    name: str
    """What `WireRecord.redactions` records when this rule changes a record."""
    reason: str
    headers: frozenset[str] = frozenset()
    """Lower-case header names whose value is replaced. The same names count as JSON keys inside bodies."""
    header_names: re.Pattern[str] | None = None
    """More header names, by pattern. Headers only: as a JSON key, "token" is usually a cart or payment token."""
    keys: re.Pattern[str] | None = None
    """JSON keys (lower case, separators removed) whose values are replaced wherever they appear."""
    text: re.Pattern[str] | None = None
    """Substrings to replace inside any string; the pattern's `secret` group is the part that goes."""
    gate: tuple[str, ...] = ()
    """Substrings of which every match of `text` contains at least one, so a string with none of them skips the
    pattern. Only a shortcut: a gate must never be narrower than its pattern. Empty runs the pattern on every string."""
    in_urls: bool = True
    """Whether `text` applies to URLs and header values too, not only to bodies and error text."""
    accept: Callable[[str], bool] | None = None
    """A last check on a candidate, for shapes a pattern cannot pin down."""
    replace: Callable[[str, str], str] = _fixed
    """(secret, salt) to its replacement: the fixed marker, or a stable placeholder."""
    keep_scheme: bool = False
    """Keep an auth scheme ("Bearer", "Payment") and replace only the credentials after it."""
    reviews: bool = False
    """This rule replaces reviewer and author names inside review objects."""
    skip_ids: bool = False
    """This rule leaves strings under identifier keys alone (`carttrace.core.cards.is_identifier_key`): an id can be any
    digits."""


REDACTION_RULES: tuple[RedactionRule, ...] = (
    RedactionRule(
        "cookie",
        "a session cookie lets whoever reads the trace replay the session",
        headers=frozenset({"cookie"}),
    ),
    RedactionRule(
        "set_cookie",
        "the session the store just issued, for the same reason",
        headers=frozenset({"set-cookie", "set-cookie2"}),
    ),
    RedactionRule(
        "authorization",
        "credentials; the scheme is kept because it says which trust tier the agent used",
        headers=frozenset({"authorization"}),
        keep_scheme=True,
    ),
    RedactionRule(
        "proxy_authorization",
        "credentials for the proxy of a network-position experiment",
        headers=frozenset({"proxy-authorization"}),
        keep_scheme=True,
    ),
    RedactionRule(
        "signature",
        "an RFC 9421 request signature can be replayed inside its validity window",
        headers=frozenset({"signature"}),
    ),
    RedactionRule(
        "signature_input",
        "the RFC 9421 parameters go with the signature they describe",
        headers=frozenset({"signature-input"}),
    ),
    RedactionRule(
        "token_header",
        "x-shopify-storefront-access-token and every other header that carries a token, key or secret",
        header_names=_TOKEN_HEADERS,
    ),
    RedactionRule(
        "credential_field",
        "a JSON field whose name says it holds a credential: access_token, client_secret, password, api_key",
        keys=_CREDENTIAL_KEYS,
    ),
    RedactionRule(
        "url_credentials",
        "user:password@ in a URL, such as a proxy URL quoted in an error",
        text=_URL_CREDENTIALS,
        gate=("@",),
    ),
    RedactionRule(
        "cart_key",
        "the ?key= secret of a Shopify cart id or cart URL grants the cart; the token is kept so carts still join",
        text=_CART_KEY,
        gate=("=", "%3D", "%3d"),
    ),
    RedactionRule(
        "secret_query",
        "query parameters named key, token, access_token, signature and their kin",
        text=_SECRET_QUERY,
        gate=("=", "%3D", "%3d"),
    ),
    RedactionRule(
        "jws",
        "a JWT or JWS (an AP2 mandate, a merchant's signed checkout) can be a bearer credential; a placeholder from "
        "sha256 replaces it whole, so a forwarded checkout_jwt still joins the one the merchant signed (the v1 "
        "review's #48: a fixed marker made every mandate look alike)",
        text=_JWS,
        gate=("eyJ",),
        replace=_jws_placeholder,
    ),
    RedactionRule(
        "credential_shape",
        "values shaped like well-known credentials (API keys, JWTs, bearer tokens, Slack webhooks), wherever they are",
        text=_CREDENTIAL_SHAPE,
        gate=_CREDENTIAL_GATE,
    ),
    RedactionRule(
        "card_number",
        "no card number anywhere, test cards included: 13 to 19 digits passing Luhn, except ids and gid:// paths",
        text=_CARD,
        in_urls=False,
        accept=looks_like_card_number,
        skip_ids=True,
    ),
    RedactionRule(
        "email",
        "personal data; a placeholder from sha256 keeps joins",
        text=_EMAIL,
        gate=("@", "%40"),
        accept=_is_email,
        replace=_email_placeholder,
    ),
    RedactionRule(
        "phone",
        "personal data; a placeholder from sha256 keeps joins",
        keys=_PHONE_KEYS,
        text=_PHONE,
        in_urls=False,
        accept=_is_phone,
        replace=_phone_placeholder,
    ),
    RedactionRule(
        "review_name",
        "reviewers' and authors' names are personal data, and no judgement needs them",
        reviews=True,
        replace=_name_placeholder,
    ),
)
"""Applied in this order; `redactions` lists rule names in this order too."""


_NOT_ALNUM = re.compile(r"[^a-z0-9]")
_CAMEL_HUMP = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_WORD = re.compile(r"[a-z0-9]+")

# Key and type names repeat across thousands of objects in one catalog reply, so their checks are cached.


@functools.lru_cache(maxsize=8192)
def _norm(key: str) -> str:
    """A JSON key or header name with case and separators gone: displayName, display_name -> displayname."""
    return _NOT_ALNUM.sub("", key.lower())


@functools.lru_cache(maxsize=8192)
def _says_review(text: str) -> bool:
    """Whether a key or type names reviews: "reviews", "productReviews", "ProductReview", but never "previews"."""
    return not {"review", "reviews"}.isdisjoint(_WORD.findall(_CAMEL_HUMP.sub(" ", text).lower()))


_CARD_RULE: Final = next(rule for rule in REDACTION_RULES if rule.name == "card_number")
_QUERY_DIGITS: Final = re.compile(r"(?P<lead>[?&;](?P<name>[A-Za-z_][\w.-]*)=)(?P<secret>\d{13,19})(?=[&;#]|$)")
"""A query parameter whose value is 13 to 19 digits, for the card rule in URLs."""
_CARD_PARAM: Final = re.compile(r"(?i)card|pan|cc|number|acct|account")
"""Query parameter names a card number goes under."""
_TEXT_RULES: tuple[tuple[RedactionRule, re.Pattern[str], re.Pattern[str] | None], ...] = tuple(
    (rule, rule.text, re.compile("|".join(map(re.escape, rule.gate))) if rule.gate else None)
    for rule in REDACTION_RULES
    if rule.text is not None
)
"""Each text rule with its pattern and its gate, compiled once: one C-level search decides whether to run it."""
_KEY_RULES: tuple[tuple[RedactionRule, frozenset[str]], ...] = tuple(
    (rule, frozenset(_norm(h) for h in rule.headers)) for rule in REDACTION_RULES if rule.headers or rule.keys
)
_REVIEW_RULE = next(rule for rule in REDACTION_RULES if rule.reviews)

# Review objects: what marks one, and which of its fields hold a person's name.
_TYPE_KEYS = frozenset({"type", "typename", "kind", "objecttype", "resourcetype"})
_REVIEW_SHAPE = frozenset(
    [
        "rating",
        "stars",
        "score",
        "ratingvalue",
        "reviewrating",
        "starrating",
        "reviewbody",
        "reviewtext",
        "body",
        "comment",
    ]
)
_AUTHOR_HOLDERS = frozenset(
    ["author", "reviewer", "user", "customer", "buyer", "writer", "postedby", "submittedby", "createdby"]
)
_PERSON_NAMES = frozenset(
    [
        "name",
        "displayname",
        "nickname",
        "username",
        "fullname",
        "firstname",
        "lastname",
        "givenname",
        "familyname",
        "screenname",
        "publicname",
    ]
)
_AUTHOR_NAMES = (_PERSON_NAMES - {"name"}) | frozenset(
    ["authorname", "reviewername", "reviewernickname", "authordisplayname", "reviewerdisplayname", "customername"]
)
_REVIEW_SIGNS = _REVIEW_SHAPE | _AUTHOR_HOLDERS | _AUTHOR_NAMES
_WRAPPERS = frozenset(
    [
        "edges",
        "node",
        "nodes",
        "items",
        "results",
        "data",
        "entries",
        "list",
        "records",
        "hits",
        "values",
        "elements",
        "page",
        "content",
    ]
)
"""Pagination wrappers (GraphQL edges and nodes, REST pages) between a reviews key and the reviews themselves."""


def _header_rule(name: str) -> RedactionRule | None:
    lowered = name.lower()
    for rule in REDACTION_RULES:
        if lowered in rule.headers or (rule.header_names is not None and rule.header_names.search(lowered)):
            return rule
    return None


@functools.lru_cache(maxsize=8192)
def _key_rule(norm: str) -> RedactionRule | None:
    for rule, header_keys in _KEY_RULES:
        if norm in header_keys or (rule.keys is not None and rule.keys.search(norm)):
            return rule
    return None


def _takes(rule: RedactionRule, item: Any) -> bool:
    """Whether a key rule replaces this value: a scalar it accepts, or a non-empty container under a key that has no
    shape check (a credential key)."""
    if item is None or isinstance(item, bool):
        return False
    if isinstance(item, float):
        # A fractional number is a price, a weight or a probability, never a phone number: under a decision's
        # `probabilities` the label "phone" holds a float that the phone shape check took for digits (157 of 600
        # shopper.field lines on 2026-10-13 stopped parsing as Decisions).
        return rule.accept is None
    if isinstance(item, str | int):
        return rule.accept is None or rule.accept(str(item))
    return rule.accept is None and bool(item)


def _is_review(obj: Mapping[Any, Any], names: set[str], *, in_container: bool) -> bool:
    """A review: under a key that names reviews, or typed as one, and shaped like one (a rating, a body, an author).
    The shape check keeps a checkout step typed "review_step" from losing its name."""
    if names.isdisjoint(_REVIEW_SIGNS):
        return False
    return in_container or any(
        _norm(str(k)) in _TYPE_KEYS and isinstance(v, str) and _says_review(v) for k, v in obj.items()
    )


def _embedded_json(value: str) -> dict[str, Any] | list[Any] | None:
    """The object or array a string carries, if it carries one. MCP tool results put their JSON in a text block,
    where a review's author is invisible to a pattern."""
    text = value.strip()
    if len(text) < 2 or text[0] not in "{[" or text[-1] not in "}]":
        return None
    try:
        parsed = loads(text)
    except (ValueError, RecursionError):
        return None
    return parsed if isinstance(parsed, dict | list) else None


class _Pass:
    """One redaction pass over one record: applies the table and remembers which rules fired."""

    __slots__ = ("_clean_keys", "_subs", "changes", "fired", "salt")

    def __init__(self, salt: str) -> None:
        self.salt = salt
        self.fired: set[str] = set()
        self.changes = 0
        self._subs = tuple((rule, pattern, gate, self._substitute(rule)) for rule, pattern, gate in _TEXT_RULES)
        self._clean_keys: set[str] = set()
        """JSON keys already seen to need no change; a catalog reply repeats the same few hundred thousands of times."""

    def rules_fired(self) -> list[str]:
        return [rule.name for rule in REDACTION_RULES if rule.name in self.fired]

    def hit(self, rule: RedactionRule) -> None:
        self.fired.add(rule.name)
        self.changes += 1

    def swap(self, rule: RedactionRule, secret: str) -> str | None:
        """The replacement for `secret`, or None when it is not one after all: blank, already a placeholder, or
        refused by the rule's own check."""
        if (
            not secret.strip()
            or _PLACEHOLDER.fullmatch(secret)
            or (rule.accept is not None and not rule.accept(secret))
        ):
            return None
        self.hit(rule)
        return rule.replace(secret, self.salt)

    def whole(self, rule: RedactionRule, value: str) -> str:
        """All of `value` replaced, or everything after its auth scheme for a rule that keeps the scheme."""
        prefix, secret = "", value
        if rule.keep_scheme:
            scheme, space, rest = value.strip().partition(" ")
            if space and rest.strip() and _AUTH_SCHEME.fullmatch(scheme):
                prefix, secret = f"{scheme} ", rest.strip()
        replacement = self.swap(rule, secret)
        return value if replacement is None else prefix + replacement

    def text(self, value: str, *, url: bool, ids: bool = False) -> str:
        """Every text rule over one string; in a URL or a header value, only the rules that belong there; under an
        identifier key, not the rules that skip ids."""
        for rule, pattern, gate, substitute in self._subs:
            if (
                (url and not rule.in_urls)
                or (ids and rule.skip_ids)
                or (gate is not None and gate.search(value) is None)
            ):
                continue
            value = pattern.sub(substitute, value)
        if url and "=" in value:
            value = _QUERY_DIGITS.sub(self._query_card, value)
        return value

    def _query_card(self, match: re.Match[str]) -> str:
        """A card number as a URL query value (`?card=4242...`) under a name that says card, PAN or account number:
        the card rule skips URLs, whose paths and parameters (Shopify's `variant=`) hold numeric ids, so these
        values are read here (the v1 review's #14)."""
        name = match.group("name")
        if is_identifier_key(name) or not _CARD_PARAM.search(name):
            return match.group(0)
        replacement = self.swap(_CARD_RULE, match.group("secret"))
        return match.group(0) if replacement is None else f"{match.group('lead')}{replacement}"

    def _substitute(self, rule: RedactionRule) -> Callable[[re.Match[str]], str]:
        def substitute(match: re.Match[str]) -> str:
            replacement = self.swap(rule, match.group("secret") or "")
            if replacement is None:
                return match.group(0)
            start, end = match.span("secret")
            whole, offset = match.group(0), match.start()
            return whole[: start - offset] + replacement + whole[end - offset :]

        return substitute

    def headers(self, headers: Mapping[str, str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for name, value in headers.items():
            rule = _header_rule(name)
            out[name] = self.text(value, url=True) if rule is None else self.whole(rule, value)
        return out

    def json(self, value: Any, *, reviews: bool = False, ids: bool = False) -> Any:
        if isinstance(value, float) and not math.isfinite(value):
            # Refused here, as carttrace.core.canon refuses it, because a pydantic JSON dump would quietly write null.
            raise ValidationError(
                f"cannot record {value!r}: NaN and Infinity are not JSON values",
                hint="record null, or a string, for a number that is not finite",
            )
        if isinstance(value, int) and not isinstance(value, bool) and not ids and _card_length_number(value):
            # a card number written as a JSON number is still a card number (the v1 review's #14)
            text = str(value)
            redacted = self.text(text, url=False)
            return redacted if redacted != text else value
        if value is None or isinstance(value, bool | int | float):
            return value
        if isinstance(value, str):
            return self.string(value, ids=ids)
        if isinstance(value, list | tuple):
            return [self.json(item, reviews=reviews, ids=ids) for item in value]
        if isinstance(value, Mapping):
            return self.mapping(value, reviews=reviews)
        raise ValidationError(
            f"cannot record a {type(value).__name__}: trace records hold JSON values only",
            hint="decode bytes, and turn sets, dataclasses and models into JSON before recording them",
        )

    def string(self, value: str, *, ids: bool = False) -> str:
        """Text, or JSON carried inside a string. Embedded JSON is rewritten only when a rule changed something in
        it, so a string with nothing to redact keeps its exact bytes."""
        embedded = _embedded_json(value)
        if embedded is None:
            return self.text(value, url=False, ids=ids)
        before = self.changes
        redacted = self.json(embedded)
        return value if self.changes == before else dumps(redacted)

    def mapping(self, obj: Mapping[Any, Any], *, reviews: bool, person: bool = False) -> dict[str, Any]:
        """One JSON object. `reviews` says it sits under a key that names reviews; `person` says it is a review's
        author, whose name fields go."""
        names = {_norm(str(key)) for key in obj}
        review = _is_review(obj, names, in_container=reviews)
        # A bare "name" is the author only when nothing else is: in schema.org it is the review's headline.
        bare_name = review and names.isdisjoint(_AUTHOR_HOLDERS) and not names.isdisjoint(_REVIEW_SHAPE)
        out: dict[str, Any] = {}
        for raw_key, item in obj.items():
            key = str(raw_key)
            norm = _norm(key)
            rule = _key_rule(norm)
            if rule is not None and _takes(rule, item):
                value = self.keyed(rule, item)
            elif (person and norm in _PERSON_NAMES) or (
                review and (norm in _AUTHOR_NAMES or (norm == "name" and bare_name))
            ):
                value = self.names(item)
            elif review and norm in _AUTHOR_HOLDERS:
                value = self.holder(item)
            else:
                value = self.json(
                    item,
                    reviews=_says_review(key) or (reviews and not review and norm in _WRAPPERS),
                    ids=is_identifier_key(key),
                )
            out[self.key(key)] = value
        return out

    def key(self, key: str) -> str:
        """A JSON key with the text rules applied: an object keyed by email address is still personal data."""
        if key in self._clean_keys:
            return key
        redacted = self.text(key, url=False)
        if redacted == key:
            self._clean_keys.add(key)
        return redacted

    def keyed(self, rule: RedactionRule, item: Any) -> Any:
        """A value under a key that names a secret: a scalar is replaced, a container under a credential key goes
        whole, because its inner fields are named for what they are, not for being secret."""
        if isinstance(item, str | int | float):
            replaced = self.whole(rule, str(item))
            return item if replaced == str(item) else replaced
        self.hit(rule)
        return REDACTED

    def names(self, item: Any) -> Any:
        if isinstance(item, str):
            return self.whole(_REVIEW_RULE, item)
        if isinstance(item, list | tuple):
            return [self.names(x) for x in item]
        return self.json(item)

    def holder(self, item: Any) -> Any:
        """A review's author, reviewer or user: a name, or a person object whose name fields go."""
        if isinstance(item, Mapping):
            return self.mapping(item, reviews=False, person=True)
        if isinstance(item, list | tuple):
            return [self.holder(x) for x in item]
        return self.names(item)


def _card_length_number(value: int) -> bool:
    """A JSON integer the card rule reads: 13 to 19 digits, except a millisecond timestamp.

    A 13-digit number starting with 1 is epoch milliseconds between 2001 and 2286 (the sandbox clock, an event's
    `at`), and no card network issues a 13-digit number starting with 1; about one in ten such timestamps passes Luhn,
    so the pilot release lost a tenth of its event times to the card rule. The release scrub's `_card_like` draws the
    same line for a bare digit run. Every other card length and prefix is still read."""
    digits = str(abs(value))
    if len(digits) == MS_TIMESTAMP_DIGITS and digits[0] == "1":
        return False
    return 13 <= len(digits) <= 19


MS_TIMESTAMP_DIGITS: Final = 13


def _merged(earlier: Iterable[str], fired: Iterable[str]) -> list[str]:
    names = set(earlier) | set(fired)
    known = [rule.name for rule in REDACTION_RULES if rule.name in names]
    return known + sorted(names - set(known))


def redact_json(value: Any, *, salt: str = "") -> tuple[Any, list[str]]:
    """A JSON value with every rule applied, and the names of the rules that changed it, in table order.

    Raises ValidationError for anything that is not JSON (bytes, sets, objects), because a value that cannot be
    redacted must not be recorded.
    """
    run = _Pass(salt)
    return run.json(value), run.rules_fired()


def redact_text(text: str, *, url: bool = False, salt: str = "") -> tuple[str, list[str]]:
    """Free text (an error message, a Slack line) or, with `url=True`, a URL, where phone numbers never appear."""
    run = _Pass(salt)
    return (run.text(text, url=True) if url else run.string(text)), run.rules_fired()


def redact_wire(record: WireRecord, *, salt: str = "") -> tuple[WireRecord, list[str]]:
    """A wire record with its headers, URL, bodies and error redacted, and the rules that fired.

    The returned record's `redactions` holds those rules merged with any it already listed.
    """
    run = _Pass(salt)
    update: dict[str, Any] = {
        "url": run.text(record.url, url=True),
        "request_headers": run.headers(record.request_headers),
        "request_body": run.json(record.request_body),
        "response_headers": run.headers(record.response_headers),
        "response_body": run.json(record.response_body),
        "error": None if record.error is None else run.string(record.error),
    }
    fired = run.rules_fired()
    update["redactions"] = _merged(record.redactions, fired)
    return record.model_copy(update=update), fired
