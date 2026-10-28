"""Merchant groups: which queue rows belong to one business, so they share a split and never leak across it.

Why. Stores owned by one company share templates, platforms, catalog conventions and often product data, so a model
trained on one has partly seen the others. The sampling plan therefore splits by merchant before any task is
generated (train 2,800 / validation 400 / locked test 800 groups). A group is keyed by the normalised parent company,
or by the registrable domain when no parent was found, and rows that share either a parent or a domain are joined:
the two Acme rows (one says "Acme Group", the other names no parent) are one group, and so are "Hooli Beauté" and
"Hooli Beaute". The key is the group's identity, `merchant_group_id = stable_id("mg", key)`, so it is the same on every
machine and every pull.

Normalising a parent name: accents and case are folded, apostrophes dropped ("D'Évora" -> "devora"), "&" read as
"and", notes in parentheses dropped ("Hooli Health Tech (shared with Store 327)"), a leading "The" and trailing legal
forms removed ("The Acme Companies" -> "acme"). "Brands" is not a legal form, so "Globex Brands" and
"Globex Company" stay apart. Variants no rule can see ("Store 154 (Initech)" is Initech) live in
`data/stores/merchant-aliases.yaml`, where anyone can add one without touching code.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from carttrace.core.errors import ValidationError
from carttrace.core.ids import stable_id
from carttrace.stores.model import MerchantBasis
from carttrace.stores.text import collapse_ws, fold

ALIASES_SCHEMA = "mc.stores.merchant-aliases.v1"

_NO_PARENT = frozenset({"", "-", "none", "none found", "n/a", "na", "unknown", "not established", "not applicable"})
_NO_PARENT_PREFIXES = ("no shared group", "no shared parent", "not established", "none found")
_LEGAL_FORMS = frozenset(
    {
        "inc",
        "incorporated",
        "llc",
        "llp",
        "lp",
        "ltd",
        "limited",
        "plc",
        "corp",
        "corporation",
        "co",
        "company",
        "companies",
        "group",
        "holding",
        "holdings",
        "se",
        "sa",
        "spa",
        "ag",
        "ab",
        "gmbh",
        "nv",
        "bv",
        "oy",
        "asa",
        "kg",
        "srl",
        "sas",
        "pty",
        "pte",
    }
)
_GENERIC_SECOND_LEVEL = frozenset(
    {"com", "co", "net", "org", "ac", "gov", "edu", "or", "ne", "go", "ltd", "plc", "gen", "firm", "ind", "me", "id"}
)
_PARENTHETICAL = re.compile(r"\([^)]*\)?")
_APOSTROPHES = re.compile(r"['’‘`´]")
_NON_ALNUM = re.compile(r"[^a-z0-9]+")


@dataclass(frozen=True, slots=True)
class MerchantGroup:
    id: str
    key: str
    basis: MerchantBasis
    members: tuple[str, ...]
    """Store ids in the group, sorted."""


def parent_key(name: str | None, aliases: Mapping[str, str] | None = None) -> str | None:
    """The normalised parent company, or None when the cell says no parent was found."""
    if name is None:
        return None
    folded = collapse_ws(fold(name))
    if folded in _NO_PARENT or folded.startswith(_NO_PARENT_PREFIXES):
        return None
    key = _base_key(folded)
    return (aliases or {}).get(key, key)


def normalise_host(host: str) -> str:
    """ "https://www.Example.com/path" -> "example.com": lowercase host, no scheme, path, port or "www."."""
    text = host.strip().lower()
    text = text.split("://", 1)[-1].split("/", 1)[0].split("?", 1)[0].split(":", 1)[0].rstrip(".")
    return text.removeprefix("www.")


def registrable_domain(host: str) -> str:
    """The domain a business registers: "store-004.example" -> "store-004.example", "store-098.example" -> "store-098.example".

    A two-level public suffix is recognised as a generic second level ("co", "com", "org", ...) under a two-letter
    country code. That stands in for the Public Suffix List and covers every market in the queue; it errs towards
    keeping one more label, so two businesses under one country suffix are never merged.
    """
    labels = normalise_host(host).split(".")
    if len(labels) <= 2:
        return ".".join(labels)
    if len(labels[-1]) == 2 and labels[-2] in _GENERIC_SECOND_LEVEL:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def group_merchants(stores: Iterable[tuple[str, str | None, str]]) -> dict[str, MerchantGroup]:
    """Merchant group per store id, from (store id, parent key or None, registrable domain) for every store.

    Stores sharing a parent key or a registrable domain are one group. The group key is the smallest parent key in
    it, or the smallest domain when no member names a parent, so the result does not depend on row order.
    """
    entries = sorted(stores)
    roots: dict[str, str] = {}

    def find(node: str) -> str:
        roots.setdefault(node, node)
        while roots[node] != node:
            roots[node] = roots[roots[node]]
            node = roots[node]
        return node

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            roots[max(ra, rb)] = min(ra, rb)

    for store_id, parent, domain in entries:
        domain_node = f"d:{domain or f'store:{store_id}'}"
        find(domain_node)
        if parent:
            union(f"p:{parent}", domain_node)

    components: dict[str, list[tuple[str, str | None, str]]] = {}
    for entry in entries:
        store_id, _, domain = entry
        components.setdefault(find(f"d:{domain or f'store:{store_id}'}"), []).append(entry)

    out: dict[str, MerchantGroup] = {}
    for members in components.values():
        parents = sorted({p for _, p, _ in members if p})
        domains = sorted({d for _, _, d in members if d})
        basis: MerchantBasis = "parent" if parents else "domain"
        key = parents[0] if parents else domains[0] if domains else members[0][0]
        group = MerchantGroup(
            id=stable_id("mg", key), key=key, basis=basis, members=tuple(sorted(sid for sid, _, _ in members))
        )
        for store_id, _, _ in members:
            out[store_id] = group
    return out


def load_aliases(path: Path) -> dict[str, str]:
    """Variant key to canonical key from the aliases file; {} when the file does not exist."""
    if not path.is_file():
        return {}
    doc: Any = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict) or doc.get("schema") != ALIASES_SCHEMA or not isinstance(doc.get("aliases"), dict):
        raise ValidationError(f"{path} is not a {ALIASES_SCHEMA} file", hint="see the comment at the top of the file")
    out: dict[str, str] = {}
    for canonical, variants in doc["aliases"].items():
        if not isinstance(canonical, str) or not isinstance(variants, list):
            raise ValidationError(f"{path}: {canonical!r} must map a name to a list of variant names")
        target = _base_key(collapse_ws(fold(canonical)))
        for variant in variants:
            if not isinstance(variant, str):
                raise ValidationError(f"{path}: variant {variant!r} of {canonical!r} is not a string")
            key = _base_key(collapse_ws(fold(variant)))
            if out.get(key, target) != target:
                raise ValidationError(f"{path}: {variant!r} is listed under two different companies")
            out[key] = target
    chained = sorted(set(out.values()) & {k for k, v in out.items() if k != v})
    if chained:
        raise ValidationError(f"{path}: {chained} are both a company and a variant of another; list each once")
    return {k: v for k, v in out.items() if k != v}


def _base_key(folded: str) -> str:
    text = _APOSTROPHES.sub("", _PARENTHETICAL.sub(" ", folded))
    text = text.replace("&", " and ").replace("+", " and ")
    words = _NON_ALNUM.sub(" ", text).split()
    if len(words) > 1 and words[0] == "the":
        words = words[1:]
    while len(words) > 1 and words[-1] in _LEGAL_FORMS:
        words.pop()
    return " ".join(words) or collapse_ws(_NON_ALNUM.sub(" ", folded)) or folded
