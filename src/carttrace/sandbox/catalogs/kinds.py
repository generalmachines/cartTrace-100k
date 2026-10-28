"""What a product is, read from its own title, and what that kind of thing plausibly costs (D104).

Why. The open stand-ins' tasks named products by a word read from a category leaf or the end of a title, and a
release of 20 traces showed what that gives: "I want a folk" (the leaf "Folk & World Strings" over a set of guitar
strings), "Please just get me a babymama under $200", "a booty", "a quality", "a lighting", and a compare task over
"your pregnancy pillows" in a store whose one such product was a newborn's pillow under a mislabelled Lazada leaf. The
same leaves priced a rubber car floor mat at $399.99, because a synthetic price was drawn from its category's band.

What a kind is. The reviewed lexicon data/catalogs/kinds.yaml lists the head nouns a shopper buys (shirt, mat,
pillow), each with the range of US dollars one costs, a few two-word kinds whose range differs (coffee table, floor
mat), the heads bought as a pair (shoes, trousers) and those bought as "some" (coffee, rice). A product's kind is a
lexicon noun its own title says in head position, that is not followed by another lexicon noun ("guitar" in "guitar
strap" is not the kind; "strap" is):
- a phrase from the product's own type or category leaf wins when the title says it ("Hoop Earrings" over "... Hoop
  Earrings for Women"), else the last head the title's first naming segment gives (before "with", "for", "in", a
  comma, a bracket or a spaced dash);
- the kind keeps one or two words before the head when the leaf or the lexicon names that phrase ("running shoe",
  "floor mat"), never a word the title does not say.
A title with no lexicon head has no kind: such a product can stand on a store's shelf but no task is written around
it. Standard library and YAML only, so the pool build (this layer) and the task writer (carttrace.agents.nouns) read one rule.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache, lru_cache
from pathlib import Path
from typing import Final

import yaml

from carttrace.core.canon import digest
from carttrace.core.errors import NotFoundError, ValidationError

KINDS_FILE: Final = "kinds.yaml"
KINDS_SCHEMA: Final = "mc.catalog-kinds.v1"
DEFAULT_PATH: Final = Path(__file__).resolve().parents[4] / "data" / "catalogs" / KINDS_FILE
MAX_PHRASE: Final = 3
"""Words a kind may have: its head and at most two words before it."""
FILLER: Final = frozenset(
    {"set", "pack", "pc", "pcs", "piece", "new", "original", "authentic", "only", "sale", "free", "size", "small",
     "medium", "large", "xl", "xxl", "xxxl", "plus", "mini", "big", "black", "white", "red", "blue", "green", "gray",
     "grey", "pink", "brown", "navy", "beige", "yellow", "purple", "orange", "silver", "gold", "ml", "oz", "cm", "mm",
     "inch", "kg", "lb", "pair", "unisex", "men", "women", "kids", "ready", "stock", "hot", "best", "seller",
     "brand", "gift", "value", "cod", "color", "colour", "made",
     "suitable", "quality", "design", "style", "collection", "material", "compatible", "waterproof", "delivery",
     "durable", "fashion", "universal", "take", "of", "per", "and", "to", "use", "find", "market", "woman", "man",
     "bundle", "thick", "duty", "type", "shipping", "home", "heavy", "premium", "korean", "cute", "simple", "classic"}
)  # fmt: skip
"""Words a title may end with after its noun: a colour, a size, a count, an audience, a seller's cry. Not "accessories",
"supplies" or "decoration": "Guitar Accessories" is no guitar, and a capo priced as one cost $219.99 (D104)."""
LEADING_ONLY: Final = 3
"""A segment of this many words whose only head is its first word names no kind ("brush dinosaur graphic art")."""

_SEGMENTS = re.compile(r"\s[|/]\s|\s[-–—]\s|[,;()\[\]{}|【】]")
"""Where a title stops naming one thing: a bracket, a comma, a bar or a spaced dash."""
_TAIL = re.compile(r"(?:^|\s)(?:with|in|for|featuring|w/)\s.*$")
"""Where a segment stops naming the product and starts describing it ("washer in gray", "pillow for kids")."""
COLOURS: Final = frozenset(
    {"black", "white", "red", "blue", "green", "gray", "grey", "pink", "brown", "navy", "beige", "yellow", "purple",
     "orange", "silver", "gold", "golden", "rose", "ivory", "cream", "teal", "turquoise", "maroon", "burgundy",
     "violet", "lavender", "coral", "peach", "mint", "olive", "khaki", "tan", "charcoal", "aqua", "cyan", "magenta",
     "lilac", "mauve", "nude", "champagne", "wine", "sky", "royal", "salmon", "emerald", "fuchsia", "indigo"}
)  # fmt: skip
"""Colour words. With a modifier before them ("light pink", "dark navy") they name a colour, never a product."""
COLOUR_MODIFIERS: Final = frozenset(
    {"light", "dark", "pale", "deep", "bright", "hot", "baby", "royal", "sky", "dusty", "pastel", "neon", "soft",
     "rich", "mini", "matte", "metallic", "jet", "off"}
)  # fmt: skip
"""Words that, before a colour word, modify the colour: "Light" in "Light Pink" is no light (ep-77587792ae1e)."""
_COLOUR_PHRASE = re.compile(
    rf"\b(?:{'|'.join(sorted(COLOUR_MODIFIERS))})[\s-]+(?:{'|'.join(sorted(COLOURS))})\b", re.IGNORECASE
)


def colourless(title: str) -> str:
    """`title` with every modified colour ("Light Pink", "dark-blue") taken out, so a colour's modifier is never
    read as the product's noun. A colour word alone is already filler (FILLER) and stays."""
    return " ".join(_COLOUR_PHRASE.sub(" ", title).split())


_LABEL_PARTS = re.compile(r"\s+[-|/(]\s*|\s*&\s*|\s+and\s+|,\s*")
_WORD = re.compile(r"[a-z][a-z'-]*[a-z]|[a-z]")
_KEEP = frozenset(
    {
        "series",
        "species",
        "news",
        "gas",
        "glass",
        "dress",
        "lens",
        "canvas",
        "bus",
        "kiss",
        "cactus",
        "chaise",
        "clothes",
    }
)
_IRREGULAR: Final[Mapping[str, str]] = {
    "knives": "knife",
    "shelves": "shelf",
    "leaves": "leaf",
    "wolves": "wolf",
    "halves": "half",
    "loaves": "loaf",
    "scarves": "scarf",
    "cookies": "cookie",
    "brownies": "brownie",
    "hoodies": "hoodie",
    "movies": "movie",
    "pies": "pie",
    "ties": "tie",
    "mice": "mouse",
    "feet": "foot",
    "teeth": "tooth",
    "children": "child",
    "men": "man",
    "women": "woman",
    "potatoes": "potato",
    "tomatoes": "tomato",
}


def singular_of(word: str) -> str:
    """The singular of one lower-case English word, for matching titles to the lexicon."""
    word = word.removesuffix("'s").removesuffix("'")
    if word in _IRREGULAR:
        return _IRREGULAR[word]
    if word in _KEEP or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s") and len(word) > 2:
        return word[:-1]
    return word


def plural_of(word: str) -> str:
    if word.endswith("y") and word[-2:-1] not in ("a", "e", "o", "u"):
        return word[:-1] + "ies"
    if word.endswith("fe"):
        return word[:-2] + "ves"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


@dataclass(frozen=True, slots=True)
class Kind:
    """A kind of product: `words` in the singular, the last being the head."""

    words: tuple[str, ...]
    band: tuple[int, int]
    """Whole US dollars a US shopper would pay for one."""
    paired: bool = False
    """Bought as "a pair of" its plural."""
    mass: bool = False
    """Bought as "some" of it, with no plural."""

    @property
    def head(self) -> str:
        return self.words[-1]

    @property
    def phrase(self) -> str:
        return " ".join(self.words)


@dataclass(frozen=True, slots=True, eq=False)
class Kinds:
    """The reviewed lexicon (data/catalogs/kinds.yaml)."""

    rule: str
    nouns: Mapping[str, tuple[int, int]]
    phrases: Mapping[str, tuple[int, int]]
    paired: Mapping[str, str]
    mass: frozenset[str]
    per_usd: Mapping[str, int]
    """Units of a currency to the US dollar, for converting a real price."""

    @property
    def digest(self) -> str:
        return digest(
            {
                "rule": self.rule,
                "nouns": {k: list(v) for k, v in sorted(self.nouns.items())},
                "phrases": {k: list(v) for k, v in sorted(self.phrases.items())},
                "paired": dict(sorted(self.paired.items())),
                "mass": sorted(self.mass),
                "per_usd": dict(sorted(self.per_usd.items())),
            },
            16,
        )

    def known(self, word: str) -> bool:
        return word in self.nouns

    def named_at(self, toks: Sequence[str], i: int) -> bool:
        """Whether toks[i] ends a kind: a lexicon noun, or the head of a listed phrase the words before complete
        ("drawer pull", "screen protector"; "pull" and "protector" alone are no kind)."""
        word = toks[i]
        if word in self.nouns:
            return True
        return any(
            " ".join(toks[i - n + 1 : i + 1]) in self.phrases for n in range(2, MAX_PHRASE + 1) if i - n + 1 >= 0
        )

    def kind(self, words: Sequence[str]) -> Kind:
        """The kind of a phrase that ends a kind (`named_at`): the longest suffix the lexicon prices on its own sets
        the band, else the head's."""
        head = words[-1]
        band = self.nouns.get(head)
        for start in range(len(words) - 1):
            found = self.phrases.get(" ".join(words[start:])) or self.phrases.get(
                " ".join(singular_of(w) for w in words[start:])
            )
            if found is not None:
                band = found
                break
        if band is None:
            raise ValidationError(f"{' '.join(words)!r} is no kind in the lexicon", hint="ask Kinds.of for a kind")
        return Kind(tuple(words), band, paired=head in self.paired, mass=head in self.mass)

    def of(self, title: str, labels: Sequence[str] = ()) -> Kind | None:
        """The kind `title` names, preferring a phrase of `labels` (the product's type and category leaf) the title
        says; None when the title names no lexicon noun in head position."""
        return _kind_of(self, title, tuple(labels))


@lru_cache(maxsize=1 << 16)
def _kind_of(kinds: Kinds, title: str, labels: tuple[str, ...]) -> Kind | None:
    segments = [_tokens(_TAIL.sub("", s.strip())) for s in _SEGMENTS.split(_fold(title))]
    segments = [(raw, sing) for raw, sing in segments if sing]
    wanted = _label_phrases(kinds, labels)
    best: tuple[int, Sequence[str], Sequence[str], int] | None = None
    for raw, toks in segments:
        for i in _heads(kinds, toks):
            for phrase in wanted:
                n = len(phrase)
                if tuple(toks[max(0, i - n + 1) : i + 1]) == phrase and (best is None or n > best[0]):
                    best = (n, raw, toks, i)
    if best is not None:
        n, said, words, i = best
        return kinds.kind(_said(said, words, i - max(n, _listed(kinds, words, i)) + 1, i))
    for raw, toks in segments:
        last = _last_word(toks)
        if last is None or last not in _heads(kinds, toks) or (last == 0 and len(toks) >= LEADING_ONLY):
            continue
        n = _listed(kinds, toks, last)
        if n > 1 or kinds.known(toks[last]):
            return kinds.kind(_said(raw, toks, last - n + 1, last))
    return None


def _last_word(toks: Sequence[str]) -> int | None:
    """The position of a segment's last word that is not filler (a size, a colour, a count word): where an English
    noun phrase has its head. A lexicon noun anywhere else in a title with no label to back it is a modifier more often
    than not ("Tire Sealant", "Tire Shine Polish and Protectant 1L")."""
    for i in range(len(toks) - 1, -1, -1):
        if len(toks[i]) > 1 and toks[i] not in FILLER:
            return i
    return None


def _listed(kinds: Kinds, toks: Sequence[str], head: int) -> int:
    """How many words of the longest lexicon phrase ending at `head` ("floor lamp" gives 2), else 1."""
    for n in range(MAX_PHRASE, 1, -1):
        if head - n + 1 >= 0 and " ".join(toks[head - n + 1 : head + 1]) in kinds.phrases:
            return n
    return 1


def _said(raw: Sequence[str], toks: Sequence[str], start: int, head: int) -> tuple[str, ...]:
    """A kind's words as the title says them before its head ("kids", "women's"), and its head in the singular."""
    return (*raw[start:head], toks[head])


def _fold(text: str) -> str:
    plain = text.replace("\u2019", "'").replace("\u2013", "-").replace("\u2014", "-")
    return " ".join(plain.lower().split())


def _tokens(text: str) -> tuple[list[str], list[str]]:
    """The words of `text` as written and in the singular, side by side."""
    raw = [w.strip("'-") for w in _WORD.findall(text) if w.strip("'-")]
    return raw, [singular_of(w) for w in raw]


def _heads(kinds: Kinds, toks: Sequence[str]) -> list[int]:
    """Positions of lexicon nouns not followed by another lexicon noun (which would make them a modifier)."""
    return [
        i
        for i in range(len(toks))
        if kinds.named_at(toks, i) and (i + 1 == len(toks) or not kinds.named_at(toks, i + 1))
    ]


def _label_phrases(kinds: Kinds, labels: Sequence[str]) -> list[tuple[str, ...]]:
    """Every suffix of each part of the labels' last step whose head is a lexicon noun: "Hoop Earrings" gives
    ("hoop", "earring") and ("earring",)."""
    out: list[tuple[str, ...]] = []
    for label in labels:
        last = _fold(label).rsplit(">", 1)[-1].replace("_", " ").strip()
        for part in _LABEL_PARTS.split(last):
            toks = _tokens(part)[1]
            if toks and kinds.named_at(toks, len(toks) - 1):
                for n in range(1, min(MAX_PHRASE, len(toks)) + 1):
                    if kinds.named_at(toks[-n:], n - 1):
                        out.append(tuple(toks[-n:]))
    return list(dict.fromkeys(out))


def load_kinds(path: Path | None = None) -> Kinds:
    """The lexicon at `path` (default: data/catalogs/kinds.yaml), parsed once per process and checked."""
    return _load((path or DEFAULT_PATH).resolve())


@cache
def _load(path: Path) -> Kinds:
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise NotFoundError(f"no product kinds at {path}", hint="restore data/catalogs/kinds.yaml from git") from None
    if not isinstance(raw, dict) or raw.get("schema") != KINDS_SCHEMA or not isinstance(raw.get("rule"), str):
        raise ValidationError(f"{path.name} is not a {KINDS_SCHEMA} file with a rule name")

    def bands(value: object, where: str) -> dict[str, tuple[int, int]]:
        if not isinstance(value, dict):
            raise ValidationError(f"{path.name}: {where} must map a noun to [low, high]")
        out: dict[str, tuple[int, int]] = {}
        for key, band in value.items():
            if not (isinstance(band, list) and len(band) == 2 and all(isinstance(b, int) for b in band)):
                raise ValidationError(f"{path.name}: {where}.{key} must be [low, high] whole dollars")
            if not 0 < band[0] <= band[1]:
                raise ValidationError(f"{path.name}: {where}.{key} must be 0 < low <= high, got {band}")
            out[str(key)] = (band[0], band[1])
        return out

    nouns = bands(raw.get("nouns"), "nouns")
    phrases = bands(raw.get("phrases") or {}, "phrases")
    # a phrase is found in a title's singular words ("kid bed" for "Kids Beds") as well as as written
    phrases = {**{" ".join(singular_of(w) for w in k.split()): v for k, v in phrases.items()}, **phrases}
    paired = {str(k): str(v) for k, v in (raw.get("paired") or {}).items()}
    mass = frozenset(str(m) for m in raw.get("mass") or ())
    stray = sorted((set(paired) | mass) - set(nouns))
    if stray:
        raise ValidationError(f"{path.name}: {stray} are not lexicon nouns", hint="add them under nouns first")
    rates = raw.get("per_usd") or {}
    if not isinstance(rates, dict) or not all(isinstance(v, int) and v > 0 for v in rates.values()):
        raise ValidationError(f"{path.name}: per_usd must map a currency code to a positive whole number")
    return Kinds(str(raw["rule"]), nouns, phrases, paired, mass, {str(k): v for k, v in rates.items()})


__all__ = [
    "COLOURS",
    "COLOUR_MODIFIERS",
    "KINDS_FILE",
    "Kind",
    "Kinds",
    "colourless",
    "load_kinds",
    "plural_of",
    "singular_of",
]
