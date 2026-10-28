"""The shopper's word for a product: "a shirt", "a pain relief tablet", "two of your shirts", "a pair of trousers".

Why a small module. Openings are read by the agent as a person's request, so "I'm looking for a t-shirts" or "a
trousers" would be noise the benchmark then measures. The noun is the store's own product type, else the last step
of its category path, at most three words with the last made singular; failing both, the last word of the title.
Garments and tools that only come in pairs read "a pair of ...". The same noun groups products onto one shelf, so a
task's candidates are the products a shopper asking for that noun would compare.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from functools import lru_cache

from carttrace.agents.products import Product
from carttrace.agents.text import fold, has_phrase
from carttrace.sandbox.catalogs.kinds import Kinds, colourless

PAIRED = frozenset(
    {
        "binoculars",
        "boxers",
        "briefs",
        "chinos",
        "earbuds",
        "glasses",
        "headphones",
        "jeans",
        "joggers",
        "leggings",
        "pajamas",
        "pants",
        "pliers",
        "pyjamas",
        "scissors",
        "shorts",
        "sunglasses",
        "sweatpants",
        "tights",
        "tongs",
        "trousers",
        "trunks",
    }
)
"""Nouns with no singular that a shopper buys "a pair of"."""
MAX_WORDS = 3
_KEEP = frozenset({"series", "species", "news", "gas", "glass", "dress", "lens", "canvas", "bus", "kiss", "cactus"})
_SPLIT = re.compile(r"\s+[-|/(]\s*|\s*&\s*|\s+and\s+|,\s*")
_WORD = re.compile(r"[a-z][a-z'-]*[a-z]|[a-z]")
_LANGUAGE = re.compile(r"^[a-z]{2}:")
"""A taxonomy tag's language prefix ("en:gummy-candies", as OpenFoodFacts writes them)."""
_TAG_HYPHEN = re.compile(r"\b([a-z]{2,})-([a-z]{2,})\b")
"""A taxonomy tag's hyphen between words ("gummy-candies"); "t-shirt" keeps its hyphen."""


@dataclass(frozen=True, slots=True)
class Noun:
    singular: str
    plural: str
    mass: bool = False
    """Bought as "some" of it: no article ("some coffee")."""

    @property
    def head(self) -> str:
        """The last word: what a product's title must contain to be this kind of thing."""
        return self.singular.rsplit(" ", 1)[-1]

    def with_article(self) -> str:
        if self.mass:
            return f"some {self.singular}"
        return f"{article(self.singular)} {self.singular}"


_YOU_SOUND = re.compile(r"^(?:u[bcdfgklmnprstvz][aeiou]|eu|uni|use|usb|ewe|one|once)")
"""Words spelt with a vowel that start with a consonant sound: "a ukulele", "a utensil", "a unicorn", "a one-piece"."""
_SILENT_H = re.compile(r"^(?:hour|honest|honou?r|heir)")
_ARTICLE = re.compile(r"\b([Aa]n?) ([A-Za-z][\w'-]*)")
_DOUBLE_THE = re.compile(r"\b([Tt]he) the\b", re.IGNORECASE)


def article(word: str) -> str:
    """ "a" or "an" before `word`, by its first sound as far as spelling tells."""
    w = word.lower()
    if _SILENT_H.match(w):
        return "an"
    if not w[:1] or w[:1] not in "aeiou" or _YOU_SOUND.match(w):
        return "a"
    return "an"


def tidy(text: str) -> str:
    """An opening's articles agreeing with the word after them ("an ukulele" -> "a ukulele", "a umbrella" -> "an
    umbrella"), and a title that begins with "The" not read as "the The Pastry Shop" (D104, a published release)."""
    text = _DOUBLE_THE.sub(lambda m: m.group(1), text)

    def fix(m: re.Match[str]) -> str:
        said, word = m.group(1), m.group(2)
        if said.lower() == "a" and len(word) == 1:
            return m.group(0)  # "a B vitamin", a letter
        right = article(word)
        right = right.capitalize() if said[:1].isupper() else right
        return f"{right} {word}"

    return _ARTICLE.sub(fix, text)


def kind_noun(product: Product, kinds: Kinds) -> Noun | None:
    """The product's kind as a shopper names it (carttrace.sandbox.catalogs.kinds, D104): a reviewed product noun its own
    title says, preferring its type's or category's phrase; "a pair of" for kinds bought in pairs, "some" for mass
    kinds. None when the title names no such noun, and then no task is written around the product. A modified colour
    ("Light Pink") is read out of the title and the labels first: ep-77587792ae1e asked for "a light" from "6mm Smooth
    Beads Light Pink 100pcs per pack", because "pink" is filler and "light" a lexicon noun."""
    labels = [colourless(product.product_type or ""), colourless(product.categories[-1]) if product.categories else ""]
    kind = kinds.of(colourless(product.title), [label for label in labels if label])
    if kind is None:
        return None
    if kind.paired:
        lead = " ".join([*kind.words[:-1], kinds.paired[kind.head]])
        return Noun(f"pair of {lead}", f"pairs of {lead}")
    if kind.mass:
        return Noun(kind.phrase, kind.phrase, mass=True)
    return Noun(kind.phrase, " ".join([*kind.words[:-1], plural_of(kind.head)]))


LABEL_WORDS = frozenset(
    {
        "all",
        "apparel",
        "body",
        "care",
        "catalog",
        "collection",
        "color",
        "colour",
        "combined",
        "default",
        "digital",
        "event",
        "face",
        "frizz",
        "full",
        "general",
        "gwp",
        "hair",
        "hydration",
        "insurance",
        "item",
        "merch",
        "merchandise",
        "misc",
        "miscellaneous",
        "new",
        "other",
        "physical",
        "product",
        "repair",
        "restock",
        "restore",
        "retail",
        "sale",
        "signature",
        "skin",
        "style",
        "swag",
        "technology",
        "volume",
        "wellness",
        "wholesale",
    }
)
"""Heads of store labels that are not a kind of thing a shopper asks for: a sales channel ("Wholesale", "Physical"),
a promotion ("Forever Summer Event"), a product line ("Signature"), a body part or a benefit ("Body", "Volume")."""
COLOURS = frozenset(
    {"beige", "black", "blue", "brown", "charcoal", "gray", "green", "grey", "navy", "pink", "purple", "red", "slate",
     "teal", "white", "yellow"}
)  # fmt: skip
"""Colour words that are never a product themselves (not cream, rose, gold or silver, which can be): never what a
product is ("I need a gray"), so never a noun's head, and dropped from the
front of a label ("White Washer & Dryer Sets" is a washer)."""
BODY_PARTS = frozenset(
    {"arm", "back", "cervical", "chest", "ear", "eye", "foot", "hand", "leg", "lip", "nail", "neck", "nose", "waist"}
)
"""Where a product goes, not what it is: "Neck & Cervical Pillows" is a pillow, "Neck & Decollete" no noun at all."""
TITLE_MOTIFS = frozenset(
    {"blessing", "dream", "fortune", "fun", "heart", "hope", "joy", "life", "love", "luck", "magic", "moon", "peace",
     "star", "sun", "warrior"}
)  # fmt: skip
"""A title's last word that names its motif, not the product ("... Good Luck", "... Crescent Moon"): never the noun
read from a title (2026-10-16, Lazada titles in the open pools)."""
LEADING_FILLER = frozenset({"all", "assorted", "more", "other", "shop", "various"})
"""Words that open a store's browse label without naming the kind ("All Washing Machines")."""
_TITLE_TAIL = re.compile(r"\s+(?:with|in|for|featuring|w/)\s+.*$")
"""Where a title stops naming the product and starts describing it: "top load washer in gray", "washer with steam"."""
DESCRIPTORS = frozenset({"edition", "liter", "litre", "mini", "original", "originale", "refill", "size", "travel"})
"""Last words of a title that say its size or edition, not what it is ("Treatment - Retail Size")."""


def _plain_label(head: str) -> bool:
    return head in LABEL_WORDS or singular_of(head) in LABEL_WORDS or singular_of(head) in BODY_PARTS


def noun_of(product: Product) -> Noun:
    """The shopper's noun for a product: its type, else its last category, else the last word of its title. A type or
    category is the noun only when the product's own title says it and it is not a store label (`LABEL_WORDS`).

    Why (2026-10-16). Overnight sandbox tasks asked for "the cheapest wholesale", "a forever summer event", "the body
    that works out cheapest per 100 ml" and "the cheapest signature you have": a store's product_type or collection
    taken as the noun. Of 942 rows built to pay, the 15 tasks with an order rate of a fifth or less held 230 rows, and
    eight of those 15 asked for a label the store's titles never call a product.

    Why colours and cut titles (2026-10-16). Open-lane tasks on WANDS twins opened "I need a gray." (the title "...
    top load washer in gray" read from its last word) and "a load washing machine" (the label "Top Load Washing
    Machines" cut to its last three words): a colour is never the noun, a title is read up to where it starts
    describing ("in", "with", "for"), and a label longer than three words keeps its last two.

    Why every part of a label (2026-10-16). Lazada tasks asked for "a neck" ("Neck & Cervical Pillows" read from its
    first part), "a moon" and "a luck" (a title's motif when "Caps & Hats" named no cap): each part of a label is a
    candidate the title may name, a body part is no noun, and a title's motif is never read as one."""
    labels = [product.product_type or "", product.categories[-1] if product.categories else ""]
    nouns = list(dict.fromkeys(n for label in labels for n in label_nouns(label)))
    for noun in nouns:
        if names_product(noun, product.title):
            return noun
    titled = noun_from(_TITLE_TAIL.sub("", fold(product.title)), max_words=1)
    refused = DESCRIPTORS | COLOURS | TITLE_MOTIFS
    if titled is not None and not _plain_label(titled.head) and titled.head not in refused:
        return titled
    for noun in nouns:
        if not _plain_label(noun.head):
            return noun
    if titled is not None and not _plain_label(titled.head) and titled.head not in TITLE_MOTIFS:
        return titled
    return Noun("item", "items")


def label_nouns(label: str) -> list[Noun]:
    """The nouns of each part of a label's last step, in order: "Caps & Hats" gives cap and hat."""
    path = _LANGUAGE.sub("", fold(label).rsplit(">", 1)[-1].strip())
    parts = [p for p in _SPLIT.split(path) if p.strip()] if path else []
    found = [noun_from(p, max_words=MAX_WORDS) for p in parts] if len(parts) > 1 else [noun_from(label)]
    return [n for n in found if n is not None]


def names_product(noun: Noun, title: str) -> bool:
    """The noun is a kind of product the title is: its head is no store label and the title says it, not as a
    colour's modifier ("Light Blue Heat Press Machine" is no light, ep-77587792ae1e)."""
    head = noun.head
    if _plain_label(head):
        return False
    said = colourless(title)
    return any(has_phrase(said, form) for form in (noun.singular, head, plural_of(head)))


@lru_cache(maxsize=1 << 15)
def noun_from(text: str, *, max_words: int = MAX_WORDS) -> Noun | None:
    """The noun phrase of a label: "Clothing > Shirts & Tops" gives shirt, "Pain Relief Tablets" gives pain relief
    tablet, and with max_words=1, "Linen Camp Shirt - Navy" gives shirt. Remembered: a shelf is every product of
    one noun, so task generation asks for the same labels' nouns many times over."""
    path = _LANGUAGE.sub("", fold(text).rsplit(">", 1)[-1].strip())
    head = _SPLIT.split(path)[0] if path else ""
    head = _TAG_HYPHEN.sub(r"\1 \2", _TAG_HYPHEN.sub(r"\1 \2", head))
    found = [w.strip("'-") for w in _WORD.findall(head)]
    found = [w for w in found if w and not w.isdigit()]
    while found and (found[0] in COLOURS or found[0] in LEADING_FILLER):
        found = found[1:]
    if not found or len(found[-1]) < 3 or found[-1] in COLOURS:
        return None
    words = found[-max_words:] if len(found) <= max_words else found[-max(1, max_words - 1) :]
    last = words[-1]
    if last in PAIRED:
        lead = " ".join(words)
        return Noun(f"pair of {lead}", f"pairs of {lead}")
    singular = " ".join([*words[:-1], singular_of(last)])
    return Noun(singular, " ".join([*words[:-1], plural_of(singular_of(last))]))


def singular_of(word: str) -> str:
    if word in _KEEP or word.endswith(("ss", "us", "is")):
        return word
    if word.endswith("ies") and len(word) > 4:
        return word[:-3] + "y"
    if word.endswith(("ches", "shes", "sses", "xes", "zes")):
        return word[:-2]
    if word.endswith("s"):
        return word[:-1]
    return word


def plural_of(word: str) -> str:
    if word.startswith("pair of "):
        return "pairs of " + word.removeprefix("pair of ")
    if word.endswith("y") and word[-2:-1] not in ("a", "e", "o", "u"):
        return word[:-1] + "ies"
    if word.endswith(("s", "x", "z", "ch", "sh")):
        return word + "es"
    return word + "s"


# -- a product as a person names it (D119) -----------------------------------------------------------------------------

SHORT_TITLE_CHARS = 60
SHORT_TITLE_WORDS = 8
"""A title at most this long is said as it is; a longer one (a marketplace's SEO title) is said in short."""
NAME_LEAD = 2
"""Words of the title said before the kind's head: the key attribute ("Fishtail Denim Skirt")."""
NAME_LEAD_MAX = 4
NO_BRANDS = frozenset({"no brand", "nobrand", "oem", "generic", "unbranded", "none", "n/a", "na", "other"})
"""What a marketplace puts in the brand field of a product with none: never said."""
LEAD_SKIP = frozenset(
    {"plus", "size", "sizes", "new", "hot", "sale", "women", "womens", "women's", "men", "mens", "men's", "ladies",
     "lady", "girls", "boys", "kids", "unisex", "korean", "fashion", "style", "stylish", "high", "quality", "best",
     "free", "shipping", "cod", "ready", "stock", "original", "official", "premium", "2023", "2024", "2025", "2026",
     "for", "and", "with", "the", "a", "an", "of", "in"}
)  # fmt: skip
"""Title words a person leaves out of a product's name: sizes on offer, audiences, a seller's selling words."""
AUDIENCES: tuple[tuple[frozenset[str], str], ...] = (
    (frozenset({"girls", "girl's", "girls'"}), "Girls'"),
    (frozenset({"boys", "boy's", "boys'"}), "Boys'"),
    (frozenset({"kids", "kid's", "kids'", "children", "childrens", "children's", "child", "child's"}), "Kids'"),
    (frozenset({"baby", "babies", "toddler", "toddlers", "infant", "infants", "newborn"}), "Baby"),
    (frozenset({"women", "womens", "women's", "ladies", "lady"}), "Women's"),
    (frozenset({"men", "mens", "men's"}), "Men's"),
)
"""Who a product is for, as a person says it. A short name keeps the first one its title states: "Girls Cotton Linen
Short Pants ... Childrens" said as "Linen Short Pants" sent an agent to adult shorts (2026-10-17 trace test)."""
_NAME_WORD = re.compile(r"^[A-Za-z][A-Za-z'&.-]*$")
_SEGMENT_END = re.compile(r"[,;|/()\[\]]$")


def _bare(word: str) -> str:
    return word.strip(",.;:!?()[]{}\"'").lower()


def spoken_name(title: str, noun: Noun, brand: str | None = None, *, lead: int = NAME_LEAD) -> str:
    """The product as a person would name it in a request (D119): a title of a few words as it is; a long one as its
    brand, the `lead` words before the kind's head in the title (its key attribute) and the head itself, as one run of
    the title's own words ("Plus Size S-5XL Women Fishtail Denim Skirt ... Jean Midi Skirt Blue" -> "Fishtail Denim
    Skirt"), deterministic from the title. A title that never says the kind's head is said by its first naming
    segment, cut to SHORT_TITLE_WORDS words. The full title stays in the hidden spec, which is what is matched."""
    title = " ".join(title.split())
    words = title.split()
    if len(title) <= SHORT_TITLE_CHARS and len(words) <= SHORT_TITLE_WORDS:
        return title
    head = noun.head
    forms = {head, plural_of(head), singular_of(head)}
    at = next((i for i, w in enumerate(words) if _bare(w) in forms), None)
    if at is None:
        first = re.split(r"\s+[-|/(]\s*|,\s*", title, maxsplit=1)[0].split()
        phrase = " ".join(first[:SHORT_TITLE_WORDS]).strip(",.;:-")
    else:
        start = at
        while start > 0 and at - start < lead:
            word = words[start - 1]
            if _SEGMENT_END.search(word) or not _NAME_WORD.match(word) or _bare(word) in LEAD_SKIP:
                break
            start -= 1
        phrase = " ".join(words[start : at + 1]).strip(",.;:-")
    title_words = {_bare(w) for w in words}
    for audience, said_as in AUDIENCES:
        if title_words & audience:
            if not {_bare(w) for w in phrase.split()} & audience:
                phrase = f"{said_as} {phrase}"
            break
    said = (brand or "").strip()
    if said and fold(said) not in NO_BRANDS and len(said) <= 30 and not has_phrase(fold(phrase), fold(said)):
        phrase = f"{said} {phrase}"
    return phrase


NOTE_WORDS = frozenset({"model", "models", "colors", "colours", "version", "pcs", "sku"})
"""Words of a seller's note inside an option value ("solid colors vests models-black"), never of the value itself."""
_CODE = re.compile(r"\b[a-z]{2,}\d{3,}[a-z0-9]*\b", re.IGNORECASE)
"""A model code ("sm2301", "ab123x"): letters then three digits or more."""
_VALUE_SPLIT = re.compile(r"\s*[-_/|:+()]\s*|\s{2,}")
VALUE_WORDS = 2
VALUE_CHARS = 24
MIN_VALUE_LETTERS = 3
"""A part shorter than this is a code, not a value a person says: "2pcs-g" was said as "I'd like it in g"."""


def _noted(text: str) -> bool:
    return bool({_bare(w) for w in text.split()} & NOTE_WORDS) or _CODE.search(text) is not None


def plain_value(value: str, ok: bool = True) -> str | None:
    """An option value as a person says it (D119): the value itself when it is plain (`ok`, the caller's own test of a
    sayable value, and no seller's note or model code in it); else the plain value the note ends in ("solid colors
    vests models-black" -> "black", "SM2301-Navy" -> "Navy"), the last part that is short and holds no note or code;
    None when no part is plain."""
    text = " ".join(value.split())
    if ok and not _noted(text):
        return text
    parts = [p.strip() for p in _VALUE_SPLIT.split(text) if p.strip()]
    plain = [
        p
        for p in parts
        if len(p) <= VALUE_CHARS
        and len(p.split()) <= VALUE_WORDS
        and not _noted(p)
        and sum(c.isalpha() for c in p) >= MIN_VALUE_LETTERS
        and not any(c.isdigit() for c in p)
    ]
    return plain[-1] if plain and len(parts) > 1 else None
