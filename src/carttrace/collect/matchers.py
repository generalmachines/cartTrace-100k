"""The per-detail matchers behind detail_hidden: which keys name a catalog detail, and which sentences, lines and
tables of product text state it.

Why one module. A detail_hidden experiment that names details (allergens, size_chart, care...) must hide exactly
those facts and leave the rest of the listing word for word, or the benchmark measures what most of a listing is
worth instead of what one detail is worth. That needs one answer per detail to two questions: does this attribute
key name the detail ("Allergens", "Care instructions", "Fabric"), and does this sentence, line or table state it
("May contain nuts", "Machine wash cold", a size table). The answers reuse what the repository already reads details
with, so hiding and reading cannot drift apart:
- the patterns the sandbox catalogs lift details out of free text with (`carttrace.sandbox.catalogs.text`): a line a
  catalog read a detail from is a line that states it;
- the readers of the decisive details (`carttrace.agents.details`): every statement marker the allergen reader looks for,
  every warning phrase the warning reader looks for, and every table and row the size chart reader reads states the
  detail, so the reference agent finds none of it in the hidden arm;
- `carttrace.agents.text` to fold text and match whole-word phrases.

Keys and labels are matched as phrases, and a phrase inside a longer one does not count: "pack size" names pack_size
and not size, "weight capacity" names nothing, and "Fabric & care" names material and care. The vocabulary of names
is data (data/categories/_fields.yaml `details`); a test fails when a name there has no matcher here.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass

from carttrace.agents.details.allergens import ALLERGENS as ALLERGEN_NAMES
from carttrace.agents.details.allergens import STATEMENT_MARKERS
from carttrace.agents.details.size_chart import MEASURES, SIZE_CHART_KEYS, read_chart, text_row
from carttrace.agents.details.warnings import WARNING_PHRASES
from carttrace.agents.products import Grid, Product
from carttrace.agents.text import fold, has_phrase
from carttrace.core.errors import ValidationError
from carttrace.sandbox.catalogs.text import ALLERGENS as CATALOG_ALLERGENS
from carttrace.sandbox.catalogs.text import CARE, COMPATIBLE, DIMENSIONS, DIRECTIONS, INGREDIENTS, WARNINGS

LABEL_CHARS = 64
"""How far into a sentence a label may end ("Allergen information: ..."); a colon further in is not a label's."""
HEADING_CHARS = 60
"""The longest text read as a heading ("Ingredients", "Fabric & care")."""
FILLER = frozenset(
    {
        "a",
        "about",
        "advice",
        "an",
        "and",
        "approx",
        "approximate",
        "chart",
        "cm",
        "complete",
        "data",
        "declaration",
        "detail",
        "details",
        "facts",
        "for",
        "full",
        "general",
        "guide",
        "important",
        "in",
        "inch",
        "inches",
        "info",
        "information",
        "instruction",
        "instructions",
        "item",
        "items",
        "key",
        "kg",
        "label",
        "lb",
        "list",
        "mm",
        "note",
        "notes",
        "of",
        "or",
        "our",
        "per",
        "please",
        "product",
        "products",
        "see",
        "spec",
        "specification",
        "specifications",
        "specs",
        "statement",
        "table",
        "the",
        "this",
        "uk",
        "us",
        "your",
    }
)
"""Words a heading or a label may hold besides the names of details: "Product details & care", "Size chart (cm)"."""
NOT_DETAILS = (
    "customer care",
    "customer service",
    "skin care",
    "hair care",
    "personal care",
    "health care",
    "baby care",
    "pet care",
    "lawn care",
    "nail care",
    "oral care",
    "foot care",
    "wound care",
    "eye care",
    "body care",
    "self care",
    "warranty length",
    "length of warranty",
    "screen size",
    "font size",
    "file size",
    "weight capacity",
    "weight limit",
    "maximum weight",
    "max weight",
    "max user weight",
    "maximum user weight",
    "user weight",
    "weight rating",
    "paper weight",
    "yarn weight",
    "fabric weight",
    "load capacity",
    "maximum load",
    "weight range",
    "body weight",
    "child weight",
    "rider weight",
    "pet weight",
    "dog weight",
    "baby weight",
    "weight class",
    "weight loss",
    "weight management",
    "thread count",
    "calorie count",
    "volume control",
    "volume button",
    "volume level",
    "order quantity",
)
"""Phrases that hold a detail's label word and name something else: a weight capacity is not the item's weight."""


@dataclass(frozen=True, slots=True)
class Matcher:
    """How one catalog detail is named and stated."""

    name: str
    labels: tuple[str, ...]
    """Phrases that name the detail as an attribute or metafield key, a heading, or a label before a colon. Plurals
    match too ("allergens", "dimensions")."""
    patterns: tuple[re.Pattern[str], ...] = ()
    """What states the detail in a sentence that carries no label, matched on folded text."""
    phrases: tuple[str, ...] = ()
    """Phrases that state it, read as `carttrace.agents.text.has_phrase` reads them: whole words, a trailing * a prefix."""
    check: Callable[[str], bool] | None = None
    """A reader's own test of a line, where one exists (a size chart row)."""

    def states(self, folded: str, raw: str) -> bool:
        """Whether a sentence states the detail by what it says; `folded` is the sentence cleaned and folded."""
        if any(pattern.search(folded) for pattern in self.patterns):
            return True
        if any(phrase.rstrip("*") in folded and has_phrase(folded, phrase) for phrase in self.phrases):
            return True
        return self.check is not None and self.check(raw)


def _any(*words: str) -> str:
    return "(?:" + "|".join(words) + ")"


_FIBRE = _any(
    "cotton",
    "polyester",
    "elastane",
    "spandex",
    "lycra",
    "nylon",
    "polyamide",
    "viscose",
    "rayon",
    "modal",
    "lyocell",
    "tencel",
    "wool",
    "merino",
    "cashmere",
    "silk",
    "linen",
    "hemp",
    "acrylic",
    "leather",
    "alpaca",
    "mohair",
    "bamboo",
    "polypropylene",
    "acetate",
    "cupro",
    "elastodiene",
    "triacetate",
    "polyurethane",
    "down",
    "feathers?",
    "jute",
    "ramie",
)
_MATERIAL = _any(
    _FIBRE,
    "suede",
    "nubuck",
    "canvas",
    "denim",
    "mesh",
    "rubber",
    "eva",
    "foam",
    "cork",
    "straw",
    "raffia",
    "pvc",
    "vinyl",
    "neoprene",
    "felt",
    "velvet",
    "boucle",
    "chenille",
    "oak",
    "walnut",
    "ash",
    "pine",
    "beech",
    "teak",
    "acacia",
    "mango ?wood",
    "maple",
    "cherry",
    "birch",
    "wood",
    "hardwood",
    "softwood",
    "rattan",
    "wicker",
    "cane",
    "mdf",
    "plywood",
    "particle ?board",
    "chipboard",
    "veneer",
    "steel",
    "aluminium",
    "aluminum",
    "iron",
    "brass",
    "copper",
    "zinc",
    "glass",
    "ceramic",
    "porcelain",
    "stoneware",
    "earthenware",
    "bone china",
    "marble",
    "granite",
    "stone",
    "concrete",
    "resin",
    "plastic",
    "polycarbonate",
    "silicone",
    "tritan",
    "melamine",
    "gold",
    "silver",
    "platinum",
    "titanium",
    "tungsten",
    "palladium",
    "pewter",
    "paper",
    "cardboard",
)


def _allergen_words() -> str:
    """Every allergen word the allergen reader knows (carttrace.agents.details.allergens.ALLERGENS, where a trailing * is a
    prefix), and the rest of the EU 14."""
    words = {
        re.escape(word.rstrip("*")) + ("[a-z]*" if word.endswith("*") else "")
        for names in ALLERGEN_NAMES.values()
        for word in names
    }
    words |= {"fish", "shellfish", "crustaceans?", "molluscs?", "mustard", "celery", "lupin", "sulph?ites?", "dairy"}
    return _any(*sorted(words))


_ALLERGEN_WORDS = _allergen_words()
_FRAGRANCE_ALLERGENS = re.compile(
    r"\b(?:limonene|linalool|citral|geraniol|citronellol|coumarin|eugenol|isoeugenol|farnesol|cinnamal|"
    r"cinnamyl alcohol|amyl cinnamal|hexyl cinnamal|benzyl (?:alcohol|benzoate|salicylate|cinnamate)|anise alcohol|"
    r"hydroxycitronellal|alpha-isomethyl ionone|butylphenyl methylpropional|evernia prunastri)\b"
)
"""The fragrance allergens EU labels declare by name."""
_DOSE_UNIT = _any(
    r"tablets?",
    r"capsules?",
    r"caplets?",
    r"softgels?",
    r"gumm(?:y|ies)",
    r"drops?",
    r"sprays?",
    r"ml",
    r"teaspoons?",
    r"tsp",
    r"tablespoons?",
    r"tbsp",
    r"sachets?",
    r"scoops?",
    r"pumps?",
    r"lozenges?",
    r"puffs?",
    r"patch(?:es)?",
)
_COUNT_UNIT = _any(
    r"tablets?",
    r"capsules?",
    r"caplets?",
    r"softgels?",
    r"gummies",
    r"sachets?",
    r"pods?",
    r"bars?",
    r"cans?",
    r"bottles?",
    r"pieces?",
    r"pcs",
    r"rolls?",
    r"sheets?",
    r"wipes?",
    r"bags?",
    r"servings?",
)
_LENGTH_UNIT = r"(?:cm|mm|m|in|inches|inch|ft|feet|\"|'')"
_INGREDIENT_MARKERS = ("ingredient*", "inci", "composition")
"""The allergen reader's statement markers that open an ingredient list; the rest of them are allergen statements."""

_MEASURE_NAMES = sorted({name for names in MEASURES.values() for name in names}, key=len, reverse=True)
_MEASURE_VALUE = re.compile(
    rf"(?<![a-z])({'|'.join(re.escape(name) for name in _MEASURE_NAMES)})"
    rf"(?:\s*[:=-]?\s*\d+(?:[.,]\d+)?(?:\s*(?:-|to)\s*\d+(?:[.,]\d+)?)?\s*(?:cm|mm|inch(?:es)?|in\b|\")"
    r"|\s*\((?:cm|mm|in|inch(?:es)?)\)\s*[:=-]?\s*\d)"
)
_MEASURE_OF = {name: measure for measure, names in MEASURES.items() for name in names}
_ON_THEIR_OWN = frozenset(MEASURES) - {"height", "length", "sleeve", "shoulder"}
"""Body measures that state a size chart on their own ("chest 102 cm"). A height or a length alone is as often a
model's height or a product's size, so it takes a second measure to count."""


def _measures_a_size(line: str) -> bool:
    """Whether a line gives sizes by measurement: a chart row the size chart reader reads ("S: chest 86-91 cm"), a
    body measure with its value, or two measures with values ("length 70 cm, sleeve 64 cm")."""
    if text_row(line) is not None:
        return True
    found = [_MEASURE_OF[m.group(1)] for m in _MEASURE_VALUE.finditer(fold(line))]
    return any(measure in _ON_THEIR_OWN for measure in found) or len(found) >= 2


def _compile(*patterns: str) -> tuple[re.Pattern[str], ...]:
    return tuple(re.compile(pattern) for pattern in patterns)


MATCHERS: dict[str, Matcher] = {
    m.name: m
    for m in (
        Matcher(
            "size",
            labels=("size", "sizes available", "available sizes", "size range"),
            patterns=_compile(
                r"\bavailable in (?:sizes?\b|(?:[a-z]+ ){0,2}sizes\b)",
                r"\b(?:comes?|offered) in (?:\d+ |several |many |all |a range of )?sizes\b",
                r"\bsizes? (?:range |available )?(?:from )?(?:xxs|xs|s|m|l|xl|xxl|\d{1,2}(?:\.5)?)\s*(?:-|to)\s*"
                r"(?:xs|s|m|l|xl|xxl|xxxl|\d{1,2}(?:\.5)?)\b",
            ),
        ),
        Matcher(
            "size_system",
            labels=("size system", "sizing system", "size standard", "size conversion", "size region"),
            patterns=_compile(
                r"\b(?:us|uk|eu|eur|fr|jp|au|cn|mx|br)\s?(?:men'?s |women'?s |kids'? )?siz(?:e|es|ing)\b",
                r"\bsiz(?:e|es|ing) (?:is |are )?(?:given |shown )?in (?:us|uk|eu|eur|fr|jp|au)\b",
                r"\bconversion (?:chart|table)\b",
            ),
        ),
        Matcher(
            "size_chart",
            labels=(
                *SIZE_CHART_KEYS,
                "sizing chart",
                "sizing guide",
                "measurements",
                "body measurements",
                "garment measurements",
                "foot length",
                "insole length",
                "ring size guide",
                "ring sizer",
            ),
            check=_measures_a_size,
        ),
        Matcher(
            "material",
            labels=(
                "material",
                "fabric",
                "fabric composition",
                "composition",
                "fibre",
                "fiber",
                "fibre content",
                "fiber content",
                "textile",
                "upper",
                "lining",
                "sole",
                "outsole",
                "midsole",
                "insole",
                "footbed",
                "shell",
                "outer",
                "filling",
                "metal",
                "metal type",
                "metal purity",
                "purity",
                "karat",
                "wood",
                "wood species",
                "wood type",
                "made of",
                "made from",
                "construction",
            ),
            patterns=_compile(
                rf"\b\d{{1,3}}(?:[.,]\d)?\s?%\s?(?!off\b|discount)(?:[a-z-]+ ){{0,2}}?{_FIBRE}\b",
                rf"\b{_FIBRE}\s?\d{{1,3}}(?:[.,]\d)?\s?%",
                rf"\bmade (?:of|from|with|out of|using) (?:\d{{1,3}}\s?% )?(?:[a-z-]+ ){{0,3}}?{_MATERIAL}\b",
                rf"\b(?:solid|genuine|real|full[- ]grain|top[- ]grain|sterling|stainless|pure|organic|recycled|vegan"
                rf"|faux|tempered|cast|enamell?ed|(?:hard[- ])?anodi[sz]ed|powder[- ]coated|galvani[sz]ed|forged"
                rf"|\d{{1,2}}\s?(?:k|kt|karat|carat|ct)) {_MATERIAL}\b",
                r"\b925\b|\bvermeil\b|\b(?:gold|silver|rhodium)[- ](?:plated|filled)\b|\b18/(?:10|8|0)\b",
                rf"\b(?:upper|lining|sole|outsole|midsole|insole|footbed|straps?|handles?|body|shell|frame|top|base"
                rf"|legs?|seat|back|cushions?|cover|filling|exterior|interior)s? (?:is |are )?(?:made )?(?:of|from|in)"
                rf" (?:[a-z-]+ ){{0,2}}?{_MATERIAL}\b",
            ),
        ),
        Matcher(
            "care",
            labels=(
                "care",
                "care instructions",
                "care guide",
                "care label",
                "product care",
                "fabric care",
                "garment care",
                "washing instructions",
                "wash care",
                "washing",
                "laundry",
                "cleaning",
                "cleaning instructions",
                "how to clean",
                "how to care",
                "maintenance",
                "dishwasher safe",
                "dishwasher",
            ),
            patterns=(
                CARE,
                *_compile(
                    r"\bwash(?:ed|ing)? (?:cold|warm|hot|separately|inside out|by hand|on (?:a )?(?:gentle|delicate"
                    r"|cold|cool|wool|hand)|at \d{2}|below \d{2}|with (?:like|similar) colou?rs)\b",
                    r"\b(?:line|flat|drip|air)[- ]dr(?:y|ying)\b",
                    r"\btumble[- ]dr(?:y|ying)\b",
                    r"\biron(?:ing)? (?:on |at )?(?:a )?(?:low|medium|high|cool|warm|moderate|reverse)\b",
                    r"\bdo not (?:bleach|tumble|iron|wash|dry[- ]clean|wring|soak|machine[- ]wash|put in the "
                    r"dishwasher)\b",
                    r"\b\d{2}\s?°\s?c?\s?(?:wash|max)\b",
                    r"\b(?:not )?dishwasher[- ](?:safe|proof)\b",
                    r"\b(?:hand|machine)[- ]wash(?:able)?\b",
                    r"\b(?:dry|wet)[- ]clean(?:ing)? only\b",
                    r"\bwipe (?:clean|down|dry|with a (?:damp|dry|soft|clean))\b",
                ),
            ),
        ),
        Matcher(
            "ingredients",
            labels=(
                "ingredients",
                "ingredient",
                "ingredient list",
                "full ingredients",
                "key ingredients",
                "inci",
                "active ingredient",
                "inactive ingredient",
                "other ingredients",
                "excipients",
                "composition",
            ),
            patterns=(
                INGREDIENTS,
                *_compile(
                    r"\beach (?:tablet|capsule|caplet|softgel|dose|sachet|5\s?ml|ml|spray|lozenge|gummy|serving|scoop"
                    r"|pouch|bar) (?:contains|provides)\b",
                    r"\balso contains\b",
                    r"\b(?:aqua|water)\s?\((?:water|aqua|eau)\)",
                ),
            ),
            phrases=(*_INGREDIENT_MARKERS, "excipient*"),
        ),
        Matcher(
            "allergens",
            labels=(
                "allergens",
                "allergen",
                "allergen information",
                "allergy information",
                "allergy advice",
                "allergy",
                "allergies",
                "traces",
                "may contain",
                "contains",
                "free from",
                "hypoallergenic",
                "nickel",
                "nickel content",
                "nickel free",
            ),
            patterns=(
                CATALOG_ALLERGENS,
                _FRAGRANCE_ALLERGENS,
                *_compile(
                    r"^contains\b",
                    rf"\b{_ALLERGEN_WORDS}[- ]free\b",
                    rf"\bcontains?\b[^.;:]{{0,60}}?\b{_ALLERGEN_WORDS}\b",
                    r"\b(?:made|produced|packed|manufactured|processed) (?:in|on) (?:a |the same )?(?:facility|factory"
                    r"|kitchen|site|line|equipment)\b",
                ),
            ),
            phrases=(
                *(marker for marker in STATEMENT_MARKERS if marker not in _INGREDIENT_MARKERS),
                "allerg*",
                "hypoallergenic",
                "nickel*",
                "free from",
            ),
        ),
        Matcher(
            "warnings",
            labels=(
                "warning",
                "caution",
                "precaution",
                "safety information",
                "safety warning",
                "important safety information",
                "side effects",
                "side effect",
                "contraindications",
                "contraindication",
                "interactions",
                "drug interactions",
                "choking hazard",
                "prop 65",
                "proposition 65",
            ),
            patterns=(
                WARNINGS,
                *_compile(
                    r"\bif (?:irritation|redness|rash|pain|swelling|symptoms|discomfort) (?:occurs?|develops?|"
                    r"persists?|worsens?)\b",
                    r"\bnot for (?:children|babies|infants|kids)\b",
                ),
            ),
            phrases=(
                *WARNING_PHRASES,
                "side effect*",
                "may cause",
                "adverse",
                "discontinue use",
                "out of reach of children",
                "out of the reach of children",
                "choking hazard",
                "small parts",
                "external use only",
                "avoid contact",
                "pregnan*",
                "breastfeed*",
                "breast-feed*",
                "breast feeding",
                "adult supervision",
                "flammable",
                "proposition 65",
                "prop 65",
            ),
        ),
        Matcher(
            "dosage",
            labels=(
                "dosage",
                "dose",
                "directions",
                "directions for use",
                "how to use",
                "how to take",
                "how to apply",
                "usage",
                "recommended use",
                "recommended dose",
                "recommended dosage",
                "recommended intake",
                "recommended daily intake",
                "suggested use",
                "suggested dose",
                "serving size",
                "feeding guide",
                "feeding guidelines",
                "feeding instructions",
                "feeding",
            ),
            patterns=(
                DIRECTIONS,
                *_compile(
                    r"\btake (?:\d|one|two|three|four|half)\b",
                    r"\btake (?:a|one) (?:tablet|capsule|caplet|softgel|gummy|sachet|dose|spoonful|teaspoon)\b",
                    rf"\b(?:\d+(?:[.,]\d+)?|one|two|three|four|half)(?:\s?(?:-|to)\s?(?:\d+|two|three|four))?\s?"
                    rf"{_DOSE_UNIT}\b[^.;]{{0,40}}\b(?:once|twice|three times|\d times|every|a day|per day|daily"
                    rf"|each day|a week|per week|with water|with food|with a meal|before|after|morning|evening"
                    rf"|bedtime)\b",
                    r"\b(?:once|twice|three times|four times|\d+ times) (?:a|per|each) (?:day|week|night)\b",
                    r"\bevery \d+(?:\s?(?:-|to)\s?\d+)? (?:hours?|hrs?)\b",
                    r"\b(?:adults?|children|kids|teenagers|teens|elderly|infants?|babies)(?: and children)?"
                    r"(?: aged)? (?:over|under|above|below|from|\d)",
                    r"\bdo not (?:exceed|take more than)\b",
                    r"\bmaximum (?:daily )?(?:dose|dosage|\d)",
                    r"\bapply (?:once|twice|daily|morning|evening|nightly|to (?:clean|dry|damp|wet|the)|a (?:small|pea"
                    r"|thin|generous|few)|liberally|evenly|generously|gently|\d)\b",
                    r"\bfeed (?:\d|up to|between|your)\b",
                    r"\b\d+(?:[.,]\d+)?\s?(?:g|grams?|cups?|ml|scoops?|tins?|pouches?) (?:per|a|each) (?:day|meal|feed"
                    r"|kg)\b",
                ),
            ),
        ),
        Matcher(
            "compatibility",
            labels=(
                "compatibility",
                "compatible",
                "compatible with",
                "compatible models",
                "compatible devices",
                "compatible model",
                "brand compatibility",
                "model compatibility",
                "system compatibility",
                "fitment",
                "vehicle fitment",
                "fits",
                "fits models",
                "for use with",
                "works with",
                "car model",
                "motorcycle model",
                "vehicle",
                "hob compatibility",
                "hob type",
                "suitable hobs",
                "cooktop compatibility",
                "oven safe",
                "induction",
                "battery platform",
                "battery system",
                "replacement for",
            ),
            patterns=(
                COMPATIBLE,
                *_compile(
                    r"\bworks with\b",
                    r"\b(?:in)?compatible\b",
                    r"\bdesigned to fit\b",
                    r"\bfits? (?:all |most |the |any |standard )?(?:[a-z0-9-]+ ){0,3}?(?:models?|handles?|devices?"
                    r"|bikes?|wheels?|frames?|vehicles?|cars?|phones?|tablets?|laptops?|consoles?|printers?|machines?"
                    r"|toothbrush(?:es)?|razors?|shavers?|strollers?|prams?|car seats?|bases?)\b",
                    r"\bfits (?:19|20)\d{2}\b",
                    r"\b(?:induction|gas|electric|ceramic|halogen|all) (?:hobs?|cooktops?|stoves?|stovetops?|hob types?"
                    r"|heat sources?)\b",
                    r"\binduction[- ](?:compatible|ready|safe|suitable|friendly)\b",
                    r"\boven[- ](?:safe|proof)\b",
                    r"\breplacement (?:heads?|brush heads?|parts?|cartridges?|filters?|refills?|blades?|pads?"
                    r"|batteries|bulbs?) for\b",
                    r"\bfor (?:iphone|ipad|samsung|galaxy|pixel|macbook|playstation|xbox|nintendo|kindle)\b",
                    r"\b(?:supports?|supported) (?:ios|android|windows|macos|linux|alexa|google assistant|siri"
                    r"|bluetooth|aptx|aac|ldac)\b",
                    r"\badapters? (?:for|needed|required|included)\b",
                ),
            ),
        ),
        Matcher(
            "dimensions",
            labels=(
                "dimensions",
                "dimension",
                "measurements",
                "product dimensions",
                "package dimensions",
                "packaged dimensions",
                "assembled dimensions",
                "overall dimensions",
                "outer dimensions",
                "external dimensions",
                "internal dimensions",
                "inner dimensions",
                "folded dimensions",
                "unfolded dimensions",
                "folded size",
                "footprint",
                "width",
                "depth",
                "height",
                "length",
                "diameter",
                "thickness",
                "seat height",
                "seat depth",
                "seat width",
                "l x w x h",
                "w x d x h",
                "size (l x w x h)",
            ),
            patterns=(
                DIMENSIONS,
                *_compile(
                    rf"\b\d+(?:[.,]\d+)?\s?{_LENGTH_UNIT}\s?(?:wide|high|tall|deep|long|thick|across|in (?:width|height"
                    rf"|depth|length|diameter))\b",
                    rf"\b(?:width|depth|height|length|diameter|thickness|footprint)\s*(?:of|is|:|=)?\s*(?:approx\.?"
                    rf"|approximately|about)?\s*\d+(?:[.,]\d+)?\s?{_LENGTH_UNIT}",
                    r"\bmeasures? (?:approximately |approx\.? |about |roughly )?\d",
                    r"\bfolds? (?:down |flat )?to \d",
                ),
            ),
        ),
        Matcher(
            "weight",
            labels=(
                "weight",
                "item weight",
                "product weight",
                "net weight",
                "overall product weight",
                "shipping weight",
                "package weight",
                "unit weight",
                "gross weight",
                "weighs",
            ),
            patterns=_compile(
                r"\bweighs? (?:only |just |about |around |approximately |approx\.? |roughly |less than |under |over "
                r"|in at )?\d",
                r"(?<!max )(?<!user )(?<!body )(?<!load )(?<!child )(?<!maximum )\b(?:item |product |net |total "
                r"|overall |unit )?weight\s*(?:of|is|:|=)?\s*(?:only |just |about |approx\.? |approximately )?\d+"
                r"(?:[.,]\d+)?\s?(?:kg|g|grams?|lbs?|pounds?|oz|ounces?)\b",
                r"\b\d+(?:[.,]\d+)?\s?(?:kg|lbs?|pounds?) (?:in weight|heavy)\b",
            ),
        ),
        Matcher(
            "pack_size",
            labels=(
                "pack size",
                "pack",
                "pack of",
                "pack count",
                "count",
                "piece count",
                "pieces",
                "number of pieces",
                "quantity",
                "net quantity",
                "net contents",
                "net content",
                "net weight",
                "net volume",
                "volume",
                "units",
                "unit count",
            ),
            patterns=_compile(
                r"\b(?:pack|box|bottle|tub|bag|jar|tin|case|carton|set|tube|pouch|packet|roll) of \d+\b",
                rf"\b\d+\s?(?:x\s?)?{_COUNT_UNIT}\s(?:per|in (?:each|a|the|one|every)|included|supplied)\b",
                r"\bnet (?:wt|weight|quantity|qty|contents?|vol(?:ume)?)\.?\s*:?\s*\d",
                r"\b\d+[- ](?:pack|count|ct|pk)\b",
                rf"\b(?:contains|includes) \d+\s?(?:x\s?\d+\s?(?:g|ml)\s)?{_COUNT_UNIT}\b",
            ),
        ),
    )
}
"""Every catalog detail a hypothesis may name, in the order data/categories/_fields.yaml lists them."""


def _label_table() -> tuple[tuple[str, re.Pattern[str], frozenset[str]], ...]:
    """Each label phrase, its pattern and the details it names, longest first; NOT_DETAILS name none. The phrase
    itself is kept to skip the pattern cheaply when the phrase is nowhere in the text."""
    named: dict[str, set[str]] = {}
    for matcher in MATCHERS.values():
        for label in matcher.labels:
            named.setdefault(fold(label), set()).add(matcher.name)
    for phrase in NOT_DETAILS:
        named.setdefault(fold(phrase), set())
    ordered = sorted(named, key=lambda phrase: (-len(phrase), phrase))
    return tuple(
        (phrase, re.compile(rf"(?<![a-z0-9]){re.escape(phrase)}(?:e?s)?(?![a-z0-9])"), frozenset(named[phrase]))
        for phrase in ordered
    )


_LABELS = _label_table()
_BULLET = re.compile(r"^\s*(?:[-*+•·‣◦▪●○►✓✔]|\d{1,2}[.)])\s+")
_MARKUP = re.compile(r"^\s*(?:#{1,6}|>)\s*|[*_`]+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])")
_KEY_PUNCTUATION = re.compile(r"[_./:\-]+")
_LABEL_END = re.compile(r"\s*(?::|\s[-–—]\s|\|)")
_WORDS = re.compile(r"[a-z0-9']+")


def clean(text: str) -> str:
    """A sentence or a line as the matchers read it: without a list marker or markdown markup, folded."""
    return fold(_MARKUP.sub(" ", _BULLET.sub("", text)))


def named(text: str) -> frozenset[str]:
    """Every detail a key, a label or a heading names: "careInstructions" and "Care instructions" name care, "Fabric
    & care" names material and care, "Weight capacity" names none."""
    key = _key(text)
    found: list[tuple[int, int, frozenset[str]]] = []
    for phrase, pattern, details in _LABELS:
        if phrase in key:
            found += [(m.start(), m.end(), details) for m in pattern.finditer(key)]
    kept = [
        (start, end, details)
        for start, end, details in found
        if not any(s <= start and end <= e and e - s > end - start for s, e, _ in found)
    ]
    return frozenset().union(*(details for _, _, details in kept))


def only_labels(text: str) -> bool:
    """Whether a heading or a label holds nothing but names of details and filler words ("Size chart (cm)")."""
    rest = _key(text)
    for phrase, pattern, _ in _LABELS:
        if phrase in rest:
            rest = pattern.sub(" ", rest)
    return all(word in FILLER for word in _WORDS.findall(rest))


def _key(text: str) -> str:
    """A key or a label as words: "careInstructions", "care_instructions" and "Care-Instructions" all read "care
    instructions"."""
    return fold(_KEY_PUNCTUATION.sub(" ", _CAMEL.sub(" ", text)))


def heading(text: str) -> frozenset[str] | None:
    """The details a heading names, when the text is nothing but a heading ("Ingredients", "Fabric & care:"), hidden
    or not; None when the text is not a heading of any detail."""
    core = clean(text).rstrip(" :.-–—|")
    if not core or len(core) > HEADING_CHARS or not only_labels(core):
        return None
    return named(core) or None


def _labelled(core: str) -> frozenset[str]:
    """The details a label at the start of a cleaned sentence names: "Allergens: may contain nuts", "Care - machine
    wash". Empty when it opens with no label, or with one that is not only names of details ("Model wears: M")."""
    end = _LABEL_END.search(core, 0, LABEL_CHARS)
    if end is None or end.start() == 0:
        return frozenset()
    head = core[: end.start()]
    return named(head) if only_labels(head) else frozenset()


class HiddenSet:
    """The details one detail_hidden experiment hides, and what states them."""

    def __init__(self, details: Iterable[str]) -> None:
        names = tuple(dict.fromkeys(details))
        unknown = sorted(name for name in names if name not in MATCHERS)
        if unknown or not names:
            raise ValidationError(
                f"detail_hidden names unknown details {unknown}" if unknown else "detail_hidden names no details",
                hint=f"name details from data/categories/_fields.yaml: {', '.join(MATCHERS)}",
            )
        self.details = frozenset(names)
        self._matchers = tuple(MATCHERS[name] for name in MATCHERS if name in self.details)

    def keyed(self, key: str) -> frozenset[str]:
        """The hidden details an attribute, metafield or detail key names."""
        return named(key) & self.details

    def stated(self, text: str) -> frozenset[str]:
        """The hidden details a sentence or a line states, by the label it opens with or by what it says."""
        core = clean(text)
        found = set(_labelled(core) & self.details)
        found |= {m.name for m in self._matchers if m.name not in found and m.states(core, text)}
        return frozenset(found)

    def chart(self, grid: Grid) -> bool:
        """Whether a table is a size chart, read as the size chart reader reads one, while size_chart is hidden."""
        if "size_chart" not in self.details:
            return False
        return read_chart(Product(id="table", title="", tables=(grid,))) is not None
