"""What an agent's message offers the shopper to choose from, and how each option stands against the hidden spec.

Why (D60). The "deliberate" persona register asks the agent to lay out the options before the shopper decides. When
the agent did, its message often carried no question mark ("Here are the four sizes: ..."), and the shopper answered
every message that asks nothing with "Okay, thanks.", so it never chose; a list closed by "Which would you like?"
got "I'm not sure about that" instead. The agent waited or ended, and the judge charged it with giving up. On
2026-10-13 that was 19 percent of the day's sessions on every surface, the sandbox included. A person shown options
picks one, says none fits, or agrees to the one recommended, so the shopper now does too, by code.

Two parts, both pure functions of the message and the spec:

- `read_offer` finds the options a message puts to the shopper (list items, table rows, "A or B" after a colon) and
  a sentence recommending one ("my pick is", "shall I add"), and says whether the message awaits a choice: it has
  a question mark or a request to pick, presents its list as options ("Here are the sizes:"), or lists priced
  options. Lines that only label a fact ("Price: $12", "Returns: 30 days") are not options, so a cart summary is
  never read as a list to choose from, and bullets describing one product are not alternatives.
- A cut message is not read at all: the shopper's caller says when the token limit cut the agent's turn.
- `judge_options` says, per option and per need, whether the option's own words meet the need, miss it, or say
  nothing about it. Only needs a reader can check in the words are read: the product named, a variant value, a
  brand, a mentioned term, a price within the budget, the cheapest of the priced options. A need the words cannot
  settle (an allergen, a size chart fit) is left unknown, never guessed from the store's data: a shopper who keeps a
  need to themselves and is offered options that ignore it may well pick a wrong one, and that is the agent's miss.

Which needs the shopper may use and what it may say is the shopper's business (carttrace.agents.shopper), not this module's.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Final, Literal

from pydantic import JsonValue

from carttrace.agents.keywords import asked_fields, names_value
from carttrace.agents.spec import Predicate
from carttrace.agents.text import cells, collapse, fold, has_phrase, separator_row, words
from carttrace.core.money import to_minor

Verdict = Literal["meets", "misses", "unknown"]
CHECKED_OPS: Final = frozenset({"kind", "is", "one_of", "prefer", "brand", "eq", "mentions", "max_price", "cheapest"})
"""Needs an option's words can settle. Every other op is left unknown."""
NAMING_OPS: Final = frozenset({"is", "one_of", "prefer", "brand", "eq", "mentions"})
"""Needs met by naming a value: the option that names it says so in the shopper's reply."""
MAX_OPTIONS: Final = 12
MAX_LABEL_WORDS: Final = 12

_ITEM = re.compile(r"^(?P<indent>\s*)(?P<marker>\d{1,2}[.)]|[-*•+])\s+(?P<text>\S.*)$")
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_BOLD = re.compile(r"\*\*(?P<a>[^*]+?)\*\*|__(?P<b>[^_]+?)__")
_MARKUP = re.compile(r"[*_`#>]+")
_HEADING = re.compile(r"^\s*(?:#{1,4}\s*)?(?:\*\*[^*]+\*\*|__[^_]+__)\s*(?:\([^)]{0,60}\)|[-–—:]\s*[^.!?]{0,40})?\s*$")
"""A line that is one bold name, at most with a bracket or a short tail ("**1. Meadowfoam Oil Balm** (`slug`)",
"**Collagen** – full size"): a heading the detail bullets below it belong to."""
_FREE = re.compile(r"(?<![-\w])free\b(?!\s+(?:shipping|delivery|returns?|gift|of|from|for|to|with|-with))", re.I)
"""An option said to cost nothing ("free sample"), as opposed to free shipping or "feel free to"."""
_FACT_LABEL = re.compile(r"^\s*(?P<label>[A-Za-z][A-Za-z /&'-]{0,30}?)\s*:\s+\S")
FACT_LABELS: Final = frozenset(
    {
        "address", "availability", "cart", "cart id", "checkout", "delivery", "email", "id", "in stock", "note",
        "notes", "one caveat", "one honest caveat", "order", "phone", "policy", "price", "prices", "qty", "quantity",
        "rating", "ratings", "return policy", "returns", "reviews", "shipping", "sku", "status", "stock", "subtotal",
        "tax", "taxes", "total", "variant id", "why", "why it wins", "caveat", "heads up", "note on stock",
    }
)  # fmt: skip
"""A line that starts with one of these and a colon states a fact; it is not an option to choose."""
ATTRIBUTE_ROWS: Final = frozenset(
    {
        "size", "scent", "format", "type", "colour", "color", "burn time", "weight", "volume", "ingredients",
        "best for", "what it does", "what's inside", "contents", "includes", "rating", "reviews", "value", "per unit",
    }
)  # fmt: skip
"""First cells of a comparison table's rows that name an attribute, so its columns are the options."""
_PICK = re.compile(
    r"\b(?:let me know which|tell me which|say which|just say|which (?:one|of|would|do|should|sounds|works)|"
    r"whichever|pick (?:one|a|the|which|from)|choose (?:one|a|the|which|between|from)|your (?:pick|choice)|"
    r"when you(?:'ve| have)? (?:decided|picked|chosen)|once you (?:decide|pick|choose)|"
    r"(?:would|do) you (?:like|want|prefer)|you(?:'d| would) (?:like|prefer))\b",
    re.IGNORECASE,
)
_RECOMMEND = re.compile(
    r"\b(?:i(?:'d| would)? (?:recommend|suggest|go with)|my (?:top |best |clear )?(?:pick|recommendation|choice|"
    r"suggestion|vote)|(?:the )?best (?:fit|match|choice|option|pick|bet) (?:for you )?(?:is|would be|here)|"
    r"(?:shall|should) i (?:add|put|get|go ahead|order)|(?:do you )?want me to (?:add|put|get|go ahead|order)|"
    r"would you like me to (?:add|put|get|go ahead|order)|i(?:'ll| will| can) (?:add|put) (?:it|this|that|the|one))\b",
    re.IGNORECASE,
)
_PRESENTS = re.compile(
    r"\b(?:here (?:are|is)|here's|(?:your|the) (?:options|choices)|laid out|to choose from|comes? in|available in)\b",
    re.IGNORECASE,
)
"""Words that present a list as options to weigh, as the deliberate register asks ("Here are the sizes:")."""
_ASKS_TO_PICK = re.compile(r"\b(?:which|one of (?:these|them|those)|any of (?:these|them|those))\b", re.I)
_INLINE = re.compile(r"(?:\bbetween\b|\beither\b|:)\s*(?P<rest>[^:]+)$", re.IGNORECASE)
_OR_COMPARES = re.compile(r"\bor (?:less|more|fewer|lower|higher|under|over|so|not|otherwise|else)\b", re.I)
"""An "or" that compares ("$4 or less") or hedges ("or not"): no options there."""
_INLINE_SPLIT = re.compile(r",\s*(?:or\s+)?|\s+or\s+", re.IGNORECASE)
_INLINE_CUE = re.compile(r"^.*?\b(?:would you like|do you want|do you prefer|you'd like|you prefer|prefer)\s+", re.I)
_NUMBER = r"(?P<num>\d{1,3}(?:,\d{3})+(?:\.\d+)?|\d+(?:\.\d+)?)"
_SYMBOL_PRICE = re.compile(rf"(?P<sym>[$£€¥₹])\s?{_NUMBER}")
_CODE_PRICE = re.compile(
    rf"{_NUMBER}\s?(?P<code>[A-Z]{{3}})\b|\b(?P<code2>[A-Z]{{3}})\s?{_NUMBER.replace('num', 'num2')}"
)
SYMBOL_CURRENCIES: Final[Mapping[str, frozenset[str]]] = {
    "$": frozenset({"USD", "CAD", "AUD", "NZD", "SGD", "HKD", "MXN"}),
    "£": frozenset({"GBP"}),
    "€": frozenset({"EUR"}),
    "¥": frozenset({"JPY", "CNY"}),
    "₹": frozenset({"INR"}),
}
_LETTERS = re.compile(r"[A-Za-z]{2}")
_UNAVAILABLE = re.compile(
    r"\b(?:out of stock|sold out|unavailable|not available|no longer|discontinued|isn't (?:in|available|sold|carried)|"
    r"is not (?:in|available|sold|carried)|not in (?:the|their|its) (?:catalog|store|range)|(?:doesn't|does not) "
    r"(?:carry|have|sell|stock|appear|exist|show up)|(?:isn't|is not|not) listed|(?:can't|cannot|couldn't|could not) "
    r"find)\b",
    re.IGNORECASE,
)
"""Words that say a product cannot be had: the options that follow are substitutes, not it."""
_NUMBER_LEAD = re.compile(r"^\d{1,2}[.)]\s+")
FULFILLMENT_WORDS: Final[Mapping[str, tuple[str, ...]]] = {
    "standard": ("standard", "ground", "regular shipping", "economy", "free shipping"),
    "express": ("express", "expedited", "next day", "next-day", "overnight", "priority"),
    "pickup": ("pickup", "pick up", "picked up", "pick-up", "collect", "in store", "in-store"),
}
"""The words an option uses for each way of delivery the buyer block can hold."""
_ARTICLES = ("a ", "an ", "some ", "the ")
_TITLE_STOP: Final = frozenset({"the", "and", "for", "with", "of", "a", "an", "in", "to", "by", "set"})


@dataclass(frozen=True, slots=True)
class Offer:
    """What a message puts to the shopper: the options, and the one it recommends."""

    options: tuple[str, ...]
    """Each option's words, as the agent wrote them (markup kept, so a bold name can label it)."""
    numbered: bool
    """The options were a numbered list, so the shopper can say which by number."""
    recommended: str | None
    """The sentence that recommends one option, or asks to go ahead with one; None when none does. With options
    listed, only a sentence that names one of them recommends it."""
    recommended_option: int | None
    """The listed option the recommending sentence names, counted from 0; None without one."""
    asked: str
    """The words that put the choice: the options, the lines that introduce them ("It comes in these sizes:") and
    the sentences asking for it. A need whose field these name is one the agent asked about, as a question naming it
    would."""


def read_offer(message: str) -> Offer | None:
    """The choice `message` awaits, or None when it offers nothing to choose."""
    text = message or ""
    options, numbered = _listed(text)
    sentences = [s.strip() for s in _SENTENCES.split(text) if s.strip()]
    if not options:
        options = _inline(sentences)
    cue = "?" in text or _PICK.search(text) is not None or _PRESENTS.search(text) is not None
    priced = sum(1 for o in options if _has_price(o))
    if not (len(options) >= 2 and (cue or 2 * priced >= len(options))):
        options = []
    options = options[:MAX_OPTIONS]
    recommended, index = _recommendation(sentences, options)
    if not options and recommended is None:
        return None
    asking = [s for s in sentences if "?" in s or s.endswith(":") or _PICK.search(s) or _RECOMMEND.search(s)]
    asked = " ".join(dict.fromkeys([*options, *asking]))
    return Offer(tuple(options), numbered and bool(options), recommended, index, asked)


def _recommendation(sentences: Sequence[str], options: Sequence[str]) -> tuple[str | None, int | None]:
    """The sentence recommending an option. With options listed it must name one of them ("my pick is the
    **Travel**"); a sentence that asks to add "one of these" asks for a pick instead. Without options, the first
    sentence that recommends or asks to go ahead ("Shall I add it to your cart?")."""
    for sentence in sentences:
        if not _RECOMMEND.search(sentence) or _ITEM.match(sentence):
            continue
        if not options:
            if _ASKS_TO_PICK.search(sentence):
                continue
            return sentence, None
        index = recommended_index(sentence, options)
        if index is not None:
            return sentence, index
    return None, None


def recommended_index(sentence: str, options: Sequence[str]) -> int | None:
    """The option `sentence` names, by a bold name it shares or the option's label. None when it names none, or
    several: a sentence naming two options weighs them and recommends neither."""
    found: list[int] = []
    bold = [fold(n) for n in bold_names(sentence) if n]
    for i, option in enumerate(options):
        label = label_of(option)
        if any(n == fold(label) or has_phrase(option, n) for n in bold) or (
            label and len(label.split()) >= 2 and has_phrase(sentence, label)
        ):
            found.append(i)
    return found[0] if len(found) == 1 else None


def _listed(text: str) -> tuple[list[str], bool]:
    """List items (numbered first, when there are any; deeper items join the one above) and table rows. When the
    items only describe (none opens with a bold name), two bold headings with items under them are the options."""
    items: list[tuple[int, bool, str]] = []
    rows: list[str] = []
    block: list[list[str]] = []
    head: list[str] = []
    headed: list[list[str]] = []
    """Bold heading lines, each with the items under it: options written as headings with detail bullets."""
    lines = text.splitlines()
    for index, line in enumerate(lines):
        if _HEADING.match(line):
            headed.append([line.strip()])
            continue
        found = cells(line) if "|" in line else []
        if found and separator_row(line.strip()):
            continue
        if found:
            if index + 1 < len(lines) and separator_row(lines[index + 1].strip()):
                head = found
            else:
                block.append(found)
            continue
        if block:
            rows += _table_options(head, block)
        block, head = [], []
        match = _ITEM.match(line)
        if match:
            items.append((len(match.group("indent").expandtabs(4)), match.group("marker")[0].isdigit(), match["text"]))
            if headed:
                headed[-1].append(match["text"])
    if block:
        rows += _table_options(head, block)
    numbered = any(n for _, n, _ in items)
    kept = [o for o in _top_items(items, numbered) if _an_option(o)]
    under = [" ".join(group) for group in headed if len(group) > 1]
    if len(under) >= 2 and not any(o.lstrip().startswith(("**", "__")) for o in kept):
        kept = under  # the items under the headings describe them (a price among the details names no option)
    elif kept and not _alternatives(kept):
        kept = []
    return kept + [t for t in rows if _an_option(t)], numbered


def _top_items(items: Sequence[tuple[int, bool, str]], numbered: bool) -> list[str]:
    """The top-level items, numbered ones first when there are any; a deeper item, or a plain bullet after a numbered
    one, joins the item above, while a bullet that names or prices a thing is an item of its own."""
    out: list[str] = []
    if not items:
        return out
    top = min(i for i, n, _ in items if n == numbered or not numbered)
    for indent, is_number, body in items:
        if indent <= top and (is_number == numbered or not numbered or names_a_thing(body)):
            out.append(body)
        elif out:
            out[-1] = f"{out[-1]} {body}"
    return out


def _table_options(head: Sequence[str], body: Sequence[Sequence[str]]) -> list[str]:
    """A table's options: its rows, or its columns when the rows are attributes ("Price", "Availability") and the
    header names the products, as an agent lays out a side-by-side comparison."""
    firsts = [fold(_MARKUP.sub("", row[0])).strip(" :") for row in body if row]
    attributes = sum(1 for first in firsts if first in FACT_LABELS or first in ATTRIBUTE_ROWS)
    if len(head) >= 3 and 2 * attributes >= len(firsts):
        columns = []
        for j in range(1, len(head)):
            values = [row[j] for row in body if j < len(row) and row[j]]
            columns.append(" | ".join([head[j], *values]))
        return [c for c in columns if c.strip(" |")]
    return [" | ".join(c for c in row if c) for row in body]


def names_a_thing(item: str) -> bool:
    """An item that names a thing to choose (it opens with a bold name) or prices one, as opposed to a course of
    action ("Walk away", "Have me look more broadly")."""
    return item.lstrip().startswith(("**", "__")) or _has_price(item) or _FREE.search(_MARKUP.sub("", item)) is not None


def _alternatives(items: Sequence[str]) -> bool:
    """List items that read as things to choose between: at least two carry a bold name or a price, or every one
    is short ("Travel", "8.5 oz"). Bullets that describe one product ("Rated 4.6 from 391 reviews", "A serum that
    pairs retinol with antioxidants") are not options."""
    marked = sum(1 for item in items if _BOLD.search(item) or _has_price(item))
    return marked >= 2 or all(len(_MARKUP.sub("", item).split()) <= 6 for item in items)


def _an_option(text: str) -> bool:
    plain = _MARKUP.sub("", text)
    label = _FACT_LABEL.match(plain)
    if label is not None and fold(label.group("label")) in FACT_LABELS:
        return False
    return len(words(plain)) >= 1


def _inline(sentences: Sequence[str]) -> list[str]:
    """Options written in one sentence: "two formats: Travel Sticks or Jar", "the Travel or the 8.5 oz?"."""
    for sentence in sentences:
        if " or " not in sentence.lower() or _OR_COMPARES.search(sentence):
            continue
        match = _INLINE.search(sentence)
        rest = match.group("rest") if match else (_INLINE_CUE.sub("", sentence) if "?" in sentence else "")
        parts = [p.strip(" .?!") for p in _INLINE_SPLIT.split(rest.strip(" .?!")) if p.strip(" .?!")]
        limit = 8 if match else 4
        if 2 <= len(parts) <= MAX_OPTIONS and all(len(p.split()) <= limit for p in parts):
            return parts
    return []


def prices(text: str, currency: str | None) -> list[int]:
    """Every price in `text` that can be in `currency`, in minor units and in the order written: "$42" in USD is 4200,
    "€3" in USD is none. An option's own price is the first: later ones are "was $68" or "$2 over your cap"."""
    if not currency:
        return []
    code = currency.upper()
    found: list[tuple[int, list[int]]] = []
    for match in _SYMBOL_PRICE.finditer(text):
        if code in SYMBOL_CURRENCIES.get(match.group("sym"), frozenset()):
            found.append((match.start(), _minor(match.group("num"), code)))
    for match in _CODE_PRICE.finditer(text):
        if code in (match.group("code"), match.group("code2")):
            found.append((match.start(), _minor(match.group("num") or match.group("num2"), code)))
    return [value for _, values in sorted(found) for value in values]


def option_prices(text: str, currency: str | None) -> list[int]:
    """The prices an option states, or nothing (0) when it says it is free and states no price."""
    found = prices(text, currency)
    if found or not currency:
        return found
    return [0] if _FREE.search(_MARKUP.sub("", text)) else []


def _minor(number: str, currency: str) -> list[int]:
    value = to_minor(number.replace(",", ""), currency)
    return [] if value is None or value < 0 else [value]


def _has_price(text: str) -> bool:
    return _SYMBOL_PRICE.search(text) is not None or _CODE_PRICE.search(text) is not None


def fulfillment_need(kind: str, source: str, say: str) -> Predicate:
    """The buyer's way of delivery as a need an offer of delivery options meets by naming it. A default one is the
    shopper's own answer when asked ("the standard option is fine"), so it is weighed as one kept for asking."""
    terms: list[JsonValue] = list(FULFILLMENT_WORDS.get(kind, (kind,)))
    kept = "stated" if source == "stated" else "on_ask"
    return Predicate(field="fulfillment", op="mentions", value=terms, source=kept, say=say)


def delivery_offer(offer: Offer) -> bool:
    """The offer is of ways to deliver: at least two of its options name one."""
    terms = [t for kind in FULFILLMENT_WORDS.values() for t in kind]
    return sum(1 for option in offer.options if any(has_phrase(option, t) for t in terms)) >= 2


def label_of(option: str) -> str:
    """How the shopper names an option back: its bold name when it has one, else its words up to the first dash,
    colon or bracket, at most twelve words, markup removed. A bold price names nothing, so it is passed over."""
    tidy = " .,;:\"'-–—"
    named = [n for n in bold_names(option) if _LETTERS.search(n)]
    base = named[0] if named else re.split(r"\s+[-–—|]\s+|:\s|\s\(", option)[0]
    plain = _NUMBER_LEAD.sub("", collapse(_MARKUP.sub("", base))).strip(tidy)
    if not _LETTERS.search(plain):  # a bare price or number names nothing: say the option's words instead
        plain = _NUMBER_LEAD.sub("", collapse(_MARKUP.sub("", option).replace("|", ","))).strip(tidy)
    return " ".join(plain.split()[:MAX_LABEL_WORDS]).strip(tidy)


def bold_names(text: str) -> list[str]:
    return [collapse(m.group("a") or m.group("b") or "") for m in _BOLD.finditer(text)]


# -- how each option stands against the needs ----------------------------------------------------------------------


def judge_options(
    options: Sequence[str],
    needs: Sequence[Predicate],
    *,
    context: str,
    currency: str | None,
    quantity: int,
    message: str = "",
) -> list[dict[str, Verdict]]:
    """For each option, each need's verdict from the option's words. `context` is the offer's own words
    (`Offer.asked`: the options, the lines that introduce them, the sentences asking): a value named nowhere in them,
    for a field the offer is about, is missed by every option; one named only by some options is missed by the
    rest. A product wanted by name that no option names is missed by every option when `message` (the whole
    message) says it is unavailable, or when it never names it and offers other named products; otherwise the options
    may be its variants and say nothing. A kind of product one option names is missed by the options that do not."""
    out: list[dict[str, Verdict]] = [{} for _ in options]
    whole = message or context
    products = sum(1 for option in options if _BOLD.search(option)) >= 2
    for need in needs:
        if need.op not in CHECKED_OPS or need.op == "cheapest":
            continue
        column = [_verdict(need, option, currency, quantity) for option in options]
        if need.op in NAMING_OPS:
            column = _relative(need, column, context, whole=whole, products=products)
        elif need.op == "kind" and "meets" in column:
            column = ["meets" if v == "meets" else "misses" for v in column]
        for verdicts, verdict in zip(out, column, strict=True):
            verdicts[need.field] = verdict
    cheapest = next((n for n in needs if n.op == "cheapest"), None)
    if cheapest is not None:
        _cheapest(cheapest, options, out, currency)
    return out


def judge_alone(text: str, needs: Sequence[Predicate], *, currency: str | None, quantity: int) -> dict[str, Verdict]:
    """A recommended option on its own: only what its words say, with no other option to compare against."""
    out: dict[str, Verdict] = {}
    for need in needs:
        if need.op in CHECKED_OPS and need.op != "cheapest":
            verdict = _verdict(need, text, currency, quantity)
            out[need.field] = "unknown" if verdict == "misses" and need.op in NAMING_OPS else verdict
    return out


def _verdict(need: Predicate, option: str, currency: str | None, quantity: int) -> Verdict:
    if need.op == "max_price":
        return _within(need, option, currency, quantity)
    if need.op == "kind":
        return "meets" if _names_noun(option, need.value) else "unknown"
    return "meets" if names(option, need) else "unknown"


def names(text: str, need: Predicate) -> bool:
    """`text` names the value `need` wants: the product's title (most of its words), a variant value, a term."""
    value = need.value
    if need.op in ("is", "prefer", "one_of"):
        titles = value if isinstance(value, list) else [value]
        return any(isinstance(t, str) and _names_title(text, t) for t in titles)
    terms = [str(v) for v in value] if isinstance(value, list) else [] if value is None else [str(value)]
    terms = [t for t in terms if t.strip() and not isinstance(value, dict)]
    return names_value(text, terms)


def _relative(need: Predicate, column: list[Verdict], context: str, *, whole: str, products: bool) -> list[Verdict]:
    """A naming need is missed by the options that do not name its value, once another option does. When no option
    names it: a product wanted by name is missed by every option when the message says it is unavailable ("the
    bergamot sample is out of stock"), or names it nowhere while offering other named products; another value is
    missed by every option when the offer is about its field and names it nowhere."""
    if "meets" in column:
        if need.op == "prefer":
            return column  # a substitute may be acceptable; the shopper does not rule one out by name
        return ["meets" if v == "meets" else "misses" for v in column]
    if need.op == "prefer":
        return column
    if need.op in ("is", "one_of"):
        # by line, not sentence: a title such as "hand+body wash. sample" holds a full stop of its own
        unavailable = any(names(line, need) and _UNAVAILABLE.search(line) for line in whole.splitlines())
        elsewhere = products and not names(whole, need)
        return ["misses" for _ in column] if unavailable or elsewhere else column
    if not names(context, need) and _about(need, context):
        return ["misses" for _ in column]
    return column


def _about(need: Predicate, context: str) -> bool:
    """The message is about the need's field: it names the store's option ("Size") or a word of the field."""
    if need.option and has_phrase(context, need.option):
        return True
    return bool(asked_fields(context, [need.field]))


def _within(need: Predicate, option: str, currency: str | None, quantity: int) -> Verdict:
    value = need.value
    if not isinstance(value, dict):
        return "unknown"
    amount, code = value.get("amount"), value.get("currency") or currency
    if not isinstance(amount, int) or isinstance(amount, bool) or not isinstance(code, str):
        return "unknown"
    found = option_prices(option, code)
    if not found:
        return "unknown"
    return "meets" if found[0] * max(quantity, 1) <= amount else "misses"


def _cheapest(need: Predicate, options: Sequence[str], out: list[dict[str, Verdict]], currency: str | None) -> None:
    priced = {
        i: found[0]
        for i, option in enumerate(options)
        if "misses" not in out[i].values() and (found := option_prices(option, currency))
    }
    for i in range(len(options)):
        out[i][need.field] = "unknown"
    if len(priced) < 2:
        return
    low = min(priced.values())
    for i, price in priced.items():
        out[i][need.field] = "meets" if price == low else "misses"


def _names_noun(text: str, value: object) -> bool:
    if not isinstance(value, str) or not value.strip():
        return False
    noun = fold(value)
    for article in _ARTICLES:
        if noun.startswith(article):
            noun = noun[len(article) :]
    forms = {noun, noun.removesuffix("s"), f"{noun}s", f"{noun}es"}
    return any(form and has_phrase(text, form) for form in forms)


def names_title(text: str, title: str) -> bool:
    """`_names_title` for callers outside this module: the shopper reads which served product an option names."""
    return _names_title(text, title)


def _names_title(text: str, title: str) -> bool:
    """A title's words are in `text`, its first word always: an agent shortens "Supershine Hydrating Shampoo" to
    "Supershine Shampoo", so a title of three words or more may lose one later word, but "Rum hand+body wash sample"
    is not the "Bergamot hand+body wash sample" it shares four words with. A short title must be there whole."""
    if not title.strip():
        return False
    if has_phrase(text, title):
        return True
    wanted = [w for w in words(title) if w not in _TITLE_STOP and len(w) > 1]
    if not wanted:
        return False
    have = set(words(text))
    if wanted[0] not in have:
        return False
    missing = sum(1 for w in wanted if w not in have)
    return missing == 0 if len(wanted) <= 2 else missing <= 1
