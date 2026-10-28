"""Which hidden-spec fields an agent's message asks about, by keyword: the rule step of the shopper.field decision.

Why a rule first. The shopper reveals a field only when asked, so a wrong mapping leaks a preference the agent never
asked for, and an engine call per message would spend a Clef pass on "What size do you wear?". The rule answers
when the words are plain; anything it cannot place goes to the decision engine (data/decisions/shopper.yaml). It
follows the lab's classifier (lab/shopper.py, 2026-09-18), with what the lab learned:

- only text that asks unlocks a field: sentences with a question mark, and requests such as "Let me know your
  size." A statement ("I found it in navy") reveals nothing;
- "pack size" is a question about the pack, not the garment size, and "how many" asks a quantity while "how much"
  asks a budget;
- naming a value counts as asking about its field ("Navy or black?" asks the colour; "M or L?" the size, read as
  capital letters standing alone so a lowercase "m" in "I'm" never counts).

Pure and deterministic: a function of the message and the fields, with no clock, network or file, so it could be
registered as a decision rule in carttrace.decide unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Mapping, Sequence

from carttrace.agents.text import fold, has_phrase

KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "product_type": (
        "what kind of",
        "which kind of",
        "what type of",
        "which type of",
        "what sort of",
        "what are you looking for",
        "what kind are you",
        "which kind are you",
        "what type are you",
        "what sort are you",
    ),
    "brand": ("brand", "brands", "which make", "manufacturer"),
    "product_name": ("which product", "product name", "exact product", "what was it called", "which one did you"),
    "quantity": ("how many", "quantity", "number of units", "how many units", "more than one"),
    "budget": ("budget", "spend", "price range", "price point", "how much", "afford", "max price", "maximum price"),
    "delivery_by": ("deliver*", "arrive", "by when", "when do you need", "deadline", "how soon", "need it by"),
    "color": (
        "colour*",
        "color*",
        "black",
        "white",
        "navy",
        "blue",
        "grey",
        "gray",
        "red",
        "green",
        "beige",
        "brown",
        "pink",
        "purple",
        "yellow",
        "orange",
        "ivory",
        "khaki",
        "olive",
    ),
    "material": ("material*", "fabric*", "made of", "cotton", "linen", "wool", "silk", "leather", "polyester"),
    "size": ("size", "sizes", "sizing", "what size", "xs", "xl", "xxl", "small", "medium", "large"),
    "size_system": ("uk size", "us size", "eu size", "sizing system", "uk or us", "us or uk", "european sizing"),
    "measurements": (
        "measurement*",
        "measure",
        "chest",
        "bust",
        "waist",
        "hips",
        "inseam",
        "inside leg",
        "foot length",
        "how tall",
        "height",
    ),
    "fit": ("fit", "slim", "relaxed", "oversized", "loose", "tight"),
    "length": ("length", "midi", "maxi", "cropped"),
    "width": ("width", "wide", "narrow"),
    "care": ("machine wash*", "washable", "dry clean*", "care instructions"),
    "occasion": ("occasion", "event", "what is it for", "what's it for", "what's the occasion"),
    "recipient": ("who is it for", "who's it for", "for whom", "recipient", "buying for"),
    "gift": ("gift", "gift wrap*", "wrapped"),
    "returnable": ("return*", "exchange*", "refund*"),
    "age": ("how old", "age", "ages", "years old", "months old"),
    "allergens": (
        "allerg*",
        "intoleran*",
        "sensitive to",
        "react to",
        "can't have",
        "cannot have",
        "peanut*",
        "nut",
        "nuts",
        "milk",
        "dairy",
        "lactose",
        "gluten",
        "wheat",
        "soy",
        "egg",
        "eggs",
        "sesame",
        "gelatin*",
        "nickel",
    ),
    "ingredients": ("ingredient*", "formula", "made with"),
    "free_from": ("free from", "paraben*", "sulfate*", "sulphate*", "silicone*"),
    "dietary": ("vegan", "vegetarian", "dietary", "diet", "halal", "kosher", "organic"),
    "skin_type": ("skin type", "dry skin", "oily skin", "combination skin", "sensitive skin", "acne"),
    "hair_type": ("hair type", "curly", "wavy", "straight hair", "fine hair", "thick hair", "colour-treated"),
    "shade": ("shade", "undertone", "skin tone", "complexion"),
    "scent": ("scent*", "fragrance*", "smell*", "unscented", "perfume*"),
    "flavor": ("flavour*", "flavor*", "taste", "roast", "grind"),
    "pack_size": ("pack size", "pack", "packs", "count", "how many tablets", "how many capsules", "bottle size"),
    "subscription": ("subscri*", "repeat delivery", "recurring", "one-off", "one off", "one-time", "autoship"),
    "active_ingredient": ("active ingredient*", "which medicine", "which drug"),
    "strength": ("strength", "mg", "dose", "how strong"),
    "dosage_form": ("tablet*", "capsule*", "liquid", "chewable*", "gumm*", "spray", "drops", "powder", "which form"),
    "symptoms": ("symptom*", "treat*", "relief", "what's wrong", "what is wrong"),
    "conditions": (
        "pregnan*",
        "breastfeed*",
        "medical condition*",
        "health condition*",
        "condition*",
        "medication*",
        "medicine*",
        "taking any",
        "asthma",
        "blood pressure",
        "health issue*",
    ),
    "rx_status": ("prescription", "pharmacy only", "over the counter", "otc"),
    "compatibility": ("compatible", "compatibility", "work with", "which model", "which device"),
    "vehicle": ("vehicle", "car", "make and model", "engine"),
    "part_number": ("part number", "oem", "part no"),
    "dimensions": ("dimension*", "how big", "how wide", "how deep", "how much space", "fit in"),
    "capacity": ("capacity", "litre*", "liter*", "volume", "how much does it hold"),
    "power": ("power", "plug", "voltage", "battery", "cordless", "wattage"),
    "specs": ("spec", "specs", "specification*", "ram", "processor", "resolution"),
    "condition": ("new or used", "refurbished", "second-hand", "secondhand", "open box", "pre-owned"),
    "assembly": ("assembl*",),
    "weight_limit": ("weight limit", "how heavy", "max weight", "maximum weight"),
    "warranty": ("warranty", "guarantee"),
    "metal": ("metal", "gold", "silver", "platinum", "plated"),
    "stone": ("stone", "diamond", "gem*"),
    "personalization": ("engrav*", "personalis*", "personaliz*", "monogram*"),
    "room": ("which room", "what room", "room"),
    "plant_zone": ("hardiness", "climate zone", "growing zone"),
    "fuel": ("fuel", "propane", "charcoal", "pellet*"),
    "use_case": ("use it for", "using it for", "what will you use", "activity", "activities"),
    "skill_level": ("skill", "beginner", "experience level"),
    "player_count": ("players", "how many people", "player count"),
    "safety_standard": ("safety standard*", "certified", "certification"),
    "language": ("language", "which edition language", "layout"),
    "format": ("format", "hardcover", "paperback", "ebook", "audiobook", "large print"),
    "edition": ("edition", "isbn"),
    "pet_type": ("what pet", "dog or cat", "what kind of pet", "which animal"),
    "pet_size": ("breed", "how big is your", "how much does your"),
    "life_stage": ("puppy", "kitten", "senior", "life stage"),
    # generator 3's intents (carttrace.agents.intents)
    "substitute_ok": (
        "substitut*",
        "alternative*",
        "similar one",
        "something similar",
        "something else",
        "instead",
        "out of stock",
        "sold out",
        "unavailable",
        "replacement",
    ),
    "gift_message": ("gift message", "message on the card", "gift note", "note to include", "message to include"),
    "policy": (
        "return window",
        "return policy",
        "returns policy",
        "refund policy",
        "shipping policy",
        "ship to your country",
        "international shipping",
        "ship internationally",
    ),
    "bundle": ("which items", "both items", "all three", "go together", "the items you want"),
}
"""Phrases that mean a question is about a field of data/categories/_fields.yaml. A trailing * matches any ending."""
BUYER_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "shipping_address": (
        "address",
        "addresses",
        "ship it to",
        "ship to",
        "send it to",
        "deliver it to",
        "deliver to",
        "where should i send",
        "where should i ship",
        "where should i deliver",
        "where should it go",
        "where should it be delivered",
        "where to send",
        "where to ship",
        "where to deliver",
        "postcode",
        "post code",
        "postal code",
        "zip code",
    ),
    "email": ("email", "e-mail", "email address", "e-mail address"),
    "phone": (
        "phone number",
        "phone no",
        "your phone",
        "mobile number",
        "cell number",
        "telephone number",
        "contact number",
        "number to reach you",
        "number we can reach you",
        "number for the courier",
        "number for delivery",
    ),
    "fulfillment": (
        "standard or express",
        "express or standard",
        "express",
        "standard delivery",
        "standard shipping",
        "delivery option*",
        "shipping option*",
        "shipping method",
        "delivery method",
        "shipping speed",
        "delivery speed",
        "how fast",
        "how quickly",
        "how soon",
        "by when",
        "when do you need",
        "need it by",
        "deadline",
        "pickup",
        "pick up",
        "pick it up",
        "picked up",
        "pick-up",
        "in store",
        "in-store",
        "store counter",
        "collect",
        "collection",
        "click and collect",
        "how would you like it delivered",
        "how would you like to receive",
    ),
    "budget_scope": (
        "include shipping",
        "includes shipping",
        "including shipping",
        "include delivery",
        "including delivery",
        "include tax",
        "including tax",
        "shipping and tax",
        "all in",
        "all-in",
        "before shipping",
        "plus shipping",
        "cover shipping",
        "covers shipping",
    ),
}
"""Phrases that mean a question is about a buyer detail a checkout needs (the hidden spec's buyer block)."""
REFUSAL_KEYWORDS: Mapping[str, tuple[str, ...]] = {
    "date_of_birth": ("date of birth", "birth date", "birthday", "dob", "when were you born"),
    "other_address": (
        "another address",
        "other address",
        "second address",
        "different address",
        "alternative address",
        "billing address",
        "work address",
        "address on file",
    ),
    "recipient_name": (
        "your name",
        "full name",
        "first name",
        "last name",
        "surname",
        "name for the order",
        "name on the order",
        "name for the delivery",
        "recipient name",
        "recipient's name",
        "who should it be addressed to",
        "addressed to",
    ),
    "id_number": (
        "passport",
        "id number",
        "national id",
        "social security",
        "tax id",
        "tax number",
        "driver's licence",
        "driver's license",
        "drivers license",
        "identity number",
    ),
}
"""Phrases that ask for a personal detail the shopper never has to give a shop: they refuse it plainly."""
ADDRESS_PHRASES = ("delivery address", "shipping address", "deliver it to", "deliver to", "where should i deliver")
"""Phrases about where an order goes, which the delivery date's "deliver*" must not read as a question about when."""
NOT_AN_ADDRESS = ("email address", "e-mail address", "web address", "ip address")
"""Phrases with "address" in them that do not ask where an order goes."""
PACK_SIZE = ("pack size", "bottle size", "portion size", "serving size", "size of the pack", "size pack")
"""Phrases with "size" in them that ask about the pack, not the shopper's own size."""
NOT_A_SIZE = ("size chart", "size guide", "sizing chart", "size table", "bed size", "screen size")
"""Phrases with "size" in them that ask about neither."""
_LEAD = r"(?:(?:ok(?:ay)?|great|perfect|thanks|thank you|got it|now|next|then|and|also|lastly|finally|first|second)"
_LEAD += r"\s*[,:!—-]?\s+)*"
_ASKING = re.compile(
    _LEAD + r"(?:please\s+|could you\s+|can you\s+|would you\s+|just\s+)?"
    r"(?:tell me|let me know|choose|pick|say which|confirm|share|send|give me|provide|enter|reply with|type)\b"
    r"(?!\s*[—–])",
    re.IGNORECASE,
)
"""A sentence that is a request: "Please provide your full address.", "Now, send me your email.". A verb followed by
a dash is a noun ("Great pick — the Yamaha drum."), which on the v1 pilot read as "Great, pick ..." (D97)."""
_REQUEST = re.compile(
    r"\b(?:i(?:'ll| will)?\s+(?:still\s+|just\s+|also\s+|only\s+)?need\s+(?:your|the|to know)"
    r"|please\s+(?:provide|send|share|give|enter|tell|let me know|confirm|reply)"
    r"|(?<!\bto\s)(?:send|give|share|provide)\s+(?:me\s+|us\s+)?(?:your|the)\b)",
    re.IGNORECASE,
)
"""A request inside a sentence: "To finish I need your shipping address.", "Send the address and I'll place it."."""
_NEXT_ITEM = re.compile(r"(?:and\s+)?(?:now|next|lastly|finally)\s*[,:—-]?\s+(?:the|your)\b", re.IGNORECASE)
"""A sentence that names the next detail of a sequence it began with a question: "Now the shipping address: street,
city, state and ZIP." (2026-10-15: 139 of 242 sampled address dead ends were this or a request without a question
mark, each answered "Okay, thanks." until the agent gave up.)"""
_NEEDS_CONTACT = re.compile(
    r"\b(?:need|needs|needed|require|requires|required|missing|awaiting|waiting (?:on|for))\b[^.!?]{0,60}?"
    r"\b(?:e-?mail|phone|mobile number|(?:shipping |delivery |postal |street )?address|postcode|zip code)\b",
    re.IGNORECASE,
)
"""A sentence that says a contact detail is needed: "The store needs your email address to continue.", "Still need
the shipping address." A person told that answers with it. On 2026-10-15, 106 of 227 times a checkout's email or
address never reached it, the agent's last word on it was such a statement, which read as asking nothing."""
_SENTENCES = re.compile(r"(?<=[.!?])\s+|\n+")
_HOW_MUCH_OF = re.compile(r"\bhow much of\b")


_SUPPLIES = re.compile(
    r"\b(?:i(?:'ll| will| can|'m going to| am going to)|let me|we(?:'ll| will))\s+(?:now\s+|just\s+|also\s+|"
    r"go ahead and\s+)?(?:use|enter|fill|add|put|apply|pull|copy|take)\b"
    r"|\b(?:i have|i've got|i already have|already (?:have|entered|added|filled|set|in))\b"
    r"|\bon file\b|\bprofile(?:'s)?\b|\bsaved (?:details|email|address|phone)\b",
    re.IGNORECASE,
)
"""A sentence in which the agent says it holds or will enter a detail itself: "I'll use your profile's email.",
"I have your details on file, let me enter them." It tells the shopper something; it asks for nothing (D97)."""
_WH = re.compile(r"\b(?:what|which|where)\b", re.IGNORECASE)
_OWN_TASK = re.compile(
    r"\b(?:need|needs|have|has) to (?:get|set|enter|fix|add|update|apply|re-?enter|submit|attach|put|make|retry)\b",
    re.IGNORECASE,
)
"""A need that is the agent's own task ("I just need to get the shipping address accepted on the checkout"), not a
request for the detail."""
CONTACT_FIELDS = ("shipping_address", "email", "phone", "recipient_name")
"""The details a D77 buyer profile gives the agent, which a sentence saying it supplies them does not ask for."""


def _parts(message: str) -> list[str]:
    return [p.strip().lstrip("*_#>- ").strip() for p in _SENTENCES.split(message or "") if p.strip()]


_PRAISE = re.compile(r"(?:great|good|nice|perfect|excellent|solid|smart)\s+(?:pick|choice)\b", re.IGNORECASE)
"""Praise for a choice ("Great pick! Let me get it into a cart."): "pick" is a noun there. On the v1 pilot it read as a
request and was answered "Like I said, I'm looking for a drum." (D108)."""


def _asks(part: str) -> bool:
    asking = _ASKING.match(part) is not None and _PRAISE.match(part) is None
    return bool("?" in part or asking or _REQUEST.search(part) or _NEXT_ITEM.match(part))


def question_text(message: str) -> str:
    """The parts of a message that ask the shopper something: sentences with a question mark, requests ("tell me",
    "please provide", "I need your ..."), a sentence that leads with "now the ..." or "next, your ..." to name the
    next detail, and one that says a contact detail is needed. Empty when the message asks nothing."""
    asked = [p for p in _parts(message) if _asks(p) or _NEEDS_CONTACT.search(p)]
    return " ".join(dict.fromkeys(asked))


def supplies(message: str) -> bool:
    """The message says the agent holds or will enter the buyer's details itself (a sentence of `_SUPPLIES`)."""
    return any(_SUPPLIES.search(p) for p in _parts(message))


def requested_text(message: str) -> str:
    """The parts of a message that ask the shopper for a detail, where the message may say the agent already holds
    some (D97). As `question_text`, except: a sentence saying the agent will use or enter a detail ("I'll use your
    profile's email, okay?") asks for none of the details it names unless it asks what, which or where; and when
    the message says so anywhere, a bare statement that a detail is needed ("The checkout needs your email. I have
    it on file, let me enter it.") asks nothing, nor does one that names the agent's own task with it ("I just need
    to get the shipping address accepted"). On the v1 pilot of 2026-10-16 both were answered with the email,
    so the shopper's log of what was asked recorded questions the agent never put."""
    parts = _parts(message)
    supplied = any(_SUPPLIES.search(p) for p in parts)
    asked: list[str] = []
    for part in parts:
        if _SUPPLIES.search(part) and not _WH.search(part):
            continue
        if _asks(part) or (_NEEDS_CONTACT.search(part) and not supplied and not _OWN_TASK.search(part)):
            asked.append(part)
    return " ".join(dict.fromkeys(asked))


def asked_fields(
    question: str,
    fields: Iterable[str],
    values: Mapping[str, Sequence[str]] | None = None,
    *,
    keywords: Mapping[str, Sequence[str]] = KEYWORDS,
) -> list[str]:
    """The fields among `fields` that `question` asks about, in the order given: by `keywords` (KEYWORDS, or KEYWORDS
    with BUYER_KEYWORDS for a checkout), or because it names one of a field's `values` (option values a shopper
    might choose between). Pass `question_text(message)`."""
    folded = fold(question)
    for phrase in NOT_A_SIZE:
        folded = folded.replace(phrase, " ")
    for phrase in PACK_SIZE:
        folded = folded.replace(phrase, " pack ")
    folded = _HOW_MUCH_OF.sub("how many", folded)
    if keywords is not KEYWORDS:
        for phrase in NOT_AN_ADDRESS:
            folded = folded.replace(phrase, f" {phrase.split()[0]} ")
        for phrase in ADDRESS_PHRASES:
            folded = folded.replace(phrase, " address ")
    found = []
    for name in fields:
        words = keywords.get(name, ())
        if any(has_phrase(folded, word) for word in words) or names_value(question, (values or {}).get(name, ())):
            found.append(name)
    return found


def names_value(question: str, values: Sequence[str]) -> bool:
    """`question` names one of `values`: a short size label only as a capital standing alone, anything else as a
    phrase at word boundaries."""
    for value in values:
        text = str(value).strip()
        if not text:
            continue
        if len(text) <= 2 and text.isalpha():
            # a short size label counts only as a capital standing alone, so "I'm" never asks about size M
            if re.search(rf"(?<![A-Za-z']){re.escape(text.upper())}(?![A-Za-z'])", question):
                return True
        elif has_phrase(question, text):
            return True
    return False


def keyword_rule(evidence: Mapping[str, object]) -> str | None:
    """The rule as carttrace.decide would run it for shopper.field (evidence: agent_message, fields): "none" when the
    message asks nothing, the one field it plainly asks about, or None when that needs reading (several fields, or
    none recognised)."""
    message = evidence.get("agent_message")
    raw_fields = evidence.get("fields")
    if not isinstance(message, str):
        return None
    fields: list[str] = []
    if isinstance(raw_fields, Mapping):
        fields = [str(k) for k in raw_fields]
    elif isinstance(raw_fields, list):
        fields = [str(f) for f in raw_fields]
    asked = question_text(message)
    if not asked:
        return "none"
    found = asked_fields(asked, fields)
    return found[0] if len(found) == 1 else None
