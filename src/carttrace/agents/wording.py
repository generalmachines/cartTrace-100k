"""The shopper's words (D118): the rules decide every fact of a reply, a small model words them, and code checks.

Why a hybrid. The deterministic shopper (carttrace.agents.shopper) is consistent and judgeable, but it says the same lines
again and again ("I'll collect it from the store", "No preference on the colour. No preference on the material.",
"Like I said, ..."), which no person does. A model playing the shopper invents preferences the judge then scores the
agent against (D9, lab/shopper.py). So the rules still decide everything: which field the agent asked about, the
answer's value or "none", which offered item is accepted or refused and why, consent and its total, and what is
disclosed (the judge reads `disclosed`, which this module never touches). The rules' line is the truth; the model
only says it in the persona's voice.

The check (`check`). The model's words are kept only when code finds that they say exactly the rules' facts:
- every value the line states (a price, a number, an email, an address, a product or option name, a quoted gift
  message, a value of the hidden spec, a field the line says there is no preference on or that is withheld) is in the
  words, numbers and prices as digits;
- nothing the line does not state: no other price or number, email or phone, no value of the hidden spec, product,
  offered option or colour the line does not name, no way of delivery the line does not name, never "ship it" with
  "I'll collect it";
- the line's stance kept: a yes (consent, a go-ahead) stays a yes with no refusal in it, a no stays a no, an option
  picked is not turned down, and a question stays the only question;
- one or two sentences (as many as the line has, when it has more), plain text, and never an earlier reply again.
A refused wording is asked once more with the reason; if that is refused too, the rules' line is sent. Every reply
says which (`wording`: model or fallback), and the calls' tokens ride on the shopper's turn as its `usage`.

The model is the `shopper` role (config.json roles), thinking off, `agents.shopper.max_tokens` (120), through the
same host-wide GPU slots as the agent (D114). Callers take a `Worder`, so tests pass a seeded fake and never call it.
"""

from __future__ import annotations

import json
import os
import re
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Final, Literal, Protocol

from carttrace.agents.nouns import COLOURS
from carttrace.agents.text import fold, has_phrase
from carttrace.core.canon import sha256_hex

if TYPE_CHECKING:
    from carttrace.config.schema import AgentShopper, ModelEndpoint, Settings
    from carttrace.inference.calls import ModelClient

WORDING_CODE: Final = "mc.agents.wording/1"
"""The wording prompt and check's version: raise it when either changes what a worded reply may say."""
WordingSource = Literal["model", "fallback"]
Stance = Literal["approve", "pick", "refuse"]
"""approve: a yes to an order or a cart (consent, a go-ahead); pick: an offered option chosen or agreed to; refuse:
a no (consent refused, none of the options fits, a cart that is not what was asked for)."""
VoiceName = Literal["terse", "chatty", "formal"]
VOICES: Final[Mapping[VoiceName, str]] = {
    "terse": "terse and to the point: short plain sentences, no small talk",
    "chatty": "warm and chatty: friendly, a little informal, may add a few words of thanks or warmth",
    "formal": "polite and formal: complete sentences, courteous phrasing, no slang",
}
LOG_ENV: Final = "CARTTRACE_SHOPPER_WORDING_LOG"
"""A JSONL path: when set, each worded reply's rules line, words and check are appended to it (for reading a run)."""
WORD_TIMEOUT_S: Final = 60.0
"""A wording call waits at most this long for its reply (GPU slot waits excluded): the rules' line is sent then."""
REFUSAL_PATIENCE_S: Final = 30.0
"""How long a wording call backs off on the GPU's refusals (D114) before the rules' line is sent; the agent's own
calls wait 15 minutes, a shopper's words are not worth that."""

_MONEY = re.compile(r"(?:[$£€¥]\s?\d[\d,]*(?:\.\d+)?)|(?:\b\d[\d,]*(?:\.\d+)?\s?(?:USD|EUR|GBP|CHF|CAD|AUD|JPY|SGD)\b)")
_NUMBER = re.compile(r"(?<![\w.])\d+(?:\.\d+)?(?![\w])")
_EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)+")
_QUOTED = re.compile(r"\"([^\"]{2,})\"")
_SENTENCE_END = re.compile(r"[.!?]+(?=\s|$)")
_EMOJI = re.compile("[\U0001f000-\U0001ffff☀-➿]")
_PREFIX = re.compile(r"^\s*(?:shopper|customer|me|reply|message)\s*:\s*", re.IGNORECASE)
_APPROVE = re.compile(
    r"\b(?:yes|yeah|yep|sure|go ahead|go for it|please (?:place|order|do|proceed)|place (?:the|my|it)|proceed|"
    r"sounds (?:good|right|great|perfect)|that works|works for me|perfect|great|happy (?:with|for)|okay|ok|"
    r"that's (?:what|it|right|fine|great|perfect))\b",
    re.IGNORECASE,
)
_REFUSE = re.compile(
    r"\b(?:no|not|none|nope|don't|dont|do not|won't|wont|isn't|aren't|doesn't|rather not|hold off|stop|cancel|"
    r"neither|wrong)\b|n't\b",
    re.IGNORECASE,
)
_ACCEPTS = re.compile(
    r"\b(?:yes|yeah|go ahead and (?:place|order)|please (?:place|order|proceed)|sounds (?:good|right|great|perfect)|"
    r"that works|works for me|i'll take|i'll go with|let's go with)\b",
    re.IGNORECASE,
)
_INDIFFERENT = re.compile(
    r"\b(?:no preference|don't mind|do not mind|any|anything|whatever|either|not fussed|doesn't matter|"
    r"does not matter|fine|no rush|happy with|up to you|your call|you (?:choose|pick|decide)|what(?:ever)? "
    r"(?:fits|works|suits)|no need|not needed|nothing|none|no)\b",
    re.IGNORECASE,
)
_INDIFFERENT_LINE = re.compile(
    r"\b(?:no preference|don't mind|any of those|any \w+ is fine|whatever|no rush|nothing like that|no \w+ needed|"
    r"no special|no allergies|no questions)\b",
    re.IGNORECASE,
)
_WITHHOLD_LINE = re.compile(r"\b(?:rather not|won't|don't have|no other|not to hand)\b", re.IGNORECASE)
_SHIP_IT = re.compile(r"\b(?:ship (?:it|them|that|this)|send (?:it|them)|deliver (?:it|them)|post (?:it|them))\b", re.I)
_COLLECT = re.compile(r"\b(?:collect|pick (?:it|them) up|pick-up|pickup|in store|in-store)\b", re.IGNORECASE)
WAY_TERMS: Final[Mapping[str, tuple[str, ...]]] = {
    "pickup": ("pickup", "pick up", "pick it up", "picked up", "pick-up", "collect*", "in store", "in-store",
               "store counter", "from the store"),
    "express": ("express", "expedited", "next day", "next-day", "overnight", "priority", "courier"),
    "standard": ("standard", "ground", "regular", "economy", "normal"),
    "ship": ("shipping", "ship", "shipped", "delivery", "delivered", "deliver", "post", "posted", "mail", "send",
             "sent"),
}  # fmt: skip
"""Ways to receive an order as a reply may word them; `ship` is shipping of no named speed. Wider than the agent-side
WAYS of carttrace.agents.shopper, because here a word that might name a way is refused, not read."""
_BRITISH: Final = (("colour", "color"), ("flavour", "flavor"), ("isation", "ization"), ("ise ", "ize "))


@dataclass(frozen=True, slots=True)
class WordingReply:
    """What a Worder returns: the model's text, the served model, and the call's token counts."""

    text: str
    model: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)


class Worder(Protocol):
    """One wording call: chat messages in, the model's reply out. Raises when the model cannot answer; the voice then
    sends the rules' line."""

    async def __call__(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int, temperature: float, seed: int
    ) -> WordingReply: ...


@dataclass(frozen=True, slots=True)
class Facts:
    """What a reply must say and must not, read from the rules' line and what the shopper knows."""

    line: str
    values: tuple[str, ...] = ()
    """Phrases the words must hold (folded, at word boundaries): values, names, quoted text, emails."""
    numbers: tuple[str, ...] = ()
    """Every number the line states, prices included, as digits: the words hold exactly these numbers."""
    money: tuple[str, ...] = ()
    """Prices the line states as written ("$43.78"); the words hold each and no other."""
    labels: tuple[str, ...] = ()
    """Field labels the line names ("colour" in "No preference on the colour"): the words name each too."""
    forbidden: tuple[str, ...] = ()
    """Phrases the shopper knows that the line does not say: a hidden value, another product or option, a colour."""
    ways: frozenset[str] = frozenset()
    """The ways of delivery the line names (WAY_TERMS)."""
    stance: Stance | None = None
    indifferent: bool = False
    """The line says there is no preference (or nothing needed): the words keep an indifferent cue."""
    withholds: bool = False
    """The line withholds or refuses a detail: the words keep a refusal cue."""
    question: bool = False
    """The line asks the agent something ("Is that possible?"): the words ask too, and only then."""

    def must_keep(self) -> list[str]:
        """What the prompt lists to keep word for word."""
        out = [*self.money, *self.values]
        out += [n for n in self.numbers if not any(n in m for m in out)]
        return list(dict.fromkeys(out))


@dataclass(frozen=True, slots=True)
class Worded:
    """A reply's words and where they came from."""

    text: str
    wording: WordingSource
    model: str | None = None
    usage: Mapping[str, int] = field(default_factory=dict)
    attempts: int = 0
    refused: tuple[str, ...] = ()
    """Why each refused try was refused, in order."""


def facts_of(
    line: str,
    *,
    known: Iterable[str] = (),
    hidden: Iterable[str] = (),
    labels: Iterable[str] = (),
    stance: Stance | None = None,
) -> Facts:
    """The facts of the rules' `line`. `known`: what the agent has been told (stated and revealed values, disclosed
    details), required when the line says it and free to say again otherwise; `hidden`: what the agent has not been
    told or the shopper did not choose (values kept for asking, details not given, other products and options),
    required when the line says it and forbidden when it does not. Each field label the line names is required, and
    a colour the line and the known values do not name is forbidden."""
    said = _norm(line)
    money = tuple(dict.fromkeys(m.group(0).strip() for m in _MONEY.finditer(line)))
    numbers = tuple(dict.fromkeys(_numbers(line)))
    values: list[str] = [m.group(0).lower() for m in _EMAIL.finditer(line)]
    values += [m.group(1).strip() for m in _QUOTED.finditer(line)]
    told = [p.strip() for p in known if p and len(p.strip()) >= 3 and not _NUMBER.fullmatch(p.strip())]
    forbidden: list[str] = []
    for phrase in told:
        if has_phrase(said, _norm(phrase)):
            values.append(phrase)
    told_norm = [_norm(p) for p in told]
    for phrase in dict.fromkeys(p.strip() for p in hidden if p and len(p.strip()) >= 3):
        if _NUMBER.fullmatch(phrase):
            continue
        if has_phrase(said, _norm(phrase)):
            values.append(phrase)
        elif _norm(phrase) not in told_norm:
            forbidden.append(phrase)
    for colour in sorted(COLOURS):
        if not has_phrase(said, colour) and not any(has_phrase(t, colour) for t in told_norm):
            forbidden.append(colour)
    named = [label for label in dict.fromkeys(labels) if label and has_phrase(said, _norm(label))]
    # a value said whole covers the shorter phrases inside it ("Moss Lane" in the address)
    values = [v for v in dict.fromkeys(values) if not any(v != w and _norm(v) in _norm(w) for w in values)]
    forbidden = [f for f in dict.fromkeys(forbidden) if not any(_norm(f) in _norm(v) for v in values)]
    return Facts(
        line=line,
        values=tuple(values),
        numbers=numbers,
        money=money,
        labels=tuple(named),
        forbidden=tuple(forbidden),
        ways=frozenset(ways_in(line)),
        stance=stance,
        indifferent=bool(_INDIFFERENT_LINE.search(line)),
        withholds=bool(_WITHHOLD_LINE.search(line)),
        question="?" in line,
    )


def check(text: str, facts: Facts, *, said: Iterable[str] = ()) -> str | None:  # noqa: PLR0911, PLR0912 - one return per rule
    """Why `text` does not say exactly the facts, or None when it does. `said`: the shopper's earlier replies."""
    if not text.strip():
        return "empty"
    if "\n" in text.strip() or _EMOJI.search(text) or any(c in text for c in "*#[]{}<>|`"):
        return "not plain text"
    if fold(text) in {fold(s) for s in said}:
        return "repeats an earlier reply"
    out = _norm(text)
    sentences = len(_SENTENCE_END.findall(text.strip())) or 1
    if sentences > max(2, len(_SENTENCE_END.findall(facts.line)) or 1):
        return f"{sentences} sentences"
    if len(text) > max(220, 2 * len(facts.line) + 40):
        return "too long"
    missing_money = [m for m in facts.money if _norm(m).replace(" ", "") not in out.replace(" ", "")]
    if missing_money:
        return f"leaves out {missing_money[0]}"
    numbers = set(_numbers(text))
    if set(facts.numbers) - numbers:
        return f"leaves out the number {sorted(set(facts.numbers) - numbers)[0]}"
    if numbers - set(facts.numbers):
        return f"adds the number {sorted(numbers - set(facts.numbers))[0]}"
    emails = {m.group(0).lower() for m in _EMAIL.finditer(text)}
    if emails - {v for v in facts.values if "@" in v}:
        return "adds an email address"
    for value in facts.values:
        if not has_phrase(out, _norm(value)):
            return f"leaves out {value!r}"
    for label in facts.labels:
        if not any(has_phrase(out, w) for w in _label_words(label)):
            return f"leaves out the {label}"
    for detail in DETAIL_WORDS:
        if has_phrase(out, detail) and not _gives(facts, detail):
            return f"adds the {detail}"
    for phrase in facts.forbidden:
        if has_phrase(out, _norm(phrase)):
            return f"adds {phrase!r}"
    ways = ways_in(text)
    if ways - facts.ways - _covered(facts.ways):
        return f"adds a way of delivery ({', '.join(sorted(ways - facts.ways))})"
    if {w for w in facts.ways if w != "ship"} - ways:
        return "leaves out the way of delivery"
    if _SHIP_IT.search(text) and _COLLECT.search(text):
        return "says ship it and collect it"
    stance = _stance_problem(text, facts)
    if stance is not None:
        return stance
    if facts.indifferent and not _INDIFFERENT.search(text):
        return "loses the no preference"
    if facts.withholds and not _REFUSE.search(text):
        return "loses the refusal"
    if facts.question != ("?" in text):
        return "adds a question" if "?" in text else "drops the question"
    return None


def _stance_problem(text: str, facts: Facts) -> str | None:
    if facts.stance == "approve":
        if not _APPROVE.search(text):
            return "loses the yes"
        if _REFUSE.search(text):
            return "a yes with a no in it"
    elif facts.stance == "pick" and _REFUSE.search(text):
        return "turns the pick down"
    elif facts.stance == "refuse":
        if not _REFUSE.search(text):
            return "loses the no"
        if _ACCEPTS.search(text):
            return "a no with a yes in it"
    return None


DETAIL_WORDS: Final = ("address", "email", "phone", "date of birth", "id number", "card")
"""Checkout details a reply names only when its line does: "I would be pleased to provide my phone number" answered
nothing that asked for it."""


def _gives(facts: Facts, detail: str) -> bool:
    """The line names the detail, or gives its value (an address's street, an email, a phone number)."""
    if has_phrase(_norm(facts.line), detail):
        return True
    if detail == "email":
        return "@" in facts.line
    if detail == "phone":
        return len(re.sub(r"\D", "", facts.line)) >= 7 and "+" in facts.line
    return detail == "address" and len(facts.numbers) >= 2 and len(facts.values) >= 1


_LABEL_STOP: Final = frozenset({"kind", "what", "the", "with", "your", "number", "address", "person"})


def _label_words(label: str) -> list[str]:
    """The words of a field label any one of which names it ("email address": email); the label whole when none."""
    found = [w for w in _norm(label).replace("'", " ").split() if len(w) >= 4 and w not in _LABEL_STOP]
    return found or [_norm(label)]


def _covered(ways: frozenset[str]) -> set[str]:
    """Ways a reply may word as plain shipping: a standard or express wish is shipped."""
    return {"ship"} if ways & {"standard", "express", "ship"} else set()


def ways_in(text: str) -> set[str]:
    """The ways of delivery `text` names."""
    return {way for way, terms in WAY_TERMS.items() if any(has_phrase(text, t) for t in terms)}


def _numbers(text: str) -> list[str]:
    plain = re.sub(r"(?<=\d),(?=\d{3}\b)", "", _EMAIL.sub(" ", text))
    return _NUMBER.findall(plain)


def _norm(text: str) -> str:
    out = fold(text)
    for british, american in _BRITISH:
        out = out.replace(british, american)
    return out


# -- the prompt ------------------------------------------------------------------------------------------------------

SYSTEM: Final = """You are role-playing a shopper chatting with an online shopping assistant. Write the shopper's \
next message. WHAT TO SAY gives its meaning: put it in your own words, the way this shopper talks, rather than \
copying its wording.
Rules:
- Say every fact in WHAT TO SAY and nothing more: no new fact, preference, product, number, price, date, delivery \
wish, question or promise.
- Copy each item under KEEP EXACTLY exactly as written (names, numbers, prices, emails, addresses); write numbers \
as digits.
- Keep the meaning: a yes stays a yes, a no stays a no, no preference stays no preference; ask a question only if \
WHAT TO SAY asks one.
- Do not mention shipping, delivery, pickup or any checkout detail unless WHAT TO SAY does.
- {length}. Plain text: no quotes, lists or emoji.
- Do not reuse the phrasing of the shopper's earlier messages.
The shopper's voice: {voice}."""


def messages_for(
    facts: Facts,
    *,
    voice: VoiceName,
    history: Sequence[tuple[str, str]],
    refused: str | None = None,
) -> list[dict[str, str]]:
    """The wording call's messages: the rules, the persona's voice, the conversation so far (`history`, as
    (speaker, text) with the speaker "Shopper" or "Assistant"), and what the shopper must say."""
    sentences = len(_SENTENCE_END.findall(facts.line)) or 1
    length = "One or two sentences" if sentences <= 2 else f"At most {sentences} short sentences"
    system = SYSTEM.format(length=length, voice=VOICES[voice])
    lines = [f"{who}: {_clip(text)}" for who, text in history]
    keep = facts.must_keep()
    user = "Conversation so far (latest last):\n" + ("\n".join(lines) or "(none)")
    user += f"\n\nWHAT TO SAY: {facts.line}"
    if keep:
        user += "\nKEEP EXACTLY: " + "; ".join(keep)
    if refused:
        user += f"\nYour last try was refused ({refused}). Try again, keeping to WHAT TO SAY."
    user += "\n\nWrite only the shopper's message."
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _clip(text: str, limit: int = 320) -> str:
    flat = " ".join(text.split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def clean(text: str) -> str:
    """The model's reply as a message: one line, without a speaker label or quotes around it."""
    out = _PREFIX.sub("", " ".join(text.strip().split()))
    if len(out) >= 2 and out[0] == out[-1] and out[0] in "\"'“”":
        out = out[1:-1].strip()
    return out.strip("“”").strip()


def voice_for(persona: str | None, task_id: str) -> VoiceName:
    """The shopper's voice for a task: drawn from the persona and the task, so one task always has the same voice."""
    names: list[VoiceName] = ["terse", "chatty", "formal"]
    return names[int(sha256_hex(f"wording.voice|{persona or ''}|{task_id}")[:8], 16) % len(names)]


# -- the shopper's voice over one episode ----------------------------------------------------------------------------


class ShopperVoice:
    """Words one episode's replies: keeps the conversation it has seen, asks the Worder, checks, falls back."""

    def __init__(
        self,
        worder: Worder,
        *,
        task_id: str,
        persona: str | None = None,
        max_tokens: int = 120,
        temperature: float = 0.8,
        retries: int = 1,
        history: int = 6,
    ) -> None:
        self.worder = worder
        self.task_id = task_id
        self.voice: VoiceName = voice_for(persona, task_id)
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.retries = retries
        self.keep = history
        self.opening: tuple[str, str] | None = None
        self.history: list[tuple[str, str]] = []
        self.said: list[str] = []
        self.turns = 0
        self.usage: dict[str, int] = {}
        """Token counts summed over every wording call of the episode."""
        self.counts: dict[WordingSource, int] = {"model": 0, "fallback": 0}

    @classmethod
    def from_settings(
        cls, worder: Worder, config: AgentShopper, *, task_id: str, persona: str | None = None
    ) -> ShopperVoice:
        return cls(
            worder,
            task_id=task_id,
            persona=persona,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            retries=config.retries,
            history=config.history,
        )

    def opened(self, text: str) -> None:
        self.opening = ("Shopper", text)
        self.said.append(text)

    def heard(self, text: str) -> None:
        """An agent message the shopper read."""
        if text.strip():
            self.history.append(("Assistant", text))

    async def word(self, facts: Facts, *, rule: str = "") -> Worded:
        """The reply's words: the model's when they pass `check` (within `retries` more tries), else the line."""
        self.turns += 1
        usage: dict[str, int] = {}
        refused: list[str] = []
        model: str | None = None
        recent = self.history[-self.keep :] if self.keep else []
        shown = ([self.opening] if self.opening is not None and self.opening not in recent else []) + recent
        for attempt in range(self.retries + 1):
            messages = messages_for(facts, voice=self.voice, history=shown, refused=refused[-1] if refused else None)
            seed = int(sha256_hex(f"wording|{self.task_id}|{self.turns}|{attempt}")[:8], 16)
            try:
                reply = await self.worder(messages, max_tokens=self.max_tokens, temperature=self.temperature, seed=seed)
            except Exception as exc:  # any failure of the wording call sends the rules' line, never ends a row
                refused.append(f"call failed: {type(exc).__name__}")
                break
            model = reply.model or model
            for key, value in reply.usage.items():
                usage[key] = usage.get(key, 0) + int(value)
            text = clean(reply.text)
            why = check(text, facts, said=self.said)
            if why is None:
                return self._done(Worded(text, "model", model, usage, attempt + 1, tuple(refused)), facts, rule)
            refused.append(why)
        attempts = sum(1 for r in refused if not r.startswith("call failed"))
        return self._done(Worded(facts.line, "fallback", model, usage, attempts, tuple(refused)), facts, rule)

    def _done(self, worded: Worded, facts: Facts, rule: str) -> Worded:
        self.counts[worded.wording] += 1
        for key, value in worded.usage.items():
            self.usage[key] = self.usage.get(key, 0) + value
        self.said.append(worded.text)
        self.history.append(("Shopper", worded.text))
        _log(self.task_id, rule, facts, worded)
        return worded


def _log(task_id: str, rule: str, facts: Facts, worded: Worded) -> None:
    path = os.environ.get(LOG_ENV)
    if not path:
        return
    row: dict[str, Any] = {
        "task_id": task_id,
        "rule": rule,
        "rules_text": facts.line,
        "text": worded.text,
        "wording": worded.wording,
        "attempts": worded.attempts,
        "refused": list(worded.refused),
        "usage": dict(worded.usage),
    }
    with Path(path).open("a", encoding="utf-8") as out:
        out.write(json.dumps(row, ensure_ascii=False) + "\n")


# -- the shopper role's model ----------------------------------------------------------------------------------------


class QwenWorder:
    """A Worder on the `shopper` role's chat endpoint: thinking off, through the host-wide GPU slots (D114)."""

    def __init__(self, settings: Settings, *, role: str = "shopper", client: ModelClient | None = None) -> None:
        from carttrace.inference.calls import ModelClient

        endpoint = settings.role(role)
        self.endpoint: ModelEndpoint = endpoint.model_copy(
            update={"timeout_s": min(endpoint.timeout_s, WORD_TIMEOUT_S)}
        )
        self.client = client or ModelClient(settings, refusal_patience_s=REFUSAL_PATIENCE_S)

    async def __call__(
        self, messages: Sequence[Mapping[str, str]], *, max_tokens: int, temperature: float, seed: int
    ) -> WordingReply:
        from carttrace.inference.openai_chat import CHAT_PATH, build_request, parse_chat_response

        endpoint = self.endpoint
        body = build_request(
            endpoint, [dict(m) for m in messages], None, max_tokens=max_tokens, temperature=temperature, seed=seed
        )
        template = body.get("chat_template_kwargs")
        body["chat_template_kwargs"] = {**(template if isinstance(template, dict) else {}), "enable_thinking": False}
        result = await self.client.post(
            endpoint, CHAT_PATH, body, lambda reply: parse_chat_response(endpoint, reply.payload, reply.latency_ms)
        )
        return WordingReply(result.text, result.served_model or endpoint.model, dict(result.usage))


_WORDERS: dict[int, tuple[Settings, QwenWorder]] = {}


def worder_for(settings: Settings) -> QwenWorder:
    """The process's QwenWorder for `settings`: one HTTP client per settings object."""
    found = _WORDERS.get(id(settings))
    if found is not None and found[0] is settings:
        return found[1]
    worder = QwenWorder(settings)
    _WORDERS[id(settings)] = (settings, worder)
    return worder
