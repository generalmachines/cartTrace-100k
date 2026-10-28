"""The voice pass: a model rewrites a task's template opening in the persona's words, behind a code guard.

Why a guard. The lab's intent writer (lab/intent_writer.py, 2026-09) let a model word every task, and its readers
found rewrites that dropped a budget, added a wish nobody judged, or named a value the shopper was meant to keep until
asked. D9 keeps free-text model output out of control flow, so here the template opening is the truth and the model
only changes the voice: it must keep every stated value word for word, must not mention any value the spec keeps for
asking (or a buyer detail), may use no digit the template did not, must not talk about paying or checking out, and may
add exactly one sentence of plausible context. Code checks all of that (`guard`); a rewrite that fails stands aside,
the template opening is used, and the reason is written into the spec's `voice` block beside the model and its cost,
so a run never calls the model again and a reader can see what was refused and why.

The model runs through the signed-in Claude Code CLI (`ClaudeCliRunner`, the agents-side twin of carttrace.ops.claude, which
this layer cannot import), Sonnet by default. Every caller takes a `VoiceRunner`, so tests pass a fake and never spend
a call. Costs are appended per call to a JSONL log, as the trace analyst records its own.
"""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final, Protocol

from pydantic import BaseModel, ConfigDict

from carttrace.agents.intents import variety_for
from carttrace.agents.slots import money
from carttrace.agents.spec import Address, HiddenSpec, Predicate, Voice
from carttrace.agents.wallet import Persona, Personas, load_personas
from carttrace.core.canon import sha256_hex
from carttrace.core.clock import Clock, SystemClock, iso
from carttrace.core.errors import ConfigError, ExternalServiceError, MCError
from carttrace.core.files import append_jsonl
from carttrace.core.ids import stable_id
from carttrace.trace.schema import TaskRef

DEFAULT_MODEL: Final = "sonnet"
TIMEOUT_S: Final = 120.0
DEFAULT_CLI: Final = "~/.local/bin/claude"
MAX_SENTENCES: Final = 6
MAX_CHARS: Final = 900
_PAYMENT: Final = re.compile(r"check ?out|pay now|buy now|purchase now|place (?:the|an|my) order|credit card", re.I)
_DIGITS: Final = re.compile(r"\d+")
_SENTENCE: Final = re.compile(r"[.!?](?:\s|$)")

PROMPT: Final = """You rewrite the opening message of an online shopping request in a particular shopper's own voice.
You change the voice, never the facts. Answer with one JSON object and nothing else: {{"opening": "<the rewrite>"}}

Rules:
- Every one of these values must appear word for word, as written here: {must_keep}
- Add exactly one short sentence of plausible context (why they want it, who it is for, what their day is like) that
  adds no requirement: no new product, quantity, price, budget, date, deadline, colour, size, brand, material,
  allergy, delivery or payment wish, and no question about the store.
- Use no digits except those in the original. Keep every quoted message exactly as quoted.
- Never mention checkout, payment, cards or buying now: the request ends at wanting the item.
- Write {sentences} sentences at most, in the first person, natural punctuation, no bullet points.

The shopper: {persona}.
The original request: {opening}
"""


class VoiceReply(BaseModel):
    """What a runner returns: the model's text, the model that served it, and the call's cost."""

    model_config = ConfigDict(frozen=True)

    text: str
    model: str | None = None
    cost_usd: float | None = None


class VoiceRunner(Protocol):
    def __call__(self, prompt: str, *, model: str, timeout_s: float) -> VoiceReply: ...


class ClaudeCliRunner:
    """`claude -p` with no tools, no MCP servers and no saved session; the prompt goes in on stdin. The same command
    carttrace.ops.claude.ClaudeCli runs; repeated here because layer 5 cannot import layer 7."""

    def __init__(self, cli: str = DEFAULT_CLI) -> None:
        path = Path(cli).expanduser()
        found = str(path) if path.exists() else shutil.which(cli)
        if found is None:
            raise ConfigError(f"the Claude Code CLI is not at {cli}", hint="install it, or pass --voice template")
        self.cli = found

    def __call__(self, prompt: str, *, model: str, timeout_s: float) -> VoiceReply:
        command = [
            self.cli, "-p", "--model", model, "--output-format", "json",
            "--no-session-persistence", "--tools", "", "--strict-mcp-config",
        ]  # fmt: skip
        try:
            done = subprocess.run(  # noqa: S603 - fixed argv, no shell; the prompt goes on stdin
                command, input=prompt, capture_output=True, text=True, timeout=timeout_s, check=False
            )
        except subprocess.TimeoutExpired as exc:
            raise ExternalServiceError("claude", f"no answer in {timeout_s:.0f} s") from exc
        if done.returncode != 0:
            detail = (done.stderr or done.stdout).strip()[-300:] or f"exit {done.returncode}"
            raise ExternalServiceError("claude", detail)
        try:
            reply = json.loads(done.stdout)
        except ValueError as exc:
            raise ExternalServiceError("claude", "the CLI did not print JSON") from exc
        if reply.get("is_error"):
            raise ExternalServiceError("claude", str(reply.get("result"))[:300])
        usage = reply.get("modelUsage") or {}
        return VoiceReply(
            text=str(reply.get("result") or ""), model=next(iter(usage), None), cost_usd=reply.get("total_cost_usd")
        )


# -- the guard ---------------------------------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class Facts:
    """What a rewrite must keep and must not say, read from the spec."""

    must_keep: tuple[str, ...]
    """Every stated value, as words."""
    forbidden: tuple[str, ...]
    """Every on_ask or default value and every buyer detail the opening does not state."""


def facts_of(spec: HiddenSpec, opening: str) -> Facts:
    keep: list[str] = []
    hide: list[str] = []
    for predicate in spec.predicates:
        words = value_words(predicate, spec.currency)
        if predicate.source == "stated":
            keep += [w for w in words if w.lower() in opening.lower()]
        else:
            hide += words
    buyer = spec.buyer
    if buyer is not None:
        for name in ("shipping_address", "email", "phone"):
            found = buyer.field(name)
            if found is None:
                continue
            value = found.value
            if isinstance(value, Address):
                hide += [str(value.lines[0]), str(value.postal_code), *([value.name] if value.name else [])]
            elif isinstance(value, str):
                hide.append(value)
    lowered = {k.lower() for k in keep}
    hidden = tuple(
        dict.fromkeys(h for h in hide if len(h) >= 2 and h.lower() not in lowered and h.lower() not in opening.lower())
    )
    return Facts(tuple(dict.fromkeys(keep)), hidden)


def value_words(predicate: Predicate, currency: str | None) -> list[str]:
    """A predicate's value as the words an opening carries: text as is, each item of a list, money as a price, a
    measurement as its amount; nothing for booleans, nulls and the names of things that are not said."""
    value = predicate.value
    if predicate.op in ("kind", "lines", "count") or value is None or isinstance(value, bool):
        return []
    if isinstance(value, str | int | float):
        return [str(value)]
    if isinstance(value, list):
        return [_item_words(item) for item in value if _item_words(item)]
    return _mapping_words(value) if isinstance(value, Mapping) else []


def _item_words(item: object) -> str:
    if isinstance(item, str):
        return item
    if isinstance(item, Mapping) and isinstance(item.get("title"), str):
        return str(item["title"])
    return ""


def _mapping_words(value: Mapping[str, Any]) -> list[str]:
    """Money as a price, a measurement as its amount, a shipping question as its country."""
    if "amount" in value and "currency" in value and isinstance(value["amount"], int):
        return [money(value["amount"], str(value["currency"]))]
    if "amount" in value and "measure" in value:
        return [str(value["amount"])]
    country = value.get("country")
    return [str(country)] if "topic" in value and isinstance(country, str) else []


def guard(opening: str, rewrite: str, facts: Facts) -> str | None:
    """Why a rewrite cannot stand, or None when it may: a stated value missing, a kept-back value mentioned, a digit
    the template never had, talk of paying, or too long."""
    text = rewrite.strip()
    if not text:
        return "the model returned no opening"
    lowered = text.lower()
    missing = [k for k in facts.must_keep if k.lower() not in lowered]
    leaked = [f for f in facts.forbidden if f.lower() in lowered]
    foreign = sorted(set(_DIGITS.findall(text)) - set(_DIGITS.findall(opening)))
    checks: list[tuple[bool, str]] = [
        (bool(missing), f"missing stated values: {', '.join(repr(m) for m in missing[:3])}"),
        (bool(leaked), f"mentions values the shopper keeps for asking: {', '.join(repr(f) for f in leaked[:3])}"),
        (bool(foreign), f"digits not in the template opening: {', '.join(foreign[:3])}"),
        (_PAYMENT.search(text) is not None, "mentions checkout or payment"),
        (len(text) > MAX_CHARS or len(_SENTENCE.findall(text)) > MAX_SENTENCES, "too long"),
    ]
    return next((reason for failed, reason in checks if failed), None)


# -- the pass ----------------------------------------------------------------------------------------------------------


def voice_prompt(opening: str, facts: Facts, persona: Persona | None) -> str:
    who = "a shopper"
    if persona is not None:
        register = variety_for(persona).register or "plain and direct"
        who = f"{persona.display_name or persona.name}, {register}; {persona.description.rstrip('.')}".strip()
    keep = "; ".join(repr(k) for k in facts.must_keep) or "(none)"
    return PROMPT.format(must_keep=keep, sentences=MAX_SENTENCES - 1, persona=who, opening=opening)


def parse_opening(text: str) -> str | None:
    """The rewrite in the model's reply: the "opening" of its JSON object, else None."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        return None
    try:
        data = json.loads(text[start : end + 1])
    except ValueError:
        return None
    found = data.get("opening") if isinstance(data, Mapping) else None
    return found.strip() if isinstance(found, str) and found.strip() else None


def voice_task(
    task: TaskRef,
    *,
    store_id: str,
    runner: VoiceRunner,
    model: str = DEFAULT_MODEL,
    personas: Personas | None = None,
    timeout_s: float = TIMEOUT_S,
) -> tuple[TaskRef, dict[str, Any]]:
    """The task with its opening rewritten when the guard allows, the voice recorded in its spec either way, and a log
    record of the call. A spec that already carries a voice is returned as it is, so a run never calls twice."""
    spec = HiddenSpec.from_task(task)
    if spec.voice is not None:
        return task, {"task_id": task.task_id, "skipped": "already voiced"}
    persona = _persona(spec, personas)
    facts = facts_of(spec, task.opening)
    prompt = voice_prompt(task.opening, facts, persona)
    try:
        reply = runner(prompt, model=model, timeout_s=timeout_s)
        rewrite = parse_opening(reply.text)
        reason = guard(task.opening, rewrite or "", facts) if rewrite is not None else "the reply holds no opening"
        served, cost = reply.model or model, reply.cost_usd
    except MCError as exc:
        rewrite, reason, served, cost = None, f"the model call failed: {exc.message}", model, None
    accepted = reason is None and rewrite is not None
    voice = Voice(
        model=served,
        accepted=accepted,
        template_opening=task.opening,
        reason=reason or "",
        cost_usd=cost,
        prompt_sha256=sha256_hex(prompt),
    )
    voiced = spec.model_copy(update={"voice": voice})
    opening = rewrite if accepted and rewrite is not None else task.opening
    hidden = voiced.to_json()
    out = task.model_copy(
        update={
            "task_id": stable_id("task", store_id, task.category, task.intent, opening, hidden),
            "opening": opening,
            "hidden_spec": hidden,
        }
    )
    record = {
        "task_id": out.task_id,
        "template_task_id": task.task_id,
        "store_id": store_id,
        "persona": persona.name if persona is not None else None,
        "model": served,
        "cost_usd": cost,
        "accepted": accepted,
        "reason": reason or "",
        "chars": len(opening),
    }
    return out, record


def voice_tasks(
    tasks: Sequence[TaskRef],
    *,
    store_id: str,
    runner: VoiceRunner,
    model: str = DEFAULT_MODEL,
    personas: Personas | None = None,
    log: Path | None = None,
    cost_cap_usd: float | None = None,
    clock: Clock | None = None,
    timeout_s: float = TIMEOUT_S,
) -> list[TaskRef]:
    """Every task through `voice_task`, in order, with one log line per call appended to `log`. Once the calls' known
    costs reach `cost_cap_usd`, the remaining tasks keep their template opening with the cap as the reason."""
    now = clock or SystemClock()
    personas = personas or load_personas()
    spent = 0.0
    out: list[TaskRef] = []
    for task in tasks:
        if cost_cap_usd is not None and spent >= cost_cap_usd:
            voiced, record = _capped(task, store_id=store_id, model=model, cap=cost_cap_usd)
        else:
            voiced, record = voice_task(
                task, store_id=store_id, runner=runner, model=model, personas=personas, timeout_s=timeout_s
            )
            spent += float(record.get("cost_usd") or 0.0)
        out.append(voiced)
        if log is not None:
            append_jsonl(log, {"at": iso(now.now()), **record})
    return out


def _capped(task: TaskRef, *, store_id: str, model: str, cap: float) -> tuple[TaskRef, dict[str, Any]]:
    spec = HiddenSpec.from_task(task)
    if spec.voice is not None:
        return task, {"task_id": task.task_id, "skipped": "already voiced"}
    reason = f"the voice pass reached its cost cap of {cap:.2f} USD before this task"
    voice = Voice(model=model, accepted=False, template_opening=task.opening, reason=reason)
    hidden = spec.model_copy(update={"voice": voice}).to_json()
    out = task.model_copy(
        update={
            "task_id": stable_id("task", store_id, task.category, task.intent, task.opening, hidden),
            "hidden_spec": hidden,
        }
    )
    return out, {"task_id": out.task_id, "template_task_id": task.task_id, "store_id": store_id, "model": model,
                 "cost_usd": None, "accepted": False, "reason": reason, "chars": len(task.opening)}  # fmt: skip


def _persona(spec: HiddenSpec, personas: Personas | None) -> Persona | None:
    name = spec.buyer.persona if spec.buyer is not None else None
    if name is None:
        return None
    try:
        return (personas or load_personas()).get(name)
    except MCError:
        return None
