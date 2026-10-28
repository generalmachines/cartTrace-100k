"""What an agent run promises, and the parts it talks to: the trace sink, the shopper and the supervisor.

Why contracts here. The collection runner (carttrace.collect) runs two kinds of agent through the same loop: the Qwen
protocol agent, whose choices come from a model, and the reference agent, which knows the hidden spec and uses only
what the store's surfaces return, so that a failure can be pinned on the store or on the agent. Both take the same
inputs (a task, the surfaces, a shopper, a sink for turns, limits and an optional supervisor) and return the same
`AgentResult`, so the runner, the judge and the benchmark never ask which one ran.

The sink is TraceWriter.turn's signature exactly, so a TraceWriter is a TurnSink and tests can pass a list-backed
double. The shopper is the other side of the conversation, and on a checkout row (D13) the one who approves the order
and its total; the supervisor is the decision engine's view of whether an episode is looping or stuck.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, ConfigDict, Field

from carttrace.core.money import Money
from carttrace.protocols.tools import ToolSurface
from carttrace.trace.schema import Actor, Cart, EndReason, TaskRef, TurnKind

if TYPE_CHECKING:
    from carttrace.config.schema import Settings

DeclaredStatus = Literal["cart_ready", "order_placed", "selection_ready", "no_match", "cannot_continue"]
"""What an agent says when it ends: a cart is ready, an order was placed (sandbox checkout rows only), nothing in the
store meets the need, or it cannot go on."""
ConsentReason = Literal["consented", "over_budget", "cart_mismatch", "total_mismatch", "currency_mismatch"]
"""Why the shopper answered an order approval as they did; everything but consented is a refusal."""


@dataclass(frozen=True, slots=True)
class Limits:
    """How far one episode may go. The runner builds these from config.json (collect and agents sections)."""

    step_cap: int
    """Agent steps before the episode ends with step_cap. A step is one model turn, or one action of the reference
    agent."""
    time_cap_s: float
    """Seconds, on the injected clock, before the episode ends with time_cap."""
    max_tokens: int
    """Reply budget of one model turn."""

    def __post_init__(self) -> None:
        if self.step_cap < 1 or self.time_cap_s <= 0 or self.max_tokens < 1:
            raise ValueError("limits are positive: step_cap >= 1, time_cap_s > 0, max_tokens >= 1")

    @classmethod
    def from_settings(cls, settings: Settings) -> Limits:
        """collect.step_cap, collect.time_cap_s and agents.max_tokens."""
        return cls(
            step_cap=settings.collect.step_cap,
            time_cap_s=float(settings.collect.time_cap_s),
            max_tokens=settings.agents.max_tokens,
        )


class TurnSink(Protocol):
    """Where every turn goes: `TraceWriter.turn`, or a test double with the same signature."""

    def turn(
        self,
        actor: Actor,
        kind: TurnKind,
        *,
        text: str | None = None,
        tool: str | None = None,
        surface: str | None = None,
        arguments: dict[str, Any] | None = None,
        result: Any = None,
        is_error: bool = False,
        error_code: str | None = None,
        model: str | None = None,
        usage: dict[str, int] | None = None,
        latency_ms: int | None = None,
        seed: int | None = None,
        call_id: str | None = None,
        tool_call_id: str | None = None,
        finish_reason: str | None = None,
        cost: Mapping[str, Any] | None = None,
        origin: Mapping[str, Any] | None = None,
        served: str | None = None,
    ) -> object: ...


@runtime_checkable
class DefinitionSink(Protocol):
    """A sink that also keeps what the agent was given (D69): `TraceWriter.agent_definition`. An agent records its
    system prompt and tool definitions there once it has listed its tools; a sink without it is simply not told."""

    def agent_definition(self, definition: Mapping[str, Any]) -> object: ...


WordingSource = Literal["model", "fallback"]
"""Who worded a shopper reply's decided facts (D118)."""


def _origin(rule: str, version: str | None, wording: WordingSource | None, model: str | None) -> dict[str, Any]:
    out: dict[str, Any] = {"rule": rule, "version": version, "author": "rules"}
    if wording == "model":
        out |= {"author": "model", "wording": "model"} | ({"voice_model": model} if model else {})
    elif wording == "fallback":
        out["wording"] = "fallback"
    return out


def _extras(model: str | None, usage: Mapping[str, int] | None) -> dict[str, Any]:
    if not usage:
        return {}
    return {"usage": {k: int(v) for k, v in usage.items()}} | ({"model": model} if model else {})


ShopperSource = Literal["lookup", "matched", "default", "model"]
"""Where a shopper's answer came from (ShopperReply.source, D115)."""


@dataclass(frozen=True, slots=True)
class ShopperReply:
    """The shopper's answer to one agent message."""

    text: str
    field: str | None
    """The hidden-spec field the message was understood to ask about (the first, when it asked about several); None
    when it asked about nothing the shopper knows."""
    revealed: bool
    """True when the reply gave a value from the hidden spec (a stated one repeated, or an on_ask one disclosed)."""
    fields: tuple[str, ...] = ()
    """Every field the reply answered about, in spec order."""
    disclosed: tuple[str, ...] = ()
    """The buyer details (carttrace.agents.spec BUYER_FIELDS) whose values this reply gave: what an agent may enter at
    checkout from here on. A field the shopper withheld or refused is never in it."""
    choice: ChoiceRecord | None = None
    """How the shopper answered an offer of options or a recommendation (D60); None for every other reply."""
    cut: bool = False
    """The agent's message came from a turn the token limit cut (D60): not the shopper's to choose from."""
    rule: str | None = None
    """The branch that produced the reply (D69): keyword_field, decided_field, nothing_asked, other or choice; also
    acknowledged and payment_offer (D97), quantity_field and payment_go_ahead (D108); on a checkout row's held cart,
    go_ahead or cart_mismatch."""
    version: str | None = None
    """The shopper's code and phrases version that produced it."""
    source: ShopperSource | None = None
    """Where the answer came from (D115): `lookup`, the task's build-time decision map (which offered item is
    acceptable, the delivery wish, the substitute policy); `matched`, the keyword rules and the store's own field
    and option names; `default`, the safe answer when nothing matched. Never a model: what a reply says is decided
    by code, and a model only words it (D118, `wording`)."""
    wording: WordingSource | None = None
    """Who worded the rules' facts (D118): `model`, the shopper's model, its words checked by code; `fallback`, the
    rules' own line, sent when the words failed the check; None when nothing was worded by model."""
    wording_model: str | None = None
    usage: Mapping[str, int] | None = None
    """The wording calls' token counts, summed over this reply's tries (D118); None when no call was made."""

    def origin(self) -> dict[str, Any] | None:
        """The turn's `origin` (D69): the rule and version behind the reply, and who worded it (D118); None for a
        shopper that names none."""
        if self.rule is None:
            return None
        return _origin(self.rule, self.version, self.wording, self.wording_model)

    def turn_extras(self) -> dict[str, Any]:
        """The turn's `model` and `usage` when a wording call was made (D118), else nothing: what the shopper's
        model spent on this reply, on the turn it worded."""
        return _extras(self.wording_model, self.usage)

    def annotation(self) -> dict[str, Any]:
        """The `arguments` a shopper's message turn carries, which the judge reads: what the reply answered about,
        whether it gave a value, and (only when it gave any) which buyer details it disclosed. A reply to an offer
        adds `choice`, and a reply to a cut message adds `cut`; every other reply keeps the bytes it had."""
        out: dict[str, Any] = {"field": self.field, "fields": list(self.fields), "revealed": self.revealed}
        if self.disclosed:
            out["disclosed"] = list(self.disclosed)
        if self.choice is not None:
            out["choice"] = self.choice.record()
        if self.cut:
            out["cut"] = True
        if self.source is not None:
            out["source"] = self.source
        return out


ChoicePath = Literal["picked", "confirmed", "none_fits", "declined", "forgot"]
"""How the shopper answered an offer: it picked an option, agreed to the recommended one, said none of the options
fits, turned the recommended one down, or did not have the deciding detail to hand (a persona that forgets)."""


@dataclass(frozen=True, slots=True)
class ChoiceRecord:
    """The shopper's choice on an offer, as the judge reads it from the turn's `choice` argument."""

    path: ChoicePath
    options: int
    """How many options the message offered (0 for a recommendation alone)."""
    option: int | None = None
    """The option chosen, counted from 1 in the order the message gave them; None when none was chosen or the
    recommendation was not one of the listed options."""
    recommended: bool = False
    """The message recommended one option, or asked to go ahead with one."""
    checked: tuple[str, ...] = ()
    """The hidden-spec fields whose need the options' words settled (met or missed), in spec order."""
    unmet: tuple[str, ...] = ()
    """The field of the need no option (or the recommended one) met, when the reply says none fits or declines."""
    named: tuple[str, ...] = ()
    """The fields whose value or need the reply's words gave: the value of the option it chose, or the need it named
    as unmet. Empty when the shopper kept the need to itself."""

    def record(self) -> dict[str, Any]:
        return {
            "path": self.path,
            "options": self.options,
            "option": self.option,
            "recommended": self.recommended,
            "checked": list(self.checked),
            "unmet": list(self.unmet),
            "named": list(self.named),
        }


@dataclass(frozen=True, slots=True)
class ConsentRequest:
    """The agent asks the shopper to approve an order: the total it states, and the store's own checkout as the agent
    platform shows it on its confirmation screen (read from the store's last reply, never from the agent)."""

    checkout_id: str
    total: Money
    """The total the agent stated, which is what the shopper approves paying."""
    checkout: Cart
    """The store's checkout: its lines, subtotal, total and currency."""
    message: str = ""
    """What the agent wrote to the shopper with the request."""


@dataclass(frozen=True, slots=True)
class ConsentReply:
    """The shopper's answer to an order approval. Decided by code from the hidden spec and the checkout."""

    text: str
    given: bool
    reason: ConsentReason
    version: str | None = None
    """The shopper's code and phrases version that decided it (D69)."""
    wording: WordingSource | None = None
    """Who worded the decision (D118, as ShopperReply.wording): the code's yes or no is `given`, whatever the words."""
    wording_model: str | None = None
    usage: Mapping[str, int] | None = None

    def origin(self) -> dict[str, Any]:
        """The turn's `origin` (D69): the consent rule and the shopper's version, and who worded it (D118)."""
        return _origin("consent", self.version, self.wording, self.wording_model)

    def turn_extras(self) -> dict[str, Any]:
        return _extras(self.wording_model, self.usage)

    def annotation(self, request: ConsentRequest) -> dict[str, Any]:
        """The `consent` the shopper's message turn carries, which the judge reads: what was approved or refused."""
        return {
            "given": self.given,
            "reason": self.reason,
            "checkout_id": request.checkout_id,
            "amount": request.total.amount,
            "currency": request.total.currency,
        }


class Shopper(Protocol):
    """The person the agent shops for. Every reply is decided from the hidden spec by code; a model may only word it
    (D118, carttrace.agents.wording), behind a check that the words keep exactly the decided facts."""

    def opening(self) -> str:
        """The first message, which states what the shopper wants."""
        ...

    async def reply(self, agent_message: str, *, cut: bool = False) -> ShopperReply:
        """The answer to a message the agent wrote to the shopper. `cut` says the message came from a turn the
        token limit cut, so the shopper does not choose from its fragment (D60)."""
        ...

    async def consent(self, request: ConsentRequest) -> ConsentReply:
        """The answer to a request to place an order for a stated total."""
        ...


@dataclass(frozen=True, slots=True)
class SupervisorVerdict:
    stop: bool
    """True ends the episode with supervisor_stop."""
    reason: str = ""
    decision_id: str | None = None
    """`<type id>@<evidence sha, 16 hex>` of the decision behind a stop, which joins it to the decision log."""


class Supervisor(Protocol):
    async def check(self, steps: Sequence[Mapping[str, Any]], cart: Mapping[str, Any] | None) -> SupervisorVerdict:
        """Called after every agent step with the steps so far (data/decisions/supervise.yaml says their shape) and
        the latest cart, or None before one exists."""
        ...


class AgentResult(BaseModel):
    """How an agent run ended. The runner turns it into the bundle's Outcome; the judge reads the bundle."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    end_reason: EndReason
    final_message: str = ""
    """The agent's last message to the shopper: the message it ended with, or its last one before a cap."""
    steps: int = Field(ge=0)
    tool_calls: int = Field(ge=0)
    """Calls the agent made to store tools (the harness's own done tool is not counted)."""
    usage: dict[str, int] = Field(default_factory=dict)
    """Token counts summed over every model turn; empty for the reference agent."""
    cart_id: str | None = None
    """The id of the cart the store returned last, when the agent made one."""
    declared_status: DeclaredStatus | None = None
    """What the agent said when it ended (cart_ready, no_match, cannot_continue); None when a cap or an error ended
    it first."""
    missing_details: dict[str, list[str]] = Field(default_factory=dict)
    """Detail name (size_chart, ingredients, warnings, ...) to the product ids whose data lacked it. The reference
    agent fills it; it is the store's side of a failure."""
    checkout_id: str | None = None
    """The id of the checkout the store returned last, on a checkout row."""
    order_id: str | None = None
    """The id of the order the store placed, as a completed checkout or get_order returned it."""
    model_unavailable: bool = False
    """The agent's model server was down when the run ended: no connection, no reply in time, or a 5xx after the
    inference retries. Such a harness_error is the infrastructure's, and the scheduler runs the row again once the
    server answers (mc-twc: one outage on 2026-10-12 turned 1,500 rows into harness_error in ten minutes)."""


class ProtocolAgent(Protocol):
    """A text-only agent that shops through protocol surfaces. Implementations: QwenProtocolAgent, ReferenceAgent."""

    async def run(
        self,
        task: TaskRef,
        surfaces: Sequence[ToolSurface],
        shopper: Shopper,
        sink: TurnSink,
        *,
        limits: Limits,
        supervisor: Supervisor | None = None,
    ) -> AgentResult: ...
