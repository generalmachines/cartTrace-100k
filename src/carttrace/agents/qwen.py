"""The protocol agent: Qwen3.8-27B in a text-only tool loop over a store's protocol surfaces.

Why this shape. It is the lab's protocol rung (lab/protocol.py, which ran on live stores through September 2026)
rebuilt on this repository's contracts: the tools are the union of the episode's surfaces (carttrace.agents.tools), the
model is the "agent" role on carttrace.inference's chat service with the prompt, temperature and token budget from
config.json, and the shopper is the user. A turn with tool calls runs them in order; a turn with only text is a
message to the shopper, whose reply comes back as the next user message; the harness's `done` tool ends it.

Every turn goes to the TurnSink as it happens: the model's reasoning as an agent note, its messages, each tool call
and result, the shopper's replies, and harness notes. The served model, token usage and latency of a model call are
stamped on the first turn that call produced, so the trace writer's token counts add up exactly once.

Checkout rows (D13, sandbox only). Given a wallet, the agent goes past the cart, and the harness adds the rail's own
wallet tools beside the store's (carttrace.agents.checkout, carttrace.agents.wallet_tools, D94), in whose official flow the shopper
approves the payment, and a 3-D Secure tool. The harness keeps the store's latest reply for each checkout and order,
which those tools read and which `done` checks. Which prompt it reads depends on the task (agent-v1.md stays the
prompt of cart rows; since D94 every checkout row reads one prompt, agent-v6.md since D108, which names no tool):
- a task whose spec has a buyer block (generator 2): data/prompts/agent-v6.md (agent-v3.md before D94), and nothing
  about the shopper in the system prompt. The shopper holds their address, email, phone and delivery preference, so
  the agent must ask for what the store's checkout reports missing, enter it through the store's own checkout tools,
  never invent a detail, and confirm the total before asking for approval;
- such a task given the shopper's buyer profile (D77, sandbox rows, agents.buyer_profile): data/prompts/agent-v6.md
  with the profile in its context, so the agent enters those details without asking;
- an older task: agent-v6.md too, with the persona's saved delivery details in the system prompt, as before.

An episode ends when:
- the agent calls done: cart_ready when it says the cart is ready and a cart exists, order_placed on a checkout row
  when it says so and the store's replies show a placed order, gave_up otherwise (the judge decides later whether a
  refusal was right);
- the step cap or the time cap (on the injected clock) is reached, checked before every model call;
- the supervisor says stop (a live decision type; shadow types never stop anything);
- a surface fails in a way no retry fixes: throttled or blocked results, or an exception from the surface;
- the model server fails (harness_error, never counted against the store or the agent).

The context budget (D82, agents.context_budget_tokens, carttrace.agents.compaction), an opt-in experiment that is off by
default and never set on rows we publish or benchmark: a call that would send more prompt
tokens than the budget is sent with the oldest tool results shortened to stubs, in the request only; a note turn says
which, and the trace keeps every result as the store returned it.

A call the harness refused (error_code harness_blocked) is never sent again with the same arguments: the agent gets
the refusal back at once, with a note, so a model that repeats itself cannot hammer a live store's allowlist.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import httpx

from carttrace.agents.checkout import CONSENT_TOOLS, HARNESS_SURFACE, HARNESS_TOOLS, CheckoutHarness
from carttrace.agents.compaction import (
    ContextGauge,
    ResultSource,
    compact,
    history_chars,
    tools_chars,
    view,
)
from carttrace.agents.dialects import Dialect, UcpDialect, dialect_for
from carttrace.agents.keywords import question_text
from carttrace.agents.profile import BuyerProfile
from carttrace.agents.service import (
    AgentResult,
    DeclaredStatus,
    DefinitionSink,
    Limits,
    Shopper,
    Supervisor,
    TurnSink,
)
from carttrace.agents.shopper import opening_origin
from carttrace.agents.tools import (
    DONE,
    FATAL_CODES,
    HIDDEN_ARGUMENTS,
    REFUSED,
    RESULT_CHARS,
    BoundTool,
    ToolIndex,
    done_tool,
    failure_reason,
    index_tools,
    result_text,
    step_record,
    summary,
    transport_arguments,
    with_transport,
)
from carttrace.agents.wallet import Wallet
from carttrace.config.schema import Settings
from carttrace.core.canon import dumps, sha256_hex
from carttrace.core.clock import Clock, SystemClock
from carttrace.core.errors import ConfigError, ExternalServiceError, MCError
from carttrace.core.ids import seeded_rng
from carttrace.inference.service import ChatResult, ChatService, ToolCall
from carttrace.protocols.tools import ToolResult, ToolSurface
from carttrace.trace.schema import Cart, EndReason, TaskRef

ROLE = "agent"
DECLARED: tuple[DeclaredStatus, ...] = (
    "cart_ready",
    "order_placed",
    "selection_ready",
    "no_match",
    "cannot_continue",
)
CHECKOUT_PROMPT = "agent-v6.md"
"""The checkout row's prompt for tasks without a buyer block, beside the cart row's under <paths.data>/prompts. A
prompt change is a code change."""
BUYER_PROMPT = "agent-v6.md"
"""The checkout row's prompt for tasks whose shopper holds the buyer's details: ask, never invent."""
PROFILE_PROMPT = "agent-v6.md"
"""The checkout row's prompt for a buyer task whose agent is given the shopper's buyer profile in its context (D77):
enter the profile's details without asking, and ask only for what it does not give."""
NUDGE = (
    "(A note from the shopping harness, not the shopper.) Your last turn was empty. Call a tool, write to the "
    "shopper, or call done."
)
CUT_NUDGE = (
    "(A note from the shopping harness, not the shopper.) Your last turn ran out of room before it acted. Think "
    "briefly, then call a tool, write to the shopper, or call done."
)
"""For a turn the token limit cut (finish_reason "length"): telling the model its turn was empty was false, and on
2026-10-13 every "empty" turn checked (89 of 89 on the website path, 539 of 539 in one UCP run) was such a cut."""
CUT_LIMIT = 4
"""Cut turns in a row before the runner ends the journey (D97). Each cut is recorded (the cut call's reasoning as the
agent's note, then a system note with `cut`, its count in a row) and the model is asked to continue, as D59 did once;
before D97 the second cut in a row ended the journey, which on 2026-10-16 ended 206 of 6,027 sandbox Qwen journeys
(3.4%) and the judge voids each as harness_error. A model that runs into the limit four times in a row is looping,
and the step and time caps bound it anyway."""
_SURFACE_FAILURES = (MCError, httpx.HTTPError, OSError, TimeoutError)
SEED_SPACE = 2**31
"""Per-call seeds are drawn below this, so every OpenAI-compatible server reads them as a plain integer."""


def _message_chars(message: Mapping[str, Any]) -> int:
    return len(str(message.get("content") or "")) + sum(len(str(call)) for call in message.get("tool_calls") or ())


def cap_history(
    messages: list[dict[str, Any]], cap_chars: int, *, keep_recent: int = 6
) -> tuple[list[dict[str, Any]], int]:
    """The history to send when agents.context_chars caps it, and how many tool results were omitted.

    The system prompt, the shopper's opening and the newest `keep_recent` messages are never cut; over the cap the
    oldest tool results (the catalog replies and pages that dominate the history) are replaced by a placeholder
    until the whole fits. The message list keeps its length and its tool_call pairing, so a capped call differs
    from an uncapped one only in old tool results' content. A cap of 0 returns the history untouched."""
    if cap_chars <= 0:
        return messages, 0
    over = sum(_message_chars(message) for message in messages) - cap_chars
    if over <= 0:
        return messages, 0
    capped = list(messages)
    stop = max(2, len(messages) - keep_recent)
    omitted = 0
    for i in range(2, stop):
        if over <= 0:
            break
        message = capped[i]
        content = str(message.get("content") or "")
        if message.get("role") != "tool" or not content:
            continue
        placeholder = (
            f"[an earlier tool result of {len(content)} chars was omitted to keep the conversation within "
            f"{cap_chars} chars; call the tool again if you need it]"
        )
        capped[i] = {**message, "content": placeholder}
        over -= len(content) - len(placeholder)
        omitted += 1
    return capped, omitted


def turn_seed(base: int | None, call: int) -> int | None:
    """The sampling seed of an episode's `call`-th model call (0 for the first), drawn from the episode's seed.

    Why per call and not one seed for the episode: vLLM seeds each request's sampler on its own, so one seed reused
    on every call samples every turn from the same random stream; a seed per call index keeps the turns independent
    while two episodes with the same base (the halves of a live and replica pair) still send the same seed at the
    same point of the conversation. None when the episode has no seed (the server samples freely)."""
    if base is None:
        return None
    return seeded_rng("agent.turn", base, call).randrange(SEED_SPACE)


class QwenProtocolAgent:
    """A ProtocolAgent whose choices come from the "agent" model role."""

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        chat: ChatService | None = None,
        clock: Clock | None = None,
        seed: int | None = None,
        context: str = "",
        prompt: str | None = None,
        result_chars: int = RESULT_CHARS,
        hidden_arguments: frozenset[str] = HIDDEN_ARGUMENTS,
        wallet: Wallet | None = None,
        profile: BuyerProfile | None = None,
        single_prompt: bool = False,
    ) -> None:
        """`chat` defaults to the service config.json chooses (pass one to share it across agents); `context` is
        appended to the system prompt (the store and the shopper's market, say); `prompt` replaces the prompt file
        named by agents.prompt, for experiments. A `wallet` makes the episode a checkout row: the checkout prompt,
        and the harness's consent and wallet tools. A `profile` (D77, sandbox checkout rows only) is the shopper's buyer
        profile: a buyer task then gets agent-v6.md and the profile in its context. `single_prompt` (sandbox v1 rows,
        D98) gives a cart row agent-v6.md too, so cart and order rows read one prompt and only the context's stopping
        point differs."""
        resolved = settings if settings is not None else _process_settings()
        self.temperature = resolved.agents.temperature
        self.transport = transport_arguments(resolved.live.agent_profile_url)
        """Hidden arguments the harness adds where a tool's schema declares them (UCP's meta)."""
        self.wallet = wallet
        self.profile = profile if wallet is not None else None
        """The shopper's buyer profile the agent is given on a checkout row (D77); None asks for every detail."""
        self.buyer_prompt: str | None = None
        """The checkout prompt for a task whose shopper holds the buyer's details; None off a checkout row."""
        self.prompt_file: str | None = None
        """The prompt's file relative to paths.data (prompts/agent-v1.md), for the trace; None for a given prompt."""
        self.buyer_prompt_file: str | None = None
        if prompt is not None:
            self.prompt = prompt
            self.buyer_prompt = prompt if wallet is not None else None
        elif wallet is not None:
            self.prompt = read_prompt(checkout_prompt_path(resolved))
            buyer_path = profile_prompt_path(resolved) if self.profile is not None else buyer_prompt_path(resolved)
            self.buyer_prompt = read_prompt(buyer_path)
            self.prompt_file = _data_relative(checkout_prompt_path(resolved), resolved)
            self.buyer_prompt_file = _data_relative(buyer_path, resolved)
        elif single_prompt:
            self.prompt = read_prompt(checkout_prompt_path(resolved))
            self.prompt_file = _data_relative(checkout_prompt_path(resolved), resolved)
        else:
            self.prompt = read_prompt(resolved.agents.prompt)
            self.prompt_file = _data_relative(Path(resolved.agents.prompt), resolved)
        self.own_gpu = _own_gpu(resolved)
        """The agent role's endpoint is one of our own model servers (no API key): its calls cost GPU time only."""
        self.context_chars = resolved.agents.context_chars
        """Experiments: the history cap from agents.context_chars; 0 sends the full history, as before."""
        self.context_budget = resolved.agents.context_budget_tokens
        """The most prompt tokens one call may send before older results are stubbed in its view (D82); 0 is off."""
        self.context = context.strip()
        self.seed = seed
        self.result_chars = result_chars
        self.hidden_arguments = hidden_arguments
        self._chat = chat if chat is not None else _chat_service(settings)
        self.clock: Clock = clock or SystemClock()

    @property
    def system_prompt(self) -> str:
        """The system prompt of a task without a buyer block (every cart row, and older checkout rows)."""
        return self.system_prompt_for(None)

    def system_prompt_for(self, task: TaskRef | None) -> str:
        """The system prompt for `task`. On a checkout row a task whose shopper holds the buyer's details gets the
        buyer prompt and no saved details; any other task gets the persona's saved details in its context."""
        if self.wallet is not None and self.buyer_prompt is not None and task is not None and _holds_buyer(task):
            prompt, brief = self.buyer_prompt, self.profile.text() if self.profile is not None else ""
        else:
            prompt = self.prompt
            brief = CheckoutHarness.brief(self.wallet) if self.wallet is not None else ""
        context = "\n".join(part for part in (self.context, brief) if part)
        return f"{prompt}\n\nContext:\n{context}" if context else prompt

    def prompt_source_for(self, task: TaskRef | None) -> tuple[str, str | None]:
        """The prompt text `task` is sent before its context, and the file it came from (None for a given prompt)."""
        if self.wallet is not None and self.buyer_prompt is not None and task is not None and _holds_buyer(task):
            return self.buyer_prompt, self.buyer_prompt_file
        return self.prompt, self.prompt_file

    @property
    def prompt_sha(self) -> str:
        """sha256 of the system prompt as sent to a task without a buyer block, for the header's model parameters."""
        return sha256_hex(self.system_prompt)

    def model_params(self, limits: Limits, task: TaskRef | None = None) -> dict[str, Any]:
        """What the runner records in the header's ModelRef.params; the prompt's sha is the one `task` is sent."""
        params = {
            "temperature": self.temperature,
            "max_tokens": limits.max_tokens,
            "seed": self.seed,
            "prompt_sha": sha256_hex(self.system_prompt_for(task))[:16],
        }
        if self.context_chars > 0:
            params["context_chars"] = self.context_chars
        if self.context_budget > 0:
            params["context_budget_tokens"] = self.context_budget
        return params

    async def run(
        self,
        task: TaskRef,
        surfaces: Sequence[ToolSurface],
        shopper: Shopper,
        sink: TurnSink,
        *,
        limits: Limits,
        supervisor: Supervisor | None = None,
    ) -> AgentResult:
        episode = _Episode(
            self, shopper, sink, limits, supervisor, system_prompt=self.system_prompt_for(task), task=task
        )
        return await episode.run(surfaces)

    async def _ask(
        self,
        messages: Sequence[dict[str, Any]],
        tools: Sequence[dict[str, Any]],
        limits: Limits,
        *,
        seed: int | None = None,
    ) -> ChatResult:
        return await self._chat.chat(
            ROLE, messages, tools, max_tokens=limits.max_tokens, temperature=self.temperature, seed=seed
        )


def checkout_prompt_path(settings: Settings) -> Path:
    """Where the checkout row's prompt lives: <paths.data>/prompts/agent-v6.md."""
    return settings.paths.data / "prompts" / CHECKOUT_PROMPT


def profile_prompt_path(settings: Settings) -> Path:
    """Where the prompt of a buyer task given the buyer profile lives: <paths.data>/prompts/agent-v6.md."""
    return settings.paths.data / "prompts" / PROFILE_PROMPT


def buyer_prompt_path(settings: Settings) -> Path:
    """Where the checkout row's prompt for buyer-block tasks lives: <paths.data>/prompts/agent-v6.md."""
    return settings.paths.data / "prompts" / BUYER_PROMPT


def _data_relative(path: Path, settings: Settings) -> str:
    """A prompt file's name as the trace records it: relative to paths.data when it is under it, else as given."""
    try:
        return path.resolve().relative_to(settings.paths.data.resolve()).as_posix()
    except ValueError:
        return path.as_posix()


def _own_gpu(settings: Settings) -> bool:
    """Whether the agent role runs on our own model servers: an endpoint with no API key is ours."""
    try:
        return settings.role(ROLE).api_key_env is None
    except ConfigError:
        return True


def _holds_buyer(task: TaskRef) -> bool:
    """The task's shopper holds the buyer's checkout details (its hidden spec has a buyer block)."""
    return isinstance(task.hidden_spec.get("buyer"), dict)


def read_prompt(path: Path) -> str:
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        raise ConfigError(
            f"no agent prompt at {path}", hint="set agents.prompt in config.json, or restore it"
        ) from None
    if not text:
        raise ConfigError(f"the agent prompt {path} is empty", hint="restore it from git")
    return text


def _process_settings() -> Settings:
    from carttrace.config import settings

    return settings()


def _chat_service(settings: Settings | None) -> ChatService:
    """The process's shared chat service when `settings` are the process's own (as the runner passes them), so every
    episode shares one HTTP client and connection pool; a service of its own for other settings."""
    from carttrace.config import settings as process_settings
    from carttrace.inference import chat_service

    if settings is None:
        return chat_service()
    try:
        own = settings is process_settings()
    except ConfigError:
        own = False
    return chat_service() if own else chat_service(settings)


class _Stamp:
    """What one model call says about the turns it produced. Every turn gets the model, the seed and the finish reason
    (D69); the first also gets the usage, latency and cost, so the trace writer's sums count each call once, and the
    served model, which fills the header."""

    def __init__(self, result: ChatResult, seed: int | None = None, *, own_gpu: bool = True) -> None:
        self.model = result.served_model or result.requested_model
        self.served = result.served_model
        self.usage = dict(result.usage) or None
        self.latency_ms = result.latency_ms
        self.seed = seed
        self.finish_reason = result.finish_reason
        self.cost = call_cost(result, own_gpu=own_gpu)
        self.used = False

    def take(self) -> dict[str, Any]:
        every: dict[str, Any] = {"model": self.model}
        if self.seed is not None:
            every["seed"] = self.seed
        if self.finish_reason is not None:
            every["finish_reason"] = self.finish_reason
        if self.used:
            return every
        self.used = True
        first = {**every, "usage": self.usage, "latency_ms": self.latency_ms, "cost": self.cost}
        return first | ({"served": self.served} if self.served else {})


def call_cost(result: ChatResult, *, own_gpu: bool) -> dict[str, Any]:
    """A model call's cost (D69): nothing in money on our own GPU servers, with the time the call held its slot as
    the GPU-time proxy; unpriced on an endpoint we pay for, which reports no cost to a chat call."""
    if own_gpu:
        return {"usd_micros": 0, "gpu_ms": result.latency_ms, "basis": "own_gpu"}
    return {"usd_micros": None, "gpu_ms": None, "basis": "unpriced"}


class _Ended(Exception):
    """Raised inside an episode to end it; carries the result."""

    def __init__(self, result: AgentResult) -> None:
        super().__init__(result.end_reason)
        self.result = result


class _Episode:
    def __init__(
        self,
        agent: QwenProtocolAgent,
        shopper: Shopper,
        sink: TurnSink,
        limits: Limits,
        supervisor: Supervisor | None,
        *,
        system_prompt: str,
        task: TaskRef | None = None,
    ) -> None:
        self.agent = agent
        self.system_prompt = system_prompt
        self.task = task
        self.shopper = shopper
        self.sink = sink
        self.limits = limits
        self.supervisor = supervisor
        self.clock = agent.clock
        self.started = self.clock.monotonic()
        self.queued = 0.0
        """Seconds the episode's model calls waited for a GPU slot, which the time cap does not count: the cap stops
        an agent that wanders, not one the shared GPUs keep waiting. On 2026-10-15 time_cap ended 18% of sandbox
        episodes, most after 9 to 14 tool calls and 160 to 290 s of model time in 900 s of wall time."""
        self.messages: list[dict[str, Any]] = []
        self.usage: Counter[str] = Counter()
        self.steps = 0
        self.tool_calls = 0
        self.cart: Cart | None = None
        self.cart_id: str | None = None
        self.refused: dict[tuple[str, str], ToolResult] = {}
        self.records: list[dict[str, Any]] = []
        self.last_message = ""
        self.held = 0
        self.cart_held = 0
        self.harness = (
            CheckoutHarness(agent.wallet, shopper, sink, clock=agent.clock) if agent.wallet is not None else None
        )
        self.sources: dict[int, ResultSource] = {}
        """The call and full result behind each tool message of the history, by its index (D82)."""
        self.stubs: dict[int, str] = {}
        """The tool messages the context budget stubbed, by index, and their stubs; a stub stays a stub (D82)."""
        self.gauge = ContextGauge()
        self.over_budget = False
        """The budget was passed with nothing left to stub, and a note said so; it is said once until a stub is made."""
        self.dialect: Dialect = UcpDialect()
        """How the store's replies are read (which results are carts, how the harness reads a checkout): the dialect
        the listed tools offer, once they are listed. The loop itself is the same on every protocol."""

    async def run(self, surfaces: Sequence[ToolSurface]) -> AgentResult:
        try:
            return await self._loop(surfaces)
        except _Ended as ended:
            return ended.result

    async def _loop(self, surfaces: Sequence[ToolSurface]) -> AgentResult:
        index = await self._index(surfaces)
        self.dialect = dialect_for({t.name: t.spec for t in index.tools})
        tools = index.definitions(done=False)
        if self.harness is not None:
            self.harness.speak(self.dialect, store=self._store_call(index))
            hidden = {t.exposed for t in index.tools if self.harness.stands_in(t.name)}  # D94: the wallet's own
            tools = [t for t in tools if t["function"]["name"] not in hidden] + self.harness.definitions()
        tools.append(done_tool(checkout=self.harness is not None, selection=self.dialect.boundary == "selection"))
        self._define(index, tools)
        opening = self.shopper.opening()
        origin = opening_origin(self.task) if self.task is not None else None
        self.sink.turn("shopper", "message", text=opening, origin=origin, **self._profile_given())
        self.messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": opening},
        ]
        nudged = False
        cuts = 0
        while True:
            self._check_caps()
            seed = turn_seed(self.agent.seed, self.steps)
            result = await self._model(tools, seed)
            stamp = _Stamp(result, seed, own_gpu=self.agent.own_gpu)
            if result.reasoning:
                self.sink.turn("agent", "note", text=result.reasoning, **stamp.take())
            text = result.text.strip()
            if result.tool_calls or text:
                cuts = 0
            if result.tool_calls:
                nudged = False
                if text:
                    self._agent_message(text, stamp)
                self.messages.append(_assistant(text, result.tool_calls))
                await self._tool_turn(result, text, index, stamp)
            elif text:
                nudged = False
                self._agent_message(text, stamp)
                self.messages.append({"role": "assistant", "content": text})
                await self._shopper_turn(text, cut=self._cut(result))
            else:
                cuts, nudged = self._empty_turn(result, stamp, cuts, nudged)
            await self._supervise()

    def _empty_turn(self, result: ChatResult, stamp: _Stamp, cuts: int, nudged: bool) -> tuple[int, bool]:
        """A turn with no tool call and no text: a cut one is continued until CUT_LIMIT cuts in a row (D97), an empty
        one once. Returns the cuts in a row and whether the last turn was an empty one nudged."""
        if self._cut(result):
            cuts += 1
            counted = {"cut": cuts, "max_tokens": self.limits.max_tokens}
            if cuts >= CUT_LIMIT:
                self._note(
                    f"the model's turn was cut at the token limit {cuts} times in a row", stamp, arguments=counted
                )
                raise _Ended(self._result("gave_up"))
            self._note(
                "the model's turn was cut at the token limit before it acted; it was asked to continue",
                stamp,
                arguments=counted,
            )
            self.messages += [{"role": "assistant", "content": ""}, {"role": "user", "content": CUT_NUDGE}]
            return cuts, False
        if nudged:
            self._note("the model returned two empty turns in a row", stamp)
            raise _Ended(self._result("gave_up"))
        self._note("the model returned an empty turn; it was asked to continue", stamp)
        self.messages += [{"role": "assistant", "content": ""}, {"role": "user", "content": NUDGE}]
        return 0, True

    def _profile_given(self) -> dict[str, Any]:
        """The opening's annotation when the agent was given the shopper's buyer profile (D77): the details it holds
        count as given from the start, as the judge reads a shopper turn's `disclosed`, so entering them is not
        inventing them."""
        profile = self.agent.profile
        if profile is None or self.task is None or not _holds_buyer(self.task):
            return {}
        return {"arguments": {"disclosed": list(profile.fields()), "profile": profile.source}}

    async def _tool_turn(self, result: ChatResult, text: str, index: ToolIndex, stamp: _Stamp) -> None:
        """A turn with tool calls: done held for a question beside it, or for a cart on a checkout row; else the
        calls run, and a question written beside them goes to the shopper after their results."""
        calls = result.tool_calls
        asked = self._asked_before_done(text, calls)
        if asked:
            await self._hold_done(calls, index, stamp, asked, cut=self._cut(result))
            return
        if self._cart_before_order(calls):
            await self._hold_cart(calls, index, stamp, text)
            return
        await self._calls(calls, index, stamp)
        beside = self._asked_beside(text, calls)
        if beside:
            self._note("the agent asked the shopper something beside its tool calls; the shopper answers")
            await self._shopper_turn(beside, cut=self._cut(result))

    def _define(self, index: ToolIndex, tools: Sequence[dict[str, Any]]) -> None:
        """Record what the agent was given (D69), once, on a sink that keeps it: the system prompt as sent, its file
        and sha256, and every tool definition as the model receives it, with what serves each one."""
        if not isinstance(self.sink, DefinitionSink):
            return
        prompt, prompt_file = self.agent.prompt_source_for(self.task)
        surfaces = {t.exposed: str(t.surface.kind) for t in index.tools}
        names = [str((tool.get("function") or {}).get("name", "")) for tool in tools]
        self.sink.agent_definition(
            {
                "name": "qwen-protocol",
                "role": ROLE,
                "system_prompt": self.system_prompt,
                "system_prompt_sha256": sha256_hex(self.system_prompt),
                "prompt_file": prompt_file,
                "prompt_sha256": sha256_hex(prompt),
                "tool_definitions": list(tools),
                "tool_surfaces": {
                    name: surfaces.get(name, DONE if name == DONE else HARNESS_SURFACE) for name in names if name
                },
                "params": {
                    "temperature": self.agent.temperature,
                    "max_tokens": self.limits.max_tokens,
                    "seed": self.agent.seed,
                }
                | ({"context_budget_tokens": self.agent.context_budget} if self.agent.context_budget > 0 else {}),
            }
        )

    async def _index(self, surfaces: Sequence[ToolSurface]) -> ToolIndex:
        reserved = frozenset({DONE, *HARNESS_TOOLS}) if self.harness is not None else frozenset({DONE})
        try:
            index = await index_tools(surfaces, hidden=self.agent.hidden_arguments, reserved=reserved)
        except _SURFACE_FAILURES as exc:
            self._note(f"listing the surfaces' tools failed: {exc}")
            raise _Ended(self._result("harness_error")) from None
        if not index.tools:
            self._note("no surface listed a tool")
            raise _Ended(self._result("no_protocol"))
        return index

    def _check_caps(self) -> None:
        if self.steps >= self.limits.step_cap:
            self._note(f"step cap reached after {self.steps} steps")
            raise _Ended(self._result("step_cap"))
        elapsed = self.clock.monotonic() - self.started - self.queued
        if elapsed >= self.limits.time_cap_s:
            self._note(f"time cap reached after {elapsed:.0f} s, not counting {self.queued:.0f} s queued for the GPU")
            raise _Ended(self._result("time_cap"))

    async def _model(self, tools: Sequence[dict[str, Any]], seed: int | None) -> ChatResult:
        self._budget(tools)
        sent = view(self.messages, self.stubs)
        messages, omitted = cap_history(sent, self.agent.context_chars)
        if omitted:
            self._note(
                f"the context cap of {self.agent.context_chars} chars omitted {omitted} older tool results "
                "from this call"
            )
        asked = self.clock.monotonic()
        try:
            result = await self.agent._ask(messages, tools, self.limits, seed=seed)
        except ExternalServiceError as exc:
            self._note(f"the model call failed: {exc}")
            raise _Ended(self._result("harness_error", model_unavailable=exc.unavailable)) from None
        # the call's wall time beyond its model time was spent waiting for a GPU slot (ChatResult.latency_ms)
        self.queued += max(0.0, self.clock.monotonic() - asked - result.latency_ms / 1000)
        self.steps += 1
        self.usage.update(result.usage)
        self.gauge.measured(history_chars(messages), result.usage.get("prompt_tokens"))
        return result

    def _budget(self, tools: Sequence[dict[str, Any]]) -> None:
        """Keep the next call within agents.context_budget_tokens (D82): over it, the oldest tool results are
        stubbed in the model's view, and a note says which and why. The trace keeps every result in full."""
        budget = self.agent.context_budget
        if budget <= 0:
            return
        fixed = tools_chars(tools)
        done = compact(
            self.messages,
            self.sources,
            self.stubs,
            budget=budget,
            estimate=lambda messages: self.gauge.estimate(history_chars(messages), fixed),
        )
        if done is None or (not done.stubbed and self.over_budget):
            return
        self.over_budget = not done.stubbed
        names = Counter(self.sources[i].tool for i in done.stubbed)
        listed = ", ".join(f"{name} x{n}" if n > 1 else name for name, n in sorted(names.items()))
        text = f"the next call would send about {done.before} prompt tokens, over the context budget of {budget}; " + (
            f"{len(done.stubbed)} older tool results ({listed}) are shortened to stubs in the model's view, "
            f"to about {done.after} tokens. The trace keeps them in full"
            if done.stubbed
            else "no older tool result was left to shorten, so the call goes as it is"
        )
        stubbed = [
            {"index": i, "tool": self.sources[i].tool, "tool_call_id": self.messages[i].get("tool_call_id")}
            for i in done.stubbed
        ]
        self.sink.turn(
            "system",
            "note",
            text=text,
            arguments={
                "context_budget_tokens": budget,
                "estimated_tokens": done.before,
                "estimated_after": done.after,
                "stubbed": stubbed,
                "stubbed_total": len(self.stubs),
            },
        )

    def _agent_message(self, text: str, stamp: _Stamp) -> None:
        self.sink.turn("agent", "message", text=text, **stamp.take())
        self.records.append({"text": text[:500]})
        self.last_message = text

    def _cut(self, result: ChatResult) -> bool:
        """The token limit cut this turn: the model said so, or it spent the whole budget (D60)."""
        spent = result.usage.get("completion_tokens", 0)
        return result.finish_reason == "length" or spent >= self.limits.max_tokens

    async def _shopper_turn(self, text: str, *, cut: bool = False) -> None:
        reply = await self.shopper.reply(text, cut=cut)
        self.sink.turn(
            "shopper",
            "message",
            text=reply.text,
            arguments=reply.annotation(),
            origin=reply.origin(),
            **reply.turn_extras(),
        )
        self.messages.append({"role": "user", "content": reply.text})

    def _asked_before_done(self, text: str, calls: Sequence[ToolCall]) -> str:
        """What the agent asked the shopper in a turn that also calls done, or "" when it asked nothing or has been held
        HOLDS times already. A turn with tool calls never reaches the shopper, so a question written beside done went
        unanswered and the episode ended without the detail it asked for (2026-10-15: 137 of 876 sandbox give-ups and
        37 of 152 carts that never paid ended on "What's your email address?" plus done). An order the store has placed
        is never held: nothing is left to ask for, and on the v1 pilot (2026-10-16, ep-86d95a250990) "Pick it up at the
        store counter." beside done(order_placed) was read as a question, held twice and answered "I'll collect it from
        the store." each time, a reply to a question nobody asked."""
        done = next((c for c in calls if c.name == DONE), None)
        if done is None or self.held >= HOLDS:
            return ""
        if self.harness is not None and self.harness.order_id is not None:
            return ""
        message = done.arguments.get("message")
        return question_text(text) or (question_text(message) if isinstance(message, str) else "")

    async def _hold_done(
        self, calls: Sequence[ToolCall], index: ToolIndex, stamp: _Stamp, asked: str, *, cut: bool
    ) -> None:
        """Run the calls before done, answer done and the calls after it with a note that it was held, then hand the
        question to the shopper, whose reply comes back as the next user message."""
        self.held += 1
        await self._held(
            calls,
            index,
            stamp,
            "done was held: the agent asked the shopper something in the same turn, so the shopper answers first",
            HELD_DONE,
        )
        await self._shopper_turn(asked, cut=cut)

    async def _held(self, calls: Sequence[ToolCall], index: ToolIndex, stamp: _Stamp, note: str, answer: str) -> None:
        """Run the calls before done, record done with `note`, and answer done (with `answer`) and the calls after it
        without running them."""
        at = next(i for i, c in enumerate(calls) if c.name == DONE)
        for call in calls[:at]:
            if self.harness is not None and self.harness.handles(call.name):
                await self._harness_call(self.harness, call, stamp)
            else:
                await self._call(call, index, stamp)
        self.sink.turn("agent", "tool_call", tool=DONE, arguments=calls[at].arguments, **stamp.take())
        self._note(note)
        for call in calls[at:]:
            held = answer if call.name == DONE else "not run: done came before it in the same turn"
            self.messages.append({"role": "tool", "tool_call_id": call.id, "content": held})

    def _cart_before_order(self, calls: Sequence[ToolCall]) -> bool:
        """done(cart_ready) on a checkout row with no order yet, not held before: the journey ends with an order, so
        the shopper hears where it stopped. On 2026-10-15, 38 of 40 checkout rows that ended at cart_ready ended
        waiting for an email or address the shopper was never asked for in a message that reached them."""
        done = next((c for c in calls if c.name == DONE), None)
        if done is None or self.harness is None or self.cart_held >= CART_HOLDS or self.harness.order_id is not None:
            return False
        return done.arguments.get("status") == "cart_ready" and self.dialect.boundary != "selection"

    async def _hold_cart(self, calls: Sequence[ToolCall], index: ToolIndex, stamp: _Stamp, text: str) -> None:
        """Hold done(cart_ready) once on a checkout row: the shopper reads the agent's last words with the cart in
        front of them, answers what they ask, or asks for the order when the cart holds what they want, or says
        what is wrong with it."""
        self.cart_held += 1
        await self._held(
            calls,
            index,
            stamp,
            "done(cart_ready) was held: this journey ends with an order, so the shopper answers",
            HELD_CART,
        )
        done = next(c for c in calls if c.name == DONE)
        message = done.arguments.get("message")
        said = message.strip() if isinstance(message, str) and message.strip() else text
        if said and said != text:
            self.sink.turn("agent", "message", text=said)
            self.last_message = said
        cart = self.harness.checkout_view() if self.harness is not None else None
        go_ahead = getattr(self.shopper, "go_ahead", None)
        if go_ahead is None:
            await self._shopper_turn(said)
            return
        reply = await go_ahead(said, cart or self.cart)
        self.sink.turn(
            "shopper",
            "message",
            text=reply.text,
            arguments=reply.annotation(),
            origin=reply.origin(),
            **reply.turn_extras(),
        )
        self.messages.append({"role": "user", "content": reply.text})

    def _asked_beside(self, text: str, calls: Sequence[ToolCall]) -> str:
        """What the agent asked the shopper in a turn whose tool calls ran, or "". A turn with tool calls never reached
        the shopper, so its question went unanswered unless asked again (2026-10-15 after 15:00: 58 of 2,101 such
        turns asked something, 34 of them the email, address or delivery a checkout needed). A turn asking for the
        order's approval is answered by the wallet step itself (D94)."""
        if any(c.name == DONE or c.name in CONSENT_TOOLS for c in calls):
            return ""
        return question_text(text)

    async def _calls(self, calls: Sequence[ToolCall], index: ToolIndex, stamp: _Stamp) -> None:
        for position, call in enumerate(calls):
            if call.name == DONE:
                self.sink.turn("agent", "tool_call", tool=DONE, arguments=call.arguments, **stamp.take())
                if position + 1 < len(calls):
                    self._note(f"{len(calls) - position - 1} tool calls after done were not run")
                raise _Ended(self._done(call.arguments, stamp))
            if self.harness is not None and self.harness.handles(call.name):
                await self._harness_call(self.harness, call, stamp)
            else:
                await self._call(call, index, stamp)

    async def _call(self, call: ToolCall, index: ToolIndex, stamp: _Stamp) -> None:
        bound = index.get(call.name)
        tool = bound.name if bound else call.name
        surface = str(bound.surface.kind) if bound else None
        self.sink.turn("agent", "tool_call", tool=tool, surface=surface, arguments=call.arguments, **stamp.take())
        self.tool_calls += 1
        if call.arguments_error is not None:
            result = ToolResult(
                content=f"your arguments could not be read ({call.arguments_error}); send them as one JSON object",
                is_error=True,
                error_code="invalid_arguments",
            )
        elif bound is None:
            names = ", ".join(t.exposed for t in index.tools)
            result = ToolResult(
                content=f"there is no tool named {call.name}; the tools are {names}, and done",
                is_error=True,
                error_code="unknown_tool",
            )
        else:
            result = await self._send(bound, call.arguments)
        self.sink.turn(
            "tool",
            "tool_result",
            tool=tool,
            surface=surface,
            result=result.content,
            is_error=result.is_error,
            error_code=result.error_code,
            latency_ms=result.latency_ms,
        )
        self.records.append(step_record(tool, call.arguments, result))
        self._tool_message(call, tool, result)
        if self.harness is not None and bound is not None:
            self.harness.observe(bound.name, result)
        cart = self.dialect.cart_in(result) if self.dialect.is_cart_tool(tool) else None
        if cart is not None:
            self.cart = cart
            self.cart_id = cart.cart_id or self.cart_id
        reason = FATAL_CODES.get(result.error_code or "")
        if reason is not None:
            self._note(f"{tool} answered {result.error_code}; the episode cannot go on")
            raise _Ended(self._result(reason))

    async def _harness_call(self, harness: CheckoutHarness, call: ToolCall, stamp: _Stamp) -> None:
        """One of the checkout row's harness tools: the shopper's approval, or the wallet. Never a store call. A
        paying step's call is written once the shopper has answered the wallet's sheet (`CheckoutHarness.call`)."""
        taken = stamp.take()

        def write_call() -> None:
            self.sink.turn(
                "agent", "tool_call", tool=call.name, surface=HARNESS_SURFACE, arguments=call.arguments, **taken
            )

        if call.arguments_error is not None:
            write_call()
            result = ToolResult(
                content=f"your arguments could not be read ({call.arguments_error}); send them as one JSON object",
                is_error=True,
                error_code="invalid_arguments",
            )
        else:
            result = await harness.call(call.name, call.arguments, write_call=write_call)
            message = call.arguments.get("message")
            if call.name in CONSENT_TOOLS and isinstance(message, str) and message.strip():
                self.records.append({"text": message.strip()[:500]})
                self.last_message = message.strip()
        self.sink.turn(
            "tool",
            "tool_result",
            tool=call.name,
            surface=HARNESS_SURFACE,
            result=result.content,
            is_error=result.is_error,
            error_code=result.error_code,
        )
        self.records.append(step_record(call.name, call.arguments, result))
        self._tool_message(call, call.name, result)
        if harness.failed is not None:  # the wallet failed, not the agent (the v1 review's #9)
            self._note(f"the wallet failed, not the agent: {harness.failed}")
            raise _Ended(self._result("harness_error"))

    def _tool_message(self, call: ToolCall, tool: str, result: ToolResult) -> None:
        """Append a result to the history as the model reads it, remembering its call and full content so the
        context budget can stub it later (D82)."""
        self.sources[len(self.messages)] = ResultSource(
            tool=tool, arguments=call.arguments, content=result.content, is_error=result.is_error
        )
        self.messages.append(
            {"role": "tool", "tool_call_id": call.id, "content": result_text(result, limit=self.agent.result_chars)}
        )

    def _store_call(self, index: ToolIndex) -> Any:
        """How the harness reaches a store tool for the wallet (ACP's delegate_payment): the call and its result are
        written to the trace as the agent's own calls are, after a note saying the wallet asked for it."""

        async def send(name: str, arguments: dict[str, Any]) -> ToolResult:
            bound = next((t for t in index.tools if t.name == name), None)  # by the store's name: D94 renames it
            if bound is None:
                return ToolResult(content=f"no surface offers {name}", is_error=True, error_code="protocol_error")
            self._note(f"the wallet vaults its credential at the store's payment processor ({name})")
            surface = str(bound.surface.kind)
            self.sink.turn("agent", "tool_call", tool=bound.name, surface=surface, arguments=arguments)
            self.tool_calls += 1
            filled = with_transport(bound.spec, arguments, self.agent.transport)
            try:
                result = await bound.surface.call(bound.name, filled)
            except _SURFACE_FAILURES as exc:
                self._note(f"{name} failed: {exc}")
                raise _Ended(self._result(failure_reason(exc))) from None
            self.sink.turn(
                "tool",
                "tool_result",
                tool=bound.name,
                surface=surface,
                result=result.content,
                is_error=result.is_error,
                error_code=result.error_code,
                latency_ms=result.latency_ms,
            )
            return result

        return send

    async def _send(self, bound: BoundTool, arguments: dict[str, Any]) -> ToolResult:
        name = bound.name
        key = (bound.exposed, dumps(arguments))
        refused = self.refused.get(key)
        if refused is not None:
            self._note(f"{name} was refused before with these arguments; it was not sent again")
            content = (
                f"not sent: this exact call was refused before ({result_text(refused, limit=300)}). It will be "
                "refused again; choose another way, or tell the shopper what the limit is"
            )
            return ToolResult(content=content, is_error=True, error_code=REFUSED)
        guarded = self.harness.guard(name, arguments) if self.harness is not None else None
        if guarded is not None:
            self._note(f"{name} would pay without a credential the shopper approved; it was not sent")
            return guarded
        try:
            result = await bound.surface.call(name, with_transport(bound.spec, arguments, self.agent.transport))
        except _SURFACE_FAILURES as exc:
            self._note(f"{name} failed: {exc}")
            raise _Ended(self._result(failure_reason(exc))) from None
        if result.error_code == REFUSED:
            self.refused[key] = result
        return result

    def _done(self, arguments: Mapping[str, Any], stamp: _Stamp) -> AgentResult:
        status = arguments.get("status")
        declared = status if status in DECLARED else None
        message = arguments.get("message")
        final = message.strip() if isinstance(message, str) else ""
        if final:
            self.sink.turn("agent", "message", text=final, **stamp.take())
            self.last_message = final
        given = arguments.get("cart_id") or arguments.get("selection_id")
        cart_id = self.cart_id or (given if isinstance(given, str) and given else None)
        if declared == "order_placed":
            return self._ordered(declared, cart_id)
        if self.dialect.boundary == "selection":
            return self._selected(declared, cart_id)
        if declared == "cart_ready" and cart_id is None:
            self._note("the agent said the cart was ready, but no cart was made")
        reason: EndReason = "cart_ready" if declared == "cart_ready" and cart_id is not None else "gave_up"
        return self._result(reason, declared=declared, cart_id=cart_id)

    def _selected(self, declared: DeclaredStatus | None, selection_id: str | None) -> AgentResult:
        """done on a store's website: selection_ready only when the agent says so (or says its cart is ready) and
        choose_variant made a selection; the harness reads it back and decides whether it holds."""
        if declared in ("selection_ready", "cart_ready") and self.cart is not None and self.cart.lines:
            return self._result("selection_ready", declared="selection_ready", cart_id=selection_id)
        if declared in ("selection_ready", "cart_ready"):
            self._note("the agent said its selection was ready, but no variant was chosen")
        return self._result("gave_up", declared=declared, cart_id=selection_id)

    def _ordered(self, declared: DeclaredStatus, cart_id: str | None) -> AgentResult:
        """done(order_placed) ends with an order only on a checkout row whose store replies show one."""
        if self.harness is None:
            self._note("the agent said an order was placed, but this journey ends at the cart")
            return self._result("gave_up", declared=declared, cart_id=cart_id)
        if self.harness.order_id is None:
            self._note("the agent said an order was placed, but no reply from the store shows one")
            return self._result("gave_up", declared=declared, cart_id=cart_id)
        return self._result("order_placed", declared=declared, cart_id=cart_id)

    async def _supervise(self) -> None:
        if self.supervisor is None:
            return
        verdict = await self.supervisor.check(self.records, summary(self.cart))
        if verdict.stop:
            self._note(f"the supervisor stopped the episode: {verdict.reason} ({verdict.decision_id})")
            raise _Ended(self._result("supervisor_stop"))

    def _note(self, text: str, stamp: _Stamp | None = None, *, arguments: dict[str, Any] | None = None) -> None:
        extra: dict[str, Any] = {"arguments": arguments} if arguments else {}
        if stamp is not None and not stamp.used:
            self.sink.turn("agent", "note", text=text, **stamp.take(), **extra)
        else:
            self.sink.turn("system", "note", text=text, **extra)

    def _result(
        self,
        reason: EndReason,
        *,
        declared: DeclaredStatus | None = None,
        cart_id: str | None = None,
        model_unavailable: bool = False,
    ) -> AgentResult:
        return AgentResult(
            end_reason=reason,
            final_message=self.last_message,
            steps=self.steps,
            tool_calls=self.tool_calls,
            usage=dict(self.usage),
            cart_id=cart_id or self.cart_id,
            declared_status=declared,
            checkout_id=self.harness.checkout_id if self.harness is not None else None,
            order_id=self.harness.order_id if self.harness is not None else None,
            model_unavailable=model_unavailable,
        )


HOLDS = 2
"""How many times one episode's done is held for a question asked beside it; after that done ends the episode."""
CART_HOLDS = 1
"""How many times one checkout row's done(cart_ready) is held before the order; after that it ends the episode."""
HELD_CART = (
    "done was not run: this journey ends with an order, not a cart. The shopper's answer is the next message; carry "
    "on to the checkout and the order, or call done with cannot_continue if you cannot."
)
HELD_DONE = (
    "done was not run: you asked the shopper something in the same turn. Their answer is the next message; carry on "
    "with it, and call done once you have what you need or cannot go on."
)


def _assistant(text: str, calls: Sequence[ToolCall]) -> dict[str, Any]:
    """The assistant message that carried tool calls, as the next request must repeat it."""
    return {
        "role": "assistant",
        "content": text,
        "tool_calls": [
            {
                "id": call.id,
                "type": "function",
                "function": {
                    "name": call.name,
                    "arguments": call.raw_arguments
                    if call.arguments_error is not None and call.raw_arguments is not None
                    else dumps(call.arguments),
                },
            }
            for call in calls
        ],
    }
