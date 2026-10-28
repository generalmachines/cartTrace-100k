"""The sandbox browser: one headless Chromium per episode, hosting the replica's pages and running their WebMCP tools
for a text agent that never sees a page.

Why a real browser engine. WebMCP tools are JavaScript a store's own page registers, and how they behave (what the
page does when one runs, what it answers, what an abort does) is the page's and the engine's, not ours to imitate in
Python. So the replica's storefront runs in Chromium, its tools are registered by its own script, and the runtime
calls them as an in-page agent would (`getTools()`, `executeTool()`), handing the agent only their names, schemas and
the JSON they answer (`carttrace.sandbox.browser.webmcp`).

What one episode gets:
- a host-wide browser slot (`slots`), so many episodes never open unbounded Chromium processes;
- Chromium launched by the repository's one policy (`carttrace.config.browser`): headless, no GPU, no name resolution;
- a fresh context per episode with the store market's locale and timezone, a fixed viewport, a frozen clock
  (Playwright's clock API, paused at the pack's start time) and seeded randomness (`webmcp.determinism_source`);
- offline routing (`routing.Relay`): the sandbox origin is answered in process, everything else is refused;
- the WebMCP API of the profile's draft: native when Chromium's conforms, else the vendored shim (`webmcp`);
- the 3-D Secure challenge page answered as the buyer: when a challenge frame loads during a call, the runtime
  confirms or cancels it according to the persona (`ChallengePolicy`), as the environment's own step;
- bounds on everything: a per-call timeout, a page budget and a total time budget, each ending the episode with a
  typed infrastructure error (`errors`), never a hang; and the browser closed on every path, errors included.

Every exchange, tool call and console error is written to the trace through the episode's recorder (`record.Wire`).
"""

from __future__ import annotations

import asyncio
import contextlib
import re
from collections.abc import Mapping
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal
from urllib.parse import urlsplit

import httpx

from carttrace.config.browser import INSTALL_HINT, context_options, launch_options
from carttrace.core.clock import Clock, SystemClock
from carttrace.core.errors import MCError
from carttrace.sandbox.browser.environment import BrowserLimits, Environment
from carttrace.sandbox.browser.errors import BrowserFailure, BrowserTimeout, BrowserUnavailable, PageBudgetExceeded
from carttrace.sandbox.browser.record import Wire, WireSink
from carttrace.sandbox.browser.routing import PROBE_PATH, Relay
from carttrace.sandbox.browser.slots import browser_slots
from carttrace.sandbox.browser.webmcp import (
    ABORT_CALL,
    CALL_TOOL,
    LIST_TOOLS,
    determinism_source,
    force_shim_source,
    probe_source,
    shim_source,
    webmcp_draft,
)
from carttrace.sandbox.surfaces.access import SESSION_HEADER

ChallengePolicy = Literal["approve", "decline"]
"""What the buyer does on a 3-D Secure challenge page: confirm it, or cancel it."""
Api = Literal["native", "shim"]
CHALLENGE_PATH = re.compile(r"^/3ds/[^/]+$")
CHALLENGE_BUTTON = """(value) => {
  const button = document.querySelector(`form button[name="approved"][value="${value}"]`);
  if (!button) return false;
  button.click();
  return true;
}"""
NATIVE_FEATURE = "--enable-features=WebMCP"
ABORT_WAIT_S = 2.0
CLOSE_WAIT_S = 10.0


@dataclass(frozen=True, slots=True)
class RuntimeInfo:
    """Which WebMCP API the pages got, and why."""

    draft: str
    api: Api
    native: Mapping[str, Any]
    """The probe's report on Chromium's own API: present, conforms, and the gaps found."""
    chromium: str

    def record(self) -> dict[str, Any]:
        return {"draft": self.draft, "api": self.api, "native": dict(self.native), "chromium": self.chromium}


@dataclass(frozen=True, slots=True)
class CallResult:
    """One executeTool, as the page answered it."""

    name: str
    result: str | None
    """The JSON string executeTool resolved with; None when it rejected."""
    error: str | None = None
    """The DOMException name it rejected with: NotFoundError, DataError, OperationError, AbortError..."""
    message: str = ""
    duration_ms: int = 0
    url: str = ""
    """The page's URL when the call ended."""


@dataclass(slots=True)
class _Watch:
    answered: set[str] = field(default_factory=set)
    queue: asyncio.Queue[Any] = field(default_factory=asyncio.Queue)


class SandboxBrowser:
    """One episode's browser on one sandbox app. `open`, then `tools` and `call`, then `close` (always)."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        session: str,
        draft: str,
        environment: Environment | None = None,
        limits: BrowserLimits | None = None,
        sink: WireSink | None = None,
        clock: Clock | None = None,
        challenge: ChallengePolicy = "approve",
        screenshots: Path | None = None,
        session_header: str = SESSION_HEADER,
    ) -> None:
        """`client` reaches the sandbox app in process (`httpx.ASGITransport`). `session` is the sandbox session
        every page request acts for. `draft` is the WebMCP draft the pages get. `screenshots` saves a PNG of each page
        the episode shows, for a person to look at."""
        self.client = client
        self.session = session
        self.draft = webmcp_draft(draft)
        self.env = environment or Environment()
        self.limits = limits or BrowserLimits()
        self.clock: Clock = clock or SystemClock()
        self.wire = Wire(sink, version=self.draft.version, clock=self.clock)
        self.challenge: ChallengePolicy = challenge
        self.screenshots = screenshots
        self.relay = Relay(
            client,
            origin=self.env.origin,
            session=session,
            wire=self.wire,
            max_pages=self.limits.max_pages,
            session_header=session_header,
        )
        self.info: RuntimeInfo | None = None
        self.shots: list[Path] = []
        self._stack = AsyncExitStack()
        self._playwright: Any = None
        self._browser: Any = None
        self._context: Any = None
        self._page: Any = None
        self._deadline = 0.0
        self._calls = 0
        self._console: list[tuple[str, str, str]] = []
        self._crashed: str | None = None
        self._last_shot: str | None = None
        self._background: set[asyncio.Future[Any]] = set()
        self._opened = False
        self._closed = False

    # -- lifecycle -----------------------------------------------------------------------------------------------------

    @property
    def closed(self) -> bool:
        return self._closed

    @property
    def page_url(self) -> str:
        return str(self._page.url) if self._page is not None else ""

    async def open(self, path: str = "/") -> RuntimeInfo:
        """Take a slot, launch, probe the WebMCP API, and load `path` on the sandbox origin. On any failure the
        browser is closed before the error is raised. Waiting for a slot is queueing, not episode time
        (mc-3r1.34): the episode's browser budget starts once a slot is held, so an episode queued behind other
        browsers is not born half-spent."""
        if self._opened:
            raise BrowserFailure("this browser was opened already", hint="open one SandboxBrowser per episode")
        self._opened = True
        loop = asyncio.get_running_loop()
        try:
            if self.limits.lock_dir is not None:
                slots = browser_slots(self.limits.lock_dir)
                await self._stack.enter_async_context(
                    slots.hold(self.limits.max_browsers, wait_s=self.limits.slot_wait_s)
                )
            self._deadline = loop.time() + self.limits.episode_timeout_s
            async with asyncio.timeout(self.limits.episode_timeout_s):
                return await self._open(path)
        except BaseException as exc:
            await self.close()
            if isinstance(exc, TimeoutError):
                raise BrowserTimeout(f"the browser did not open within {self.limits.episode_timeout_s:g} s") from exc
            translated = _translated(exc)
            if translated is exc:
                raise
            raise translated from exc

    async def _open(self, path: str) -> RuntimeInfo:
        from playwright.async_api import async_playwright

        self._playwright = await async_playwright().start()
        extra = (NATIVE_FEATURE,) if self.limits.native_webmcp else ()
        try:
            self._browser = await self._playwright.chromium.launch(**launch_options(extra_args=extra))
        except Exception as exc:
            if "Executable doesn't exist" in str(exc) or "playwright install" in str(exc):
                raise BrowserUnavailable("headless Chromium is not installed", hint=INSTALL_HINT) from exc
            raise
        env = self.env
        options = context_options(viewport=env.viewport, locale=env.locale, timezone=env.timezone)
        self._context = await self._browser.new_context(**options)
        self._context.set_default_timeout(self.limits.navigation_timeout_s * 1000)
        await self._context.clock.install(time=env.time)
        await self._context.clock.pause_at(env.time)
        await self._context.add_init_script(determinism_source(env.seed))
        await self._context.route("**/*", self.relay.handle)
        await self._context.route_web_socket("**/*", self.relay.refuse_socket)
        self._context.on("page", self._on_page)
        chromium = str(self._browser.version)
        self.wire.browser("browser/open", env.origin, body={"environment": env.record(), "chromium": chromium})
        info = await self._probe(chromium)
        page = await self._context.new_page()
        self._page = page
        page.on("console", self._on_console)
        page.on("pageerror", self._on_page_error)
        page.on("crash", self._on_crash)
        page.on("dialog", self._on_dialog)
        response = await page.goto(f"{env.origin}{path}", wait_until="load")
        self._flush_console()
        if response is None or response.status >= 400:
            status = None if response is None else response.status
            raise BrowserFailure(f"the storefront at {path} answered {status}")
        self._check()
        await self._shoot()
        return info

    async def _probe(self, chromium: str) -> RuntimeInfo:
        """Judge Chromium's own WebMCP API on a blank page, then give every later document the API the draft says."""
        probe = await self._context.new_page()
        try:
            await probe.goto(f"{self.env.origin}{PROBE_PATH}", wait_until="load")
            report = await probe.evaluate(probe_source(self.draft.version))
        finally:
            await probe.close()
        native = {
            "present": bool(report.get("present")),
            "conforms": bool(report.get("conforms")),
            "gaps": [str(g) for g in report.get("gaps") or ()],
        }
        api: Api = "native" if native["conforms"] else "shim"
        if api == "shim":
            if native["present"]:
                await self._context.add_init_script(force_shim_source())
            await self._context.add_init_script(shim_source(self.draft.version))
        self.info = RuntimeInfo(draft=self.draft.version, api=api, native=native, chromium=chromium)
        self.wire.browser("webmcp/runtime", self.env.origin, body=self.info.record())
        return self.info

    async def close(self) -> None:
        """Close the page, the context, the browser and Playwright, and give the slot back. Safe to call twice and
        after a failure; a step that fails is recorded and the rest still run."""
        if self._closed:
            return
        self._closed = True
        problems: list[str] = []
        self._flush_console()
        for name, target in (("context", self._context), ("browser", self._browser)):
            if target is None:
                continue
            try:
                async with asyncio.timeout(CLOSE_WAIT_S):
                    await target.close()
            except Exception as exc:  # recorded below; the browser is stopped with Playwright either way
                problems.append(f"{name}: {type(exc).__name__}: {exc}"[:300])
        if self._playwright is not None:
            try:
                async with asyncio.timeout(CLOSE_WAIT_S):
                    await self._playwright.stop()
            except Exception as exc:
                problems.append(f"playwright: {type(exc).__name__}: {exc}"[:300])
        await self._stack.aclose()
        self._page = self._context = self._browser = self._playwright = None
        if self._opened:
            self.wire.browser("browser/close", self.env.origin, error="; ".join(problems) or None)

    # -- tools ---------------------------------------------------------------------------------------------------------

    async def tools(self) -> list[dict[str, Any]]:
        """The page's registered tools as getTools() lists them, without their windows."""
        page = self._live()
        start = self.clock.monotonic()
        try:
            listed = await asyncio.wait_for(page.evaluate(LIST_TOOLS), timeout=self._budget())
        except TimeoutError as exc:
            raise BrowserTimeout("getTools() did not answer in time") from exc
        except Exception as exc:
            translated = _translated(exc)
            if translated is exc:
                raise
            raise translated from exc
        self._flush_console()
        self._check()
        tools = [dict(t) for t in listed]
        self.wire.write("webmcp", "getTools", page.url, response_body=tools, duration_ms=self.wire.elapsed_ms(start))
        return tools

    async def call(self, name: str, arguments: Mapping[str, Any]) -> CallResult:
        """Run one tool with executeTool, answering a 3-D Secure challenge as the buyer if one appears meanwhile.
        Raises `BrowserTimeout` when the call outlives its timeout (the page is told to cancel it first)."""
        page = self._live()
        self._flush_console()
        self._calls += 1
        call_id = f"call-{self._calls}"
        timeout = self._budget()
        start = self.clock.monotonic()
        watch = _Watch()

        def navigated(frame: Any) -> None:
            if frame is not page.main_frame and CHALLENGE_PATH.match(urlsplit(frame.url).path):
                watch.queue.put_nowait(frame)

        page.on("framenavigated", navigated)
        task = asyncio.ensure_future(page.evaluate(CALL_TOOL, [name, dict(arguments), call_id]))
        answering = asyncio.ensure_future(self._answer_challenges(watch))
        try:
            outcome = await asyncio.wait_for(asyncio.shield(task), timeout=timeout)
        except TimeoutError as exc:
            await self._abort(call_id)
            task.cancel()
            self._flush_console()
            self.wire.write(
                "webmcp",
                "executeTool",
                page.url,
                request_body={"name": name, "arguments": dict(arguments)},
                duration_ms=self.wire.elapsed_ms(start),
                error="timeout",
            )
            raise BrowserTimeout(f"the page's tool {name} did not answer within {timeout:g} s") from exc
        except Exception as exc:
            unloaded = _unloaded(exc)
            if unloaded is None:
                translated = _translated(exc)
                if translated is exc:
                    raise
                raise translated from exc
            outcome = unloaded
        finally:
            page.remove_listener("framenavigated", navigated)
            answering.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await answering
        self._flush_console()
        self._check()
        result = CallResult(
            name=name,
            result=outcome.get("result") if isinstance(outcome.get("result"), str) else None,
            error=str(outcome["error"]) if outcome.get("error") is not None else None,
            message=str(outcome.get("message") or ""),
            duration_ms=self.wire.elapsed_ms(start),
            url=page.url,
        )
        self.wire.write(
            "webmcp",
            "executeTool",
            result.url,
            request_body={"name": name, "arguments": dict(arguments)},
            response_body=_parsed(result.result),
            duration_ms=result.duration_ms,
            error=None if result.error is None else f"{result.error}: {result.message}"[:500],
        )
        await self._shoot()
        return result

    async def _answer_challenges(self, watch: _Watch) -> None:
        """Answer each 3-D Secure challenge frame that loads, once, as the persona would."""
        approved = self.challenge == "approve"
        while True:
            frame = await watch.queue.get()
            try:
                await frame.wait_for_load_state("load")
                if frame.url in watch.answered:
                    continue
                pressed = await frame.evaluate(CHALLENGE_BUTTON, "1" if approved else "0")
            except Exception as exc:  # the frame went away (answered, or the page moved on): nothing to answer
                if _gone(exc):
                    continue
                raise
            if pressed:
                watch.answered.add(frame.url)
                self.wire.browser("3ds/answer", frame.url, body={"approved": approved, "persona": self.challenge})

    async def _abort(self, call_id: str) -> None:
        """Cancel a call in the page (its signal aborts and the page receives toolcancel), if the page still answers."""
        if self._page is None:
            return
        try:
            await asyncio.wait_for(self._page.evaluate(ABORT_CALL, call_id), timeout=ABORT_WAIT_S)
        except Exception as exc:  # a hung page cannot be told; the episode ends either way
            self.wire.browser("webmcp/abort", self.page_url, error=f"{type(exc).__name__}"[:200])

    # -- pages ---------------------------------------------------------------------------------------------------------

    async def screenshot(self, path: Path) -> Path:
        page = self._live()
        path.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(path), full_page=True)
        return path

    async def _shoot(self) -> None:
        """One screenshot per page the episode shows (a new URL), when screenshots were asked for."""
        if self.screenshots is None or self._page is None or self._page.url == self._last_shot:
            return
        self._last_shot = self._page.url
        parts = urlsplit(self._page.url)
        slug = re.sub(r"[^A-Za-z0-9]+", "-", f"{parts.path} {parts.query}").strip("-")[:60] or "home"
        self.shots.append(await self.screenshot(self.screenshots / f"{len(self.shots) + 1:02d}-{slug}.png"))

    # -- events --------------------------------------------------------------------------------------------------------

    def _on_console(self, message: Any) -> None:
        if message.type in ("error", "warning"):
            self._console.append(("console", str(message.type), str(message.text)[:1000]))

    def _on_page_error(self, error: Any) -> None:
        self._console.append(("pageerror", "error", str(error)[:1000]))

    def _on_crash(self, page: Any) -> None:
        self._crashed = "the page crashed"

    def _on_dialog(self, dialog: Any) -> None:
        self._console.append(("dialog", str(dialog.type), str(dialog.message)[:1000]))
        self._spawn(dialog.dismiss())

    def _on_page(self, page: Any) -> None:
        """A page the episode did not open (a popup): recorded, counted against the budget, and closed."""
        if self._page is None or page is self._page:
            return
        self.relay.documents += 1
        self.wire.browser("blocked", str(page.url), body={"resource": "popup"}, error="popups are closed")
        self._spawn(page.close())

    def _spawn(self, work: Any) -> None:
        """Run a small cleanup (dismiss a dialog, close a popup) beside the episode, keeping it referenced."""
        task = asyncio.ensure_future(work)
        self._background.add(task)
        task.add_done_callback(self._background.discard)

    def _flush_console(self) -> None:
        url = self.page_url
        for kind, level, text in self._console:
            self.wire.browser(kind, url, body={"type": level, "text": text}, error=text)
        self._console.clear()

    # -- bounds --------------------------------------------------------------------------------------------------------

    def _live(self) -> Any:
        if self._closed or self._page is None:
            raise BrowserFailure("the browser is closed or was never opened", hint="open a SandboxBrowser first")
        self._check()
        return self._page

    def _budget(self) -> float:
        """Seconds the next step may take: the call timeout, cut to what is left of the episode's budget."""
        left = self._deadline - asyncio.get_running_loop().time()
        if left <= 0:
            raise BrowserTimeout(f"the episode's browser budget of {self.limits.episode_timeout_s:g} s is spent")
        return min(self.limits.call_timeout_s, left)

    def _check(self) -> None:
        if self._crashed is not None:
            raise BrowserFailure(self._crashed)
        if self.relay.over_budget is not None:
            raise PageBudgetExceeded(
                f"the episode loaded more than {self.limits.max_pages} documents ({self.relay.over_budget})",
                hint="the replica page loads without end; the episode is VOID",
            )
        if self.relay.documents > self.limits.max_pages:
            raise PageBudgetExceeded(f"the episode opened more than {self.limits.max_pages} pages")
        if self.relay.failures:
            raise BrowserFailure(f"the sandbox app failed a page request: {self.relay.failures[0]}")


def _parsed(result: str | None) -> Any:
    if result is None:
        return None
    import json

    try:
        return json.loads(result)
    except ValueError:
        return result


def _gone(exc: BaseException) -> bool:
    text = str(exc)
    return any(s in text for s in ("detached", "Target closed", "destroyed", "has been closed", "navigat"))


def _unloaded(exc: BaseException) -> dict[str, Any] | None:
    """A call whose page went away while it ran: the draft's "target document unloaded", an AbortError."""
    text = str(exc)
    if "Execution context was destroyed" in text or "navigation" in text.lower():
        return {"error": "AbortError", "message": "the page unloaded before the tool answered"}
    return None


def _translated(exc: BaseException) -> BaseException:
    """Playwright's own errors as the runtime's typed ones; typed errors pass through."""
    if isinstance(exc, MCError | asyncio.CancelledError | KeyboardInterrupt | SystemExit):
        return exc
    first = str(exc).strip().splitlines()[0] if str(exc).strip() else type(exc).__name__
    if "Timeout" in type(exc).__name__ or isinstance(exc, TimeoutError):
        return BrowserTimeout(first[:300])
    return BrowserFailure(f"{type(exc).__name__}: {first}"[:400])
