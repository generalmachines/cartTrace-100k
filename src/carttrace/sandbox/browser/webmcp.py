"""WebMCP drafts the browser can host, as data: the API surface of each, its shim, its conformance probe, and the
in-page agent the runtime drives tools with.

Why a table. WebMCP is a Community Group draft that changes shape between snapshots: the 2026-10-09 draft has
`document.modelContext` with registerTool, getTools and executeTool, four annotation hints and the toolactivated and
toolcancel events, while the Chromium 153 build installed here (behind its WebMCP feature) exposes an older shape:
`inputSchema` as a string, no consequentialHint, debugging, toolactivated or toolcancel, and an executeTool that
refuses an object input. A profile names the draft a journey runs under (`ProtocolProfile.webmcp`), so the browser
must give pages exactly that draft's API, whichever Chromium it has.

The shim policy. For every episode the runtime first checks the browser's own API on a probe document:
- no native `document.modelContext`: the draft's shim is installed before any page script;
- a native one that passes the draft's probe: it is used as it is;
- a native one that fails the probe: the shim is installed over it, and the probe's findings are recorded.
Which ran, and why, is in the trace (`RuntimeInfo`), so two bundles made with different Chromium builds can be told
apart. The probe checks the members the draft defines and runs one tool through register, list and execute.

The in-page agent. The draft says a browser's own agent reads tools through an implementation-defined observation;
in-page agents use getTools() and executeTool(). The runtime is the latter: it evaluates `getTools()` and
`executeTool(tool, input, {signal})` in the page, so a tool runs exactly as an in-page agent would run it, and the
signal lets the runtime cancel a call that outlives its timeout (the draft's toolcancel then reaches the page).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import MappingProxyType

from carttrace.sandbox.surfaces.errors import UnsupportedVersion

JS = Path(__file__).resolve().parent / "js"


@dataclass(frozen=True, slots=True)
class WebMcpDraft:
    """One dated snapshot of the WebMCP draft, as the browser hosts it."""

    version: str
    spec: str
    """The vendored specification, relative to the repository root."""
    annotations: tuple[str, ...]
    """ToolAnnotations members, in the draft's order."""
    form_attributes: tuple[str, ...]
    """The declarative explainer's form attributes. The draft itself defines no declarative API."""
    shim: str
    """The shim's file under carttrace/sandbox/browser/js."""
    errors: tuple[str, ...]
    """The DOMException names a failed execution can reject with (tool execution errors and executeTool's checks)."""


DRAFTS: Mapping[str, WebMcpDraft] = MappingProxyType(
    {
        "2026-10-09": WebMcpDraft(
            version="2026-10-09",
            spec="vendor/webmcp/2026-10-09/index.bs",
            annotations=("readOnlyHint", "untrustedContentHint", "consequentialHint", "debugging"),
            form_attributes=("toolname", "tooldescription", "toolparamdescription", "toolautosubmit"),
            shim="webmcp-2026-10-09.shim.js",
            errors=("NotFoundError", "DataError", "OperationError", "AbortError", "InvalidStateError"),
        ),
    }
)
DEFAULT_WEBMCP = "2026-10-09"
"""The draft a storefront speaks when the profile names none: the newest one vendored."""


def served_webmcp() -> tuple[str, ...]:
    return tuple(sorted(DRAFTS))


def webmcp_draft(version: str | None) -> WebMcpDraft:
    """The draft a profile names (the default when it names none), or `UnsupportedVersion`."""
    found = DRAFTS.get(version or DEFAULT_WEBMCP)
    if found is None:
        raise UnsupportedVersion(
            f"the sandbox browser hosts no WebMCP draft {version!r}",
            hint=f"serve the profile at one of {list(served_webmcp())} (preset webmcp-2026-10-09)",
        )
    return found


@cache
def shim_source(version: str) -> str:
    """The shim of a draft, as the init script that installs it."""
    return (JS / webmcp_draft(version).shim).read_text(encoding="utf-8")


def force_shim_source() -> str:
    """Run before the shim when the native API failed the probe: the shim then replaces it."""
    return "window.__mcWebMcpForce = true;"


PROBE = """async () => {
  const report = {present: "modelContext" in Document.prototype, conforms: false, gaps: []};
  if (!report.present) return report;
  const gaps = report.gaps;
  let mc;
  try { mc = document.modelContext; } catch (e) { gaps.push(`document.modelContext threw ${e.name}`); return report; }
  if (!mc || typeof mc !== "object") { gaps.push("document.modelContext is not an object"); return report; }
  for (const m of ["registerTool", "getTools", "executeTool"]) {
    if (typeof mc[m] !== "function") gaps.push(`no ${m}()`);
  }
  if (gaps.length) return report;
  for (const h of ["ontoolchange", "ontoolactivated", "ontoolcancel"]) { if (!(h in mc)) gaps.push(`no ${h}`); }
  for (const i of ["ToolActivatedEvent", "ToolCancelEvent"]) {
    if (typeof window[i] !== "function") gaps.push(`no ${i}`);
  }
  const controller = new AbortController();
  try {
    await mc.registerTool({
      name: "mc_probe", description: "A probe of the WebMCP API.",
      inputSchema: {type: "object", properties: {n: {type: "number"}}},
      annotations: {readOnlyHint: true, consequentialHint: true},
      execute: async (input) => ({echo: input.n}),
    }, {signal: controller.signal});
    const tools = await mc.getTools();
    const tool = tools.find((t) => t.name === "mc_probe");
    if (tool === undefined) gaps.push("getTools() does not list a registered tool");
    else {
      if (typeof tool.inputSchema !== "object") gaps.push(`RegisteredTool.inputSchema is a ${typeof tool.inputSchema}`);
      const a = tool.annotations || {};
      for (const k of ANNOTATIONS) { if (typeof a[k] !== "boolean") gaps.push(`ToolAnnotations has no ${k}`); }
      if (tool.origin !== location.origin) gaps.push("RegisteredTool.origin is not the page's origin");
      try {
        const out = await mc.executeTool(tool, {n: 7});
        if (out !== '{"echo":7}') gaps.push(`executeTool() resolved with ${JSON.stringify(out)}`);
      } catch (e) { gaps.push(`executeTool() rejected: ${e.name}: ${e.message}`); }
    }
  } catch (e) { gaps.push(`${e.name}: ${e.message}`); }
  try { controller.abort(); } catch (e) { gaps.push(`unregistering threw ${e.name}`); }
  report.conforms = gaps.length === 0;
  return report;
}"""
"""The conformance probe, as a function for `page.evaluate`; `ANNOTATIONS` is filled per draft."""


def probe_source(version: str) -> str:
    names = ", ".join(f'"{a}"' for a in webmcp_draft(version).annotations)
    return PROBE.replace("ANNOTATIONS", f"[{names}]")


LIST_TOOLS = """async () => (await document.modelContext.getTools()).map((t) => ({
  name: t.name,
  title: t.title === undefined ? "" : t.title,
  description: t.description,
  inputSchema: t.inputSchema === undefined ? null : t.inputSchema,
  origin: t.origin,
  annotations: t.annotations === undefined ? null : t.annotations,
}))"""
"""getTools() as JSON: every member of a RegisteredTool but its `window`."""

CALL_TOOL = """async ([name, input, callId]) => {
  const calls = (window.__mcCalls = window.__mcCalls || {});
  const controller = new AbortController();
  calls[callId] = controller;
  try {
    const tools = await document.modelContext.getTools();
    const tool = tools.find((t) => t.name === name);
    if (tool === undefined) return {error: "NotFoundError", message: `no tool named ${name} is registered`};
    return {result: await document.modelContext.executeTool(tool, input, {signal: controller.signal})};
  } catch (e) {
    const failure = e === null || e === undefined ? "Error" : e.name || "Error";
    return {error: String(failure), message: e && e.message ? String(e.message) : String(e)};
  } finally {
    delete calls[callId];
  }
}"""
"""One tool call as an in-page agent makes it: find the tool with getTools(), run it with executeTool()."""

ABORT_CALL = """(callId) => {
  const controller = window.__mcCalls && window.__mcCalls[callId];
  if (controller) controller.abort(new DOMException("the sandbox browser's call timeout passed", "TimeoutError"));
  return Boolean(controller);
}"""


def determinism_source(seed: int) -> str:
    """Seeded randomness for every document: Math.random, crypto.getRandomValues and crypto.randomUUID draw from one
    mulberry32 stream that starts at the episode's seed in each document, so a page that rolls dice rolls the same."""
    return (JS / "determinism.js").read_text(encoding="utf-8").replace("__SEED__", str(seed & 0xFFFFFFFF))
