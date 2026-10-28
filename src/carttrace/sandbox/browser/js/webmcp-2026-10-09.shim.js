// WebMCP shim for the W3C Web Machine Learning CG draft of 2026-10-09 (vendor/webmcp/2026-10-09/index.bs).
//
// The sandbox browser installs this before any page script when Chromium has no native document.modelContext, or
// when the native one does not conform to this draft (carttrace.sandbox.browser.webmcp records which ran and why). It
// follows the draft's algorithms step by step: `Document.modelContext` ([SecureContext, SameObject]), ModelContext
// with registerTool / getTools / executeTool and the toolchange / toolactivated / toolcancel events, the tool
// execution errors and their DOMException names, the pending tool executions map of the top-level traversable, and
// the unloading document cleanup steps.
//
// The declarative form API is not part of this draft; its explainer (declarative-api-explainer.md) is followed for
// forms that carry `toolname`: `tooldescription`, `toolparamdescription` and `toolautosubmit`, an input schema
// synthesised from the form's controls (the explainer leaves the algorithm open; ours is below), and
// SubmitEvent.agentInvoked / respondWith(). Declarative tools live in the same tool map, so getTools() lists them and
// executeTool() runs them, as the explainer expects ("Should they be invokable from that interface? Almost certainly,
// yes"). The :tool-form-active pseudo-classes cannot be polyfilled and are not.
//
// Where the draft queues a task on the "webmcp task source", the shim uses a microtask: the sandbox freezes the
// page's timers (Playwright's clock), and a microtask keeps every documented ordering (toolchange before the register
// promise resolves, getTools' results after registration).
(() => {
  "use strict";
  const HOOK = Symbol.for("mc.webmcp.2026-10-09");
  if (window[HOOK] !== undefined || !window.isSecureContext) return;
  // A native API stays unless the runtime found it does not conform and asked for the shim over it. (`in` rather
  // than reading the attribute: a native WebIDL getter throws when called on the prototype.)
  if ("modelContext" in Document.prototype && window.__mcWebMcpForce !== true) return;

  const TOOL_NAME = /^[A-Za-z0-9_.\-]{1,128}$/;
  const ANNOTATIONS = ["readOnlyHint", "untrustedContentHint", "consequentialHint", "debugging"];
  const FORM_ATTRIBUTES = ["toolname", "tooldescription", "toolparamdescription", "toolautosubmit", "name", "required"];
  const SECRET = Symbol("ModelContext");
  const later = (fn) => Promise.resolve().then(fn);

  // -- helpers -----------------------------------------------------------------------------------------------------

  function domError(name, message) {
    return new DOMException(message || name, name);
  }

  // "serializing a JavaScript value to a JSON string" (Infra): JSON.stringify, and a TypeError when it gives undefined.
  function serialize(value) {
    const text = JSON.stringify(value);
    if (text === undefined) throw new TypeError("the value cannot be serialized to JSON");
    return text;
  }

  function isObject(value) {
    return (typeof value === "object" && value !== null) || typeof value === "function";
  }

  // "is origin potentially trustworthy?" (Secure Contexts) for a parsed URL's origin.
  function trustworthy(url) {
    if (url.origin === "null") return false;
    if (url.protocol === "https:" || url.protocol === "wss:" || url.protocol === "file:") return true;
    const host = url.hostname.replace(/^\[|\]$/g, "");
    return host === "localhost" || host.endsWith(".localhost") || /^127\./.test(host) || host === "::1";
  }

  function parseOrigins(list) {
    const out = [];
    for (const raw of list) {
      let url;
      try {
        url = new URL(String(raw));
      } catch (e) {
        throw domError("SecurityError", `${raw} is not a URL`);
      }
      if (!trustworthy(url)) throw domError("SecurityError", `${url.origin} is not potentially trustworthy`);
      out.push(url.origin);
    }
    return out;
  }

  // "tool is exposed to an origin"
  function exposed(ownerOrigin, exposedOrigins, accessingOrigin) {
    return ownerOrigin === accessingOrigin || exposedOrigins.includes(accessingOrigin);
  }

  function annotationsOf(raw) {
    if (raw === undefined || raw === null) return null;
    if (!isObject(raw)) throw new TypeError("annotations must be a dictionary");
    const out = {};
    for (const key of ANNOTATIONS) out[key] = raw[key] === undefined ? false : Boolean(raw[key]);
    return out;
  }

  // Every document's hook in this tree that the shim can reach (same-origin frames), the top first.
  function treeHooks() {
    const out = [];
    const visit = (win) => {
      let hook;
      try {
        hook = win[HOOK];
      } catch (e) {
        return; // a cross-origin frame: its tools are not reachable from script
      }
      if (hook !== undefined) out.push(hook);
      for (let i = 0; i < win.frames.length; i += 1) visit(win.frames[i]);
    };
    visit(window.top);
    return out;
  }

  // The traversable navigable's pending tool executions map, kept on the top-level document's hook.
  function traversablePending() {
    try {
      return window.top[HOOK].pending;
    } catch (e) {
      return pending;
    }
  }

  // -- events --------------------------------------------------------------------------------------------------------

  class ToolActivatedEvent extends Event {
    #toolName;
    constructor(type, init = {}) {
      super(type, init);
      this.#toolName = init.toolName === undefined ? "" : String(init.toolName);
    }
    get toolName() {
      return this.#toolName;
    }
  }

  class ToolCancelEvent extends Event {
    #toolName;
    constructor(type, init = {}) {
      super(type, init);
      this.#toolName = init.toolName === undefined ? "" : String(init.toolName);
    }
    get toolName() {
      return this.#toolName;
    }
  }

  // -- the model context -----------------------------------------------------------------------------------------------

  const internals = new WeakMap(); // ModelContext -> {toolMap, local, active}
  const pending = new Map(); // this document's copy of the traversable's map; only the top's is used

  class ModelContext extends EventTarget {
    constructor(key, active = true) {
      if (key !== SECRET) throw new TypeError("Illegal constructor");
      super();
      internals.set(this, { toolMap: new Map(), local: new Map(), active });
    }

    #fullyActive() {
      if (!internals.get(this).active || !document.defaultView) {
        throw domError("InvalidStateError", "the document is not fully active");
      }
    }

    registerTool(tool, options = {}) {
      try {
        return this.#register(tool, options);
      } catch (e) {
        return Promise.reject(e);
      }
    }

    #register(tool, options) {
      if (!isObject(tool)) throw new TypeError("registerTool needs a ModelContextTool dictionary");
      if (tool.name === undefined) throw new TypeError("ModelContextTool.name is required");
      if (tool.description === undefined) throw new TypeError("ModelContextTool.description is required");
      if (typeof tool.execute !== "function") throw new TypeError("ModelContextTool.execute must be a function");
      if (tool.inputSchema !== undefined && !isObject(tool.inputSchema)) {
        throw new TypeError("ModelContextTool.inputSchema must be an object");
      }
      const name = String(tool.name);
      const title = tool.title === undefined ? null : String(tool.title);
      const description = String(tool.description);
      const annotations = annotationsOf(tool.annotations);
      const execute = tool.execute;
      this.#fullyActive();
      const toolMap = internals.get(this).toolMap;
      if (toolMap.has(name)) throw domError("InvalidStateError", `a tool named ${name} is already registered`);
      if (!TOOL_NAME.test(name)) throw domError("InvalidStateError", `${JSON.stringify(name)} is not a valid tool name`);
      if (description === "") throw domError("InvalidStateError", "a tool needs a description");
      let schema = "";
      if (tool.inputSchema !== undefined) schema = serialize(tool.inputSchema);
      const signal = options === undefined || options === null ? undefined : options.signal;
      if (signal !== undefined && signal.aborted) return Promise.reject(signal.reason);
      const exposedOrigins = options && options.exposedTo !== undefined ? parseOrigins(options.exposedTo) : [];
      let resolve;
      let reject;
      const promise = new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      });
      if (signal !== undefined) {
        signal.addEventListener("abort", () => {
          unregister(this, name);
          reject(signal.reason);
        }, { once: true });
      }
      toolMap.set(name, {
        name,
        title,
        description,
        inputSchema: schema,
        annotations,
        exposedOrigins,
        execute: (target, input, completion, uuid) => invokeExecute(execute, name, target, input, completion, uuid),
      });
      later(() => {
        notifyToolChange(location.origin, exposedOrigins);
        later(() => resolve(undefined));
      });
      return promise;
    }

    getTools(options = {}) {
      try {
        return this.#getTools(options);
      } catch (e) {
        return Promise.reject(e);
      }
    }

    #getTools(options) {
      this.#fullyActive();
      const fromOrigins = options && options.fromOrigins !== undefined ? parseOrigins(options.fromOrigins) : [];
      const callerOrigin = location.origin;
      return new Promise((resolve) => {
        later(() => {
          const tools = [];
          for (const hook of treeHooks()) {
            const targetOrigin = hook.origin;
            if (targetOrigin !== callerOrigin && !fromOrigins.includes(targetOrigin)) continue;
            for (const definition of hook.toolMap().values()) {
              if (!exposed(targetOrigin, definition.exposedOrigins, callerOrigin)) continue;
              const registered = {
                name: definition.name,
                title: definition.title === null ? "" : definition.title,
                description: definition.description,
                window: hook.window,
                origin: targetOrigin,
              };
              if (definition.inputSchema !== "") registered.inputSchema = JSON.parse(definition.inputSchema);
              if (definition.annotations !== null) registered.annotations = { ...definition.annotations };
              tools.push(registered);
            }
          }
          tools.sort((a, b) => (a.name < b.name ? -1 : a.name > b.name ? 1 : 0));
          later(() => resolve(tools));
        });
      });
    }

    executeTool(tool, inputObject, options = {}) {
      try {
        return this.#execute(tool, inputObject, options);
      } catch (e) {
        return Promise.reject(e);
      }
    }

    #execute(tool, inputObject, options) {
      if (!isObject(tool)) throw new TypeError("executeTool needs a RegisteredTool dictionary");
      for (const member of ["name", "description", "window", "origin"]) {
        if (tool[member] === undefined) throw new TypeError(`RegisteredTool.${member} is required`);
      }
      if (inputObject !== undefined && !isObject(inputObject)) throw new TypeError("inputObject must be an object");
      this.#fullyActive();
      let expected;
      try {
        expected = new URL(String(tool.origin));
      } catch (e) {
        throw domError("NotSupportedError", "the tool's origin is not a URL");
      }
      if (expected.origin === "null") throw domError("NotSupportedError", "the tool's origin is opaque");
      const input = inputObject === undefined ? {} : inputObject;
      const inputArguments = serialize(input);
      const targetWindow = tool.window;
      const toolName = String(tool.name);
      const uuid = Symbol(toolName); // a unique internal value
      let resolve;
      let reject;
      const promise = new Promise((res, rej) => {
        resolve = res;
        reject = rej;
      });
      const signal = options === undefined || options === null ? undefined : options.signal;
      if (signal !== undefined) {
        if (signal.aborted) return Promise.reject(signal.reason);
        signal.addEventListener("abort", () => {
          reject(signal.reason);
          later(() => cancelPendingExecution(uuid));
        }, { once: true });
      }
      const callerDocument = document;
      later(() => {
        let sameTree = false;
        let targetHook;
        try {
          sameTree = targetWindow.top === window.top;
          targetHook = targetWindow[HOOK];
        } catch (e) {
          targetHook = undefined;
        }
        if (!sameTree) {
          later(() => reject(domError("InvalidStateError", "the tool's document is in another tab")));
          return;
        }
        if (targetHook === undefined || targetHook.origin !== expected.origin) {
          later(() => reject(domError("NotFoundError", "no such tool")));
          return;
        }
        const definition = targetHook.toolMap().get(toolName);
        if (definition === undefined || !exposed(targetHook.origin, definition.exposedOrigins, location.origin)) {
          later(() => reject(domError("NotFoundError", "no such tool")));
          return;
        }
        const map = traversablePending();
        const completion = (result, error) => {
          later(() => {
            if (!map.has(uuid)) return;
            map.delete(uuid);
            if (error === null) later(() => resolve(result));
            else later(() => reject(domError(error, `the tool execution failed: ${error}`)));
          });
        };
        map.set(uuid, {
          caller: callerDocument,
          target: targetHook.document,
          targetHook,
          toolName,
          completion,
        });
        later(() => targetHook.executeSteps(toolName, inputArguments, completion, uuid));
      });
      return promise;
    }
  }

  // Event handler IDL attributes.
  for (const type of ["toolchange", "toolactivated", "toolcancel"]) {
    const handlers = new WeakMap();
    Object.defineProperty(ModelContext.prototype, `on${type}`, {
      configurable: true,
      enumerable: true,
      get() {
        const entry = handlers.get(this);
        return entry === undefined ? null : entry.callback;
      },
      set(callback) {
        const entry = handlers.get(this);
        if (entry !== undefined) this.removeEventListener(type, entry.listener);
        if (typeof callback !== "function" && !isObject(callback)) {
          handlers.delete(this);
          return;
        }
        const listener = (event) => (typeof callback === "function" ? callback.call(this, event) : undefined);
        handlers.set(this, { callback, listener });
        this.addEventListener(type, listener);
      },
    });
  }

  // The tool execute steps (on the target document) and "invoke the execute callback".
  function executeSteps(context, toolName, inputArguments, completion, uuid) {
    const definition = internals.get(context).toolMap.get(toolName);
    if (definition === undefined) {
      completion(null, "NotFoundError");
      return;
    }
    definition.execute(context, inputArguments, completion, uuid);
  }

  function invokeExecute(execute, toolName, context, inputArguments, completion, uuid) {
    let input;
    try {
      input = JSON.parse(inputArguments);
    } catch (e) {
      completion(null, "DataError");
      return;
    }
    if (!isObject(input)) {
      completion(null, "DataError");
      return;
    }
    context.dispatchEvent(new ToolActivatedEvent("toolactivated", { toolName }));
    const controller = new AbortController();
    const local = internals.get(context).local;
    local.set(uuid, { controller });
    let toolPromise;
    try {
      toolPromise = Promise.resolve(execute.call(undefined, input, { signal: controller.signal }));
    } catch (e) {
      toolPromise = Promise.reject(e);
    }
    toolPromise.then(
      (value) => {
        if (!local.has(uuid)) return;
        local.delete(uuid);
        let text;
        try {
          text = serialize(value);
        } catch (e) {
          completion(null, "OperationError"); // unserializable result
          return;
        }
        completion(text, null);
      },
      (reason) => {
        console.warn(`WebMCP tool ${toolName} rejected:`, reason instanceof Error ? reason.message : String(reason));
        if (!local.has(uuid)) return;
        local.delete(uuid);
        completion(null, "OperationError"); // tool rejected
      },
    );
  }

  // "cancel a pending tool execution"
  function cancelPendingExecution(uuid) {
    const map = traversablePending();
    const execution = map.get(uuid);
    if (execution === undefined) return;
    map.delete(uuid);
    later(() => execution.targetHook.cancelLocal(uuid, execution.toolName));
  }

  function cancelLocal(context, uuid, toolName) {
    const local = internals.get(context).local;
    const entry = local.get(uuid);
    if (entry === undefined) return;
    local.delete(uuid);
    entry.controller.abort();
    context.dispatchEvent(new ToolCancelEvent("toolcancel", { toolName }));
  }

  // "unregister a tool"
  function unregister(context, name) {
    const toolMap = internals.get(context).toolMap;
    const definition = toolMap.get(name);
    if (definition === undefined) return;
    toolMap.delete(name);
    later(() => notifyToolChange(location.origin, definition.exposedOrigins));
  }

  // "notify documents of a tool change"
  function notifyToolChange(ownerOrigin, exposedOrigins) {
    for (const hook of treeHooks()) {
      if (exposed(ownerOrigin, exposedOrigins, hook.origin)) later(() => hook.fire("toolchange"));
    }
  }

  // -- this document ---------------------------------------------------------------------------------------------------

  const context = new ModelContext(SECRET);
  const hook = {
    version: "2026-10-09",
    origin: location.origin,
    window,
    document,
    pending,
    toolMap: () => internals.get(context).toolMap,
    executeSteps: (toolName, input, completion, uuid) => executeSteps(context, toolName, input, completion, uuid),
    cancelLocal: (uuid, toolName) => cancelLocal(context, uuid, toolName),
    fire: (type) => context.dispatchEvent(new Event(type)),
  };
  Object.defineProperty(window, HOOK, { value: hook, enumerable: false, configurable: false, writable: false });

  // [SameObject]: one ModelContext per Document. Documents without a browsing context (DOMParser,
  // createHTMLDocument) get one that is never fully active, so its methods reject with InvalidStateError.
  const others = new WeakMap();
  Object.defineProperty(Document.prototype, "modelContext", {
    configurable: true,
    enumerable: true,
    get() {
      if (this === document) return context;
      if (!(this instanceof Document)) throw new TypeError("Illegal invocation");
      if (!others.has(this)) others.set(this, new ModelContext(SECRET, false));
      return others.get(this);
    },
  });
  for (const [name, value] of [
    ["ModelContext", ModelContext],
    ["ToolActivatedEvent", ToolActivatedEvent],
    ["ToolCancelEvent", ToolCancelEvent],
  ]) {
    Object.defineProperty(window, name, { value, configurable: true, writable: true, enumerable: false });
  }

  // This specification's unloading document cleanup steps.
  window.addEventListener("pagehide", () => {
    const map = traversablePending();
    for (const [uuid, execution] of [...map.entries()]) {
      const isTarget = execution.target === document;
      const isCaller = execution.caller === document;
      if (isTarget && !isCaller) execution.completion(null, "AbortError"); // target document unloaded
      else if (isCaller && !isTarget) cancelPendingExecution(uuid);
      else if (isTarget || isCaller) map.delete(uuid);
    }
  });

  // -- declarative tools (explainer, not normative at this draft) ------------------------------------------------------

  const agentInvoked = new WeakSet();
  const responses = new WeakMap();
  Object.defineProperty(SubmitEvent.prototype, "agentInvoked", {
    configurable: true,
    enumerable: true,
    get() {
      return agentInvoked.has(this);
    },
  });
  Object.defineProperty(SubmitEvent.prototype, "respondWith", {
    configurable: true,
    writable: true,
    enumerable: true,
    value: function respondWith(agentResponse) {
      if (!agentInvoked.has(this)) throw domError("InvalidStateError", "only an agent-invoked submission responds");
      if (!this.defaultPrevented) throw domError("InvalidStateError", "call preventDefault() before respondWith()");
      if (responses.has(this)) throw domError("InvalidStateError", "respondWith() was already called");
      responses.set(this, Promise.resolve(agentResponse));
    },
  });

  // The input schema of a form: one property per named control, in tree order. Text-like inputs, textareas and
  // hidden inputs are strings; number and range inputs are numbers (with their min and max); a checkbox is a boolean;
  // a radio group and a select are strings with their values as `enum` (and `anyOf` const/title pairs naming each
  // option's label, as Chromium's own synthesis does). `required` lists the controls marked required, and
  // `toolparamdescription` gives each property's description.
  const SKIPPED = new Set(["submit", "button", "reset", "image", "file"]);

  function controlsOf(form) {
    return [...form.elements].filter(
      (el) => el.name && !el.disabled && !(el instanceof HTMLButtonElement) && !(el instanceof HTMLFieldSetElement)
        && !(el instanceof HTMLOutputElement) && !(el instanceof HTMLObjectElement)
        && !(el instanceof HTMLInputElement && SKIPPED.has(el.type)),
    );
  }

  function schemaOf(form) {
    const properties = {};
    const required = [];
    for (const el of controlsOf(form)) {
      const name = el.name;
      const described = el.getAttribute("toolparamdescription");
      let prop;
      if (el instanceof HTMLSelectElement) {
        const options = [...el.options].filter((o) => !o.disabled);
        prop = { type: "string" };
        if (options.length) {
          prop.anyOf = options.map((o) => ({ type: "string", const: o.value, title: o.label }));
          prop.enum = options.map((o) => o.value);
        }
      } else if (el instanceof HTMLInputElement && el.type === "radio") {
        const existing = properties[name];
        prop = existing || { type: "string", enum: [] };
        if (!prop.enum.includes(el.value)) prop.enum.push(el.value);
      } else if (el instanceof HTMLInputElement && el.type === "checkbox") {
        prop = { type: "boolean" };
      } else if (el instanceof HTMLInputElement && (el.type === "number" || el.type === "range")) {
        prop = { type: "number" };
        if (el.min !== "") prop.minimum = Number(el.min);
        if (el.max !== "") prop.maximum = Number(el.max);
      } else {
        prop = { type: "string" };
        if (el.minLength > 0) prop.minLength = el.minLength;
        if (el.maxLength >= 0 && el.maxLength < 524288) prop.maxLength = el.maxLength;
        if (el instanceof HTMLInputElement && el.pattern) prop.pattern = `^(?:${el.pattern})$`;
      }
      if (described) prop.description = described;
      properties[name] = prop;
      if (el.required && !required.includes(name)) required.push(name);
    }
    const schema = { type: "object", properties };
    if (required.length) schema.required = required;
    return schema;
  }

  function fill(form, input) {
    for (const el of controlsOf(form)) {
      if (!Object.prototype.hasOwnProperty.call(input, el.name)) continue;
      const value = input[el.name];
      if (el instanceof HTMLInputElement && el.type === "checkbox") el.checked = Boolean(value);
      else if (el instanceof HTMLInputElement && el.type === "radio") el.checked = el.value === String(value);
      else el.value = value === null || value === undefined ? "" : String(value);
      el.dispatchEvent(new Event("input", { bubbles: true }));
      el.dispatchEvent(new Event("change", { bubbles: true }));
    }
  }

  function submitButton(form) {
    return form.querySelector("button[type=submit], button:not([type]), input[type=submit]");
  }

  function runForm(form, toolName, inputArguments, completion, uuid) {
    let input;
    try {
      input = JSON.parse(inputArguments);
    } catch (e) {
      completion(null, "DataError");
      return;
    }
    if (!isObject(input)) {
      completion(null, "DataError");
      return;
    }
    fill(form, input);
    context.dispatchEvent(new ToolActivatedEvent("toolactivated", { toolName }));
    const controller = new AbortController();
    const local = internals.get(context).local;
    local.set(uuid, { controller });
    let toolPromise;
    if (!form.hasAttribute("toolautosubmit")) {
      const button = submitButton(form);
      if (button) button.focus();
      toolPromise = Promise.resolve({ submitted: false, note: "The form is filled in; the user reviews and submits it." });
    } else {
      const event = new SubmitEvent("submit", { bubbles: true, cancelable: true, submitter: submitButton(form) });
      agentInvoked.add(event);
      form.dispatchEvent(event);
      if (responses.has(event)) toolPromise = responses.get(event);
      else if (event.defaultPrevented) toolPromise = Promise.resolve(null);
      else {
        local.delete(uuid);
        form.submit(); // navigates: the unloading cleanup answers the caller with AbortError
        return;
      }
    }
    toolPromise.then(
      (value) => {
        if (!local.has(uuid)) return;
        local.delete(uuid);
        let text;
        try {
          text = serialize(value);
        } catch (e) {
          completion(null, "OperationError");
          return;
        }
        completion(text, null);
      },
      () => {
        if (!local.has(uuid)) return;
        local.delete(uuid);
        completion(null, "OperationError");
      },
    );
  }

  const declared = new Map(); // tool name -> form
  function scanForms() {
    const toolMap = internals.get(context).toolMap;
    const seen = new Map();
    for (const form of document.querySelectorAll("form[toolname]")) {
      const name = form.getAttribute("toolname");
      const description = form.getAttribute("tooldescription") || "";
      if (!TOOL_NAME.test(name) || description === "" || seen.has(name)) continue;
      const existing = toolMap.get(name);
      if (existing !== undefined && !existing.declarative) continue; // an imperative tool holds the name
      seen.set(name, form);
    }
    let changed = false;
    for (const [name, form] of declared) {
      if (seen.get(name) !== form) {
        toolMap.delete(name);
        declared.delete(name);
        changed = true;
      }
    }
    for (const [name, form] of seen) {
      const schema = serialize(schemaOf(form));
      const description = form.getAttribute("tooldescription");
      const existing = toolMap.get(name);
      if (existing !== undefined && existing.form === form && existing.inputSchema === schema
          && existing.description === description) continue;
      toolMap.set(name, {
        name,
        title: null,
        description,
        inputSchema: schema,
        annotations: null,
        exposedOrigins: [],
        declarative: true,
        form,
        execute: (target, input, completion, uuid) => runForm(form, name, input, completion, uuid),
      });
      declared.set(name, form);
      changed = true;
    }
    if (changed) later(() => notifyToolChange(location.origin, []));
  }

  new MutationObserver(scanForms).observe(document, {
    subtree: true,
    childList: true,
    attributes: true,
    attributeFilter: FORM_ATTRIBUTES,
  });
  document.addEventListener("DOMContentLoaded", scanForms);
})();
