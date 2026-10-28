// The replica storefront's own script: it registers the page's WebMCP tools and keeps the page in step with them.
//
// Imperative tools are registered with document.modelContext.registerTool() from the definitions the server embeds
// (carttrace.sandbox.surfaces.webmcp.tools). Each one posts its input to the storefront's API, swaps the returned fragment
// into the page, moves the address bar to the page now shown (history.pushState, so the document never unloads
// under a running tool), and resolves with the view the page shows. The checkout's forms are declarative tools: an
// agent-invoked submission is answered with SubmitEvent.respondWith() and the same view; a person's submission
// updates the page the same way. place_order shows the bank's 3-D Secure page in a frame when the payment needs it,
// waits for the buyer's answer there, and then resolves with the order or with why it failed.
(() => {
  "use strict";
  const config = JSON.parse(document.getElementById("mc-storefront").textContent);
  const API = "/shop/api/";

  async function post(op, input, signal) {
    const response = await fetch(API + encodeURIComponent(op), {
      method: "POST",
      headers: { "content-type": "application/json", accept: "application/json" },
      body: JSON.stringify(input === undefined ? {} : input),
      signal,
    });
    try {
      return await response.json();
    } catch (e) {
      return { view: { error: { code: "store_error", message: `the store answered ${response.status}` } } };
    }
  }

  function show(body) {
    if (typeof body.html === "string") document.getElementById("view").innerHTML = body.html;
    if (typeof body.title === "string") document.title = body.title;
    const view = body.view || {};
    if (view.page) document.body.dataset.page = view.page;
    if (view.url && view.url !== location.pathname + location.search) history.pushState({}, "", view.url);
  }

  async function run(op, input, signal) {
    const body = await post(op, input, signal);
    show(body);
    return body.view;
  }

  function confirmWithBank(url, signal) {
    return new Promise((resolve, reject) => {
      const frame = document.createElement("iframe");
      frame.title = "Confirm the payment with your bank";
      frame.dataset.mc3ds = "loading";
      const finish = (settle) => {
        frame.remove();
        settle();
      };
      frame.addEventListener("load", () => {
        let doc = null;
        try {
          doc = frame.contentDocument;
        } catch (e) {
          doc = null;
        }
        if (doc === null || doc.location.href === "about:blank") return;
        if (doc.querySelector('form button[name="approved"]')) {
          frame.dataset.mc3ds = "waiting"; // the bank's page is showing: the buyer answers it
          return;
        }
        frame.dataset.mc3ds = "answered";
        post("payment_result", {}, signal).then(
          (body) => finish(() => resolve(body)),
          (error) => finish(() => reject(error)),
        );
      });
      if (signal) signal.addEventListener("abort", () => finish(() => reject(signal.reason)), { once: true });
      frame.src = url;
      (document.getElementById("challenge") || document.body).append(frame);
    });
  }

  async function placeOrder(input, options) {
    let body = await post("place_order", input, options.signal);
    show(body);
    const challenge = body.view && body.view.challenge;
    if (challenge && challenge.url) {
      body = await confirmWithBank(challenge.url, options.signal);
      show(body);
    }
    return body.view;
  }

  const special = { place_order: placeOrder };
  for (const definition of config.tools) {
    const execute = special[definition.name] || ((input, options) => run(definition.name, input, options.signal));
    document.modelContext
      .registerTool({ ...definition, execute })
      .catch((error) => console.error(`registering ${definition.name} failed: ${error.name}`));
  }

  document.addEventListener("submit", (event) => {
    const form = event.target;
    const name = form instanceof HTMLFormElement ? form.getAttribute("toolname") : null;
    if (!name) return;
    event.preventDefault();
    const input = Object.fromEntries(new FormData(form));
    const answer = run(name, input);
    if (event.agentInvoked) event.respondWith(answer);
  });
})();
