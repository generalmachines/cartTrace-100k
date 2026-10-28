// Seeded randomness for a sandbox page (carttrace.sandbox.browser.webmcp.determinism_source): the same episode draws the same
// numbers. The clock is frozen separately, by Playwright's clock API.
(() => {
  "use strict";
  let state = __SEED__ >>> 0;
  // mulberry32
  const next = () => {
    state = (state + 0x6d2b79f5) | 0;
    let t = Math.imul(state ^ (state >>> 15), 1 | state);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
  Object.defineProperty(Math, "random", { value: next, configurable: true, writable: true });
  const fill = (array) => {
    const bytes = new Uint8Array(array.buffer, array.byteOffset, array.byteLength);
    for (let i = 0; i < bytes.length; i += 1) bytes[i] = Math.floor(next() * 256);
    return array;
  };
  const uuid = () => {
    const b = fill(new Uint8Array(16));
    b[6] = (b[6] & 0x0f) | 0x40;
    b[8] = (b[8] & 0x3f) | 0x80;
    const h = [...b].map((x) => x.toString(16).padStart(2, "0")).join("");
    return `${h.slice(0, 8)}-${h.slice(8, 12)}-${h.slice(12, 16)}-${h.slice(16, 20)}-${h.slice(20)}`;
  };
  if (typeof Crypto !== "undefined") {
    Object.defineProperty(Crypto.prototype, "getRandomValues", { value: fill, configurable: true, writable: true });
    Object.defineProperty(Crypto.prototype, "randomUUID", { value: uuid, configurable: true, writable: true });
  }
})();
