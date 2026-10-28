"""Layer 3: the trace bundle (mc.trace.v1), its writer, reader and redaction. See schema.py for the layout.

- `schema`: the bundle contract every consumer depends on
- `layout`: file names, storage keys (bundle_prefix), the TraceStore shape, and reading bundle JSON as models
- `writer`: TraceWriter stages a bundle and uploads it with bundle.json last; recover_staged finishes crashed ones
- `reader`: scan() for summaries, open_bundle() for one bundle with lazy logs, verify() and a merged timeline
- `events`: verify_events, the sandbox event chain of a bundle's events.jsonl recomputed from the bundle alone (D33)
- `redact`: REDACTION_RULES, the one table of what never reaches a trace
- `provenance`: provenance_of(header), live, replica or twin: the one rule that keeps stand-in outcomes out of a
  store's own results
- `rights`: live_rights(platform), the D10 rights of anything taken from a live store
- `carts`, `products`: the one UCP cart converter and the one UCP catalog product reader, for live and sandbox alike
- `capture`: StoreCapture (mc.capture.v1), what one read-only live session saw of a store, the ground truth its
  replica is built from (D12), saved under <paths.var>/captures/<store>/<capture id>.json
- `inventory`: StoreDataInventory (mc.inventory.v1), what one store gives agents element by element on every surface
  (the live data audit, D19), saved under <paths.var>/inventory/<store>/<day>.json
- `retention`: purge_raw for live Shopify bundles' raw logs, purge_captures for store captures, purge_inventories for
  data inventories (`carttrace trace purge`)
- `prune`: tombstones (D79): a crashed or judge-VOID bundle keeps its header, outcome, bundle.json, newest judge card
  and pruned.json, and loses its logs (`carttrace store prune`)
- `cli`: `carttrace trace scan | verify | recover | purge`
"""

from carttrace.trace.capture import CAPTURES_DIR, CaptureBudget, StoreCapture, load_capture, seal_capture
from carttrace.trace.errors import ERROR_TYPES, error_type_of
from carttrace.trace.events import EventsCheck, verify_events
from carttrace.trace.layout import (
    AGENT_FILE,
    BUNDLE_FILES,
    DECISIONS_FILE,
    EPISODE_FILE,
    EPISODES_PREFIX,
    EVENTS_FILE,
    EXPOSURE_FILE,
    LOG_FILES,
    MANIFEST_FILE,
    OPTIONAL_FILES,
    OUTCOME_FILE,
    PRUNED_FILE,
    STATE_FILE,
    TURNS_FILE,
    WIRE_FILE,
    TraceStore,
    bundle_prefix,
    parse_json_as,
    validate_as,
)
from carttrace.trace.pages import CLEANER, KeptPage, keep_page
from carttrace.trace.provenance import (
    PROVENANCES,
    STAND_IN,
    STORE_PROVENANCES,
    Provenance,
    ResultMode,
    agent_of,
    mode_of,
    provenance_of,
    store_source_of,
    surface_of,
)
from carttrace.trace.reader import Bundle, Record, completed_episodes, open_bundle, scan
from carttrace.trace.redact import (
    PLACEHOLDER_DOMAIN,
    REDACTED,
    REDACTION_RULES,
    SECRET_QUERY_PARAMS,
    RedactionRule,
    redact_json,
    redact_text,
    redact_wire,
)
from carttrace.trace.schema import (
    SCHEMA,
    SCHEMA_MINOR,
    AgentDefinition,
    BundleManifest,
    BundleSummary,
    CallCost,
    Cart,
    CartLine,
    CheckoutFault,
    CheckoutFaultDraw,
    Condition,
    EpisodeHeader,
    ExposureCount,
    ExposureMask,
    FaultFiring,
    ModelRef,
    Outcome,
    Rights,
    SandboxEvent,
    SoldOutRate,
    StateDigest,
    StateSnapshot,
    StoreErrorRate,
    StoreRef,
    StoreReplies,
    SurfaceRef,
    TaskRef,
    Turn,
    TurnOrigin,
    WireRecord,
)
from carttrace.trace.writer import COUNT_KEYS, RecoveryReport, StagedBundle, TraceWriter, recover_staged

__all__ = [
    "AGENT_FILE",
    "BUNDLE_FILES",
    "CAPTURES_DIR",
    "CLEANER",
    "COUNT_KEYS",
    "DECISIONS_FILE",
    "EPISODES_PREFIX",
    "EPISODE_FILE",
    "ERROR_TYPES",
    "EVENTS_FILE",
    "EXPOSURE_FILE",
    "LOG_FILES",
    "MANIFEST_FILE",
    "OPTIONAL_FILES",
    "OUTCOME_FILE",
    "PLACEHOLDER_DOMAIN",
    "PROVENANCES",
    "PRUNED_FILE",
    "REDACTED",
    "REDACTION_RULES",
    "SCHEMA",
    "SCHEMA_MINOR",
    "SECRET_QUERY_PARAMS",
    "STAND_IN",
    "STATE_FILE",
    "STORE_PROVENANCES",
    "TURNS_FILE",
    "WIRE_FILE",
    "AgentDefinition",
    "Bundle",
    "BundleManifest",
    "BundleSummary",
    "CallCost",
    "CaptureBudget",
    "Cart",
    "CartLine",
    "CheckoutFault",
    "CheckoutFaultDraw",
    "Condition",
    "EpisodeHeader",
    "EventsCheck",
    "ExposureCount",
    "ExposureMask",
    "FaultFiring",
    "KeptPage",
    "ModelRef",
    "Outcome",
    "Provenance",
    "Record",
    "RecoveryReport",
    "RedactionRule",
    "ResultMode",
    "Rights",
    "SandboxEvent",
    "SoldOutRate",
    "StagedBundle",
    "StateDigest",
    "StateSnapshot",
    "StoreCapture",
    "StoreErrorRate",
    "StoreRef",
    "StoreReplies",
    "SurfaceRef",
    "TaskRef",
    "TraceStore",
    "TraceWriter",
    "Turn",
    "TurnOrigin",
    "WireRecord",
    "agent_of",
    "bundle_prefix",
    "completed_episodes",
    "error_type_of",
    "keep_page",
    "load_capture",
    "mode_of",
    "open_bundle",
    "parse_json_as",
    "provenance_of",
    "recover_staged",
    "redact_json",
    "redact_text",
    "redact_wire",
    "scan",
    "seal_capture",
    "store_source_of",
    "surface_of",
    "validate_as",
    "verify_events",
]
