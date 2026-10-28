"""Layer 5: the text-only protocol agent, the reference agent, the deterministic shopper, and task generation.

- `service`: the contracts: Limits, TurnSink, Shopper and ShopperReply, ConsentRequest and ConsentReply, Supervisor
  and SupervisorVerdict, AgentResult, ProtocolAgent
- `qwen`: QwenProtocolAgent, the model's tool loop over the episode's surfaces
- `reference`: ReferenceAgent, which knows the hidden spec and uses only what the surfaces return
- `shopper`, `keywords`: DeterministicShopper, its keyword rule, the shopper.field decision behind it, and its consent
  to an order on a checkout row
- `wallet`: personas (data/agent/personas.yaml) and the Wallet the agent pays from, only after the shopper consents
- `profile`: the shopper's buyer profile a sandbox checkout row's agent is given in its context (D77)
- `checkout`: the store's checkout documents as the harness reads them, and a checkout row's harness tools
- `supervisor`: SupervisorFromDecisions, supervise.looping and supervise.stuck as verdicts
- `tasks`, `spec`: generate_tasks from fixtures and a category's intents, and the hidden spec (`mc.spec.v1`)
- `buyer`, `entry`, `reserved`: the spec's buyer block drawn from a persona, how each protocol's checkout takes the
  buyer's details, and the reserved contact values every persona and spec holds
- `dialects`: how an agent speaks UCP, ACP and WebMCP (tool names, argument and document shapes), each protocol's
  checkout entry, and ACP's delegated token issuer; chosen from the tools a surface lists
- `measures`: quantities and nutrients behind compare criteria (cheapest per unit, most protein per 100 g)
- `intents`, `intents_voice`, `personas`: generator 3's intents and the variety a persona brings (D20), the guarded
  model voice pass, and the persona generator behind data/agent/personas.yaml
- `brief`, `checks`, `intents_custom`: the store brief read from a capture by code, the niche checklists as data,
  and the per-store intent writer: a model writes free-form intents from the brief and the checks, code validates
  each against the capture and spreads them by quota (D27)
- `details`, `predicates`, `products`, `slots`, `nouns`, `text`, `tools`, `phrases`: the parts they share
- `cli`: `carttrace agents tasks | personas | reply | brief | intents write | intents show | checks`

PLUGINS lists the agents, the decisive details and the wallet's credential issuers by name, so the runner can choose an
agent per condition, a new detail is one module and one line, and so is a new credential kind (ACP, MPP, AP2). See
README.md in this directory for the map and the hidden-spec format.
"""

from carttrace.agents.brief import DATA_KINDS, StoreBrief, brief_of, brief_of_capture, cached_brief
from carttrace.agents.checkout import HARNESS_TOOLS, CheckoutBook, CheckoutHarness
from carttrace.agents.checks import Check, Checklists, StaticChecks, agent_checks, load_checklists, static_checks
from carttrace.agents.dialects import (
    AcpDialect,
    AcpEntry,
    DelegatedTokenIssuer,
    Dialect,
    WebMcpDialect,
    WebMcpEntry,
    WebsiteDialect,
    dialect_for,
    dialect_for_surface,
)
from carttrace.agents.entry import CheckoutEntry, Details, UcpEntry, entry_for
from carttrace.agents.intents import Variety, load_intent_words, task_writer, variety_for
from carttrace.agents.intents_custom import (
    CustomIntent,
    IntentSet,
    custom_task_writer,
    load_intent_set,
    needs_replica,
    save_intent_set,
    tasks_from_intents,
    write_intents,
)
from carttrace.agents.intents_voice import VoiceRunner, voice_tasks
from carttrace.agents.personas import PersonaConstraints, PersonaStyle, render_personas
from carttrace.agents.products import fixtures_from_snapshot, read_product, read_products
from carttrace.agents.qwen import QwenProtocolAgent
from carttrace.agents.reference import ReferenceAgent, UcpDialect
from carttrace.agents.service import (
    AgentResult,
    ChoiceRecord,
    ConsentReason,
    ConsentReply,
    ConsentRequest,
    DeclaredStatus,
    Limits,
    ProtocolAgent,
    Shopper,
    ShopperReply,
    Supervisor,
    SupervisorVerdict,
    TurnSink,
)
from carttrace.agents.shopper import DeterministicShopper, FieldDecider, disclosure_of, engine_field_decider, shopper_for
from carttrace.agents.spec import (
    ALL_INTENTS,
    BUYER_FIELDS,
    NEW_INTENTS,
    SPEC_SCHEMA,
    Address,
    Buyer,
    Decisive,
    HiddenSpec,
    Predicate,
    Voice,
    validate_spec,
)
from carttrace.agents.supervisor import SupervisorFromDecisions, decision_id
from carttrace.agents.tasks import GENERATOR, IMPOSSIBLE_SHARE, generate_tasks, validate_task, witnesses_of
from carttrace.agents.wallet import (
    CREDENTIAL_KINDS,
    ChallengePage,
    Consent,
    Credential,
    CredentialIssuer,
    CredentialKind,
    CredentialRequest,
    DelegatedTokenRequest,
    HandlerOffer,
    HandlerTokenIssuer,
    HandlerTokenRequest,
    HttpChallengePage,
    HttpShopPay,
    MandateRequest,
    PaymentChallengeRequest,
    Persona,
    Personas,
    Refusal,
    ShopPayTokenIssuer,
    ShopTokenRequest,
    Wallet,
    load_personas,
    wallet_for,
)

PLUGINS = {
    "agents.agent": {
        "qwen": "carttrace.agents.qwen:QwenProtocolAgent",
        "reference": "carttrace.agents.reference:ReferenceAgent",
    },
    "agents.detail": {
        "size_chart": "carttrace.agents.details.size_chart:SizeChartDetail",
        "allergens": "carttrace.agents.details.allergens:AllergenDetail",
        "warnings": "carttrace.agents.details.warnings:WarningDetail",
    },
    "agents.credential": {
        "ucp_handler_token": "carttrace.agents.wallet:HandlerTokenIssuer",
        "acp_delegated_token": "carttrace.agents.dialects.acp:DelegatedTokenIssuer",
        "mpp_credential": "carttrace.agents.dialects.mpp:ChallengeCredentialIssuer",
        "ap2_mandate": "carttrace.agents.dialects.ap2:MandateIssuer",
        "shop_pay_token": "carttrace.agents.wallet:ShopPayTokenIssuer",
    },
    "agents.checkout_entry": {
        "ucp": "carttrace.agents.entry:UcpEntry",
        "acp": "carttrace.agents.dialects.acp:AcpEntry",
        "webmcp": "carttrace.agents.dialects.webmcp:WebMcpEntry",
        "ap2": "carttrace.agents.dialects.ap2:Ap2Entry",
        "ap2_v1": "carttrace.agents.dialects.ap2:Ap2V1Entry",
        "ucp_checkout": "carttrace.agents.dialects.ucp_checkout:UcpCheckoutEntry",
    },
    "agents.dialect": {
        "mpp": "carttrace.agents.dialects.mpp:MppDialect",
        "ap2": "carttrace.agents.dialects.ap2:Ap2Dialect",
        "ap2_v1": "carttrace.agents.dialects.ap2:Ap2V1Dialect",
        "ucp": "carttrace.agents.dialects.ucp:UcpDialect",
        "ucp_checkout": "carttrace.agents.dialects.ucp_checkout:UcpCheckoutDialect",
        "acp": "carttrace.agents.dialects.acp:AcpDialect",
        "webmcp": "carttrace.agents.dialects.webmcp:WebMcpDialect",
        "website": "carttrace.agents.dialects.website:WebsiteDialect",
    },
}
"""`agents.credential` holds one issuer per credential kind the wallet hands out, behind the same consent check
(carttrace.agents.wallet): UCP handler tokens, ACP delegated tokens, MPP challenge credentials, AP2 mandate credentials
(D66) and Shop Pay single-use tokens (D73). `agents.checkout_entry` holds how each protocol's checkout takes the
buyer's details
(carttrace.agents.entry): UCP, ACP, WebMCP and AP2. `agents.dialect` holds how an agent speaks each protocol's tools and reads
its documents (carttrace.agents.dialects), chosen from the tools a surface lists, the first offered in this order (MPP's
tools sit beside UCP's, so MPP comes first); a new protocol is one module and a line in each table."""

__all__ = [
    "ALL_INTENTS",
    "BUYER_FIELDS",
    "CREDENTIAL_KINDS",
    "DATA_KINDS",
    "GENERATOR",
    "HARNESS_TOOLS",
    "IMPOSSIBLE_SHARE",
    "NEW_INTENTS",
    "PLUGINS",
    "SPEC_SCHEMA",
    "AcpDialect",
    "AcpEntry",
    "Address",
    "AgentResult",
    "Buyer",
    "ChallengePage",
    "Check",
    "Checklists",
    "CheckoutBook",
    "CheckoutEntry",
    "CheckoutHarness",
    "ChoiceRecord",
    "Consent",
    "ConsentReason",
    "ConsentReply",
    "ConsentRequest",
    "Credential",
    "CredentialIssuer",
    "CredentialKind",
    "CredentialRequest",
    "CustomIntent",
    "Decisive",
    "DeclaredStatus",
    "DelegatedTokenIssuer",
    "DelegatedTokenRequest",
    "Details",
    "DeterministicShopper",
    "Dialect",
    "FieldDecider",
    "HandlerOffer",
    "HandlerTokenIssuer",
    "HandlerTokenRequest",
    "HiddenSpec",
    "HttpChallengePage",
    "HttpShopPay",
    "IntentSet",
    "Limits",
    "MandateRequest",
    "PaymentChallengeRequest",
    "Persona",
    "PersonaConstraints",
    "PersonaStyle",
    "Personas",
    "Predicate",
    "ProtocolAgent",
    "QwenProtocolAgent",
    "ReferenceAgent",
    "Refusal",
    "ShopPayTokenIssuer",
    "ShopTokenRequest",
    "Shopper",
    "ShopperReply",
    "StaticChecks",
    "StoreBrief",
    "Supervisor",
    "SupervisorFromDecisions",
    "SupervisorVerdict",
    "TurnSink",
    "UcpDialect",
    "UcpEntry",
    "Variety",
    "Voice",
    "VoiceRunner",
    "Wallet",
    "WebMcpDialect",
    "WebMcpEntry",
    "WebsiteDialect",
    "agent_checks",
    "brief_of",
    "brief_of_capture",
    "cached_brief",
    "custom_task_writer",
    "decision_id",
    "dialect_for",
    "dialect_for_surface",
    "disclosure_of",
    "engine_field_decider",
    "entry_for",
    "fixtures_from_snapshot",
    "generate_tasks",
    "load_checklists",
    "load_intent_set",
    "load_intent_words",
    "load_personas",
    "needs_replica",
    "read_product",
    "read_products",
    "render_personas",
    "save_intent_set",
    "shopper_for",
    "static_checks",
    "task_writer",
    "tasks_from_intents",
    "validate_spec",
    "validate_task",
    "variety_for",
    "voice_tasks",
    "wallet_for",
    "witnesses_of",
    "write_intents",
]
