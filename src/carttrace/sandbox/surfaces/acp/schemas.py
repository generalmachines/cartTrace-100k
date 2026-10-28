"""The ACP release's own JSON Schemas: what each request is checked against, and what each reply must satisfy.

Why the release's own. A request valid at one API-Version can be refused at the next (2026-01-30 dropped `items` for
`line_items`, and an item's quantity with it), so every body is checked against the `$defs` of the vendored
`schema.agentic_checkout.json` or `schema.delegate_payment.json` of the version the store speaks, and the contract
tests check every reply against the same files.

A known upstream defect, read the way its authors meant it. The 2025-09-29 schema writes the item quantity as
`"minimum": 0, "exclusiveMinimum": true`, an OpenAPI 3.0 flag inside a Draft 2020-12 document, where
`exclusiveMinimum` must be a number. A strict 2020-12 validator compares the quantity with `true` (that is, 1) and
refuses a quantity of one. The validator here reads a boolean `exclusiveMinimum` or `exclusiveMaximum` as OpenAPI 3.0
did, a flag that makes `minimum` or `maximum` strict, and treats numbers exactly as 2020-12 does
(tests/sandbox/surfaces/acp/test_schemas.py pins it).
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from functools import cache
from typing import Any, Final, Literal
from urllib.parse import urljoin

import jsonschema
from jsonschema.exceptions import ValidationError as SchemaError

from carttrace.protocols.bundle import bundle
from carttrace.protocols.refs import tree_of
from carttrace.protocols.registry import DRAFT_2020_12, registry
from carttrace.sandbox.surfaces.problems import Problem, problems

Api = Literal["checkout", "psp"]
FILES: Mapping[Api, str] = {
    "checkout": "spec/{v}/json-schema/schema.agentic_checkout.json",
    "psp": "spec/{v}/json-schema/schema.delegate_payment.json",
}
"""The vendored schema bundle of each ACP API: the merchant's checkout API and the PSP's delegate payment API."""


def _flag_or_number(strict: str, bound: str) -> Any:
    def check(validator: Any, value: Any, instance: Any, schema: Mapping[str, Any]) -> Iterator[SchemaError]:
        if not validator.is_type(instance, "number"):
            return
        if isinstance(value, bool):
            limit = schema.get(bound)
            if not value or isinstance(limit, bool) or not isinstance(limit, int | float):
                return
        else:
            limit = value
        lower = strict == "exclusiveMinimum"
        if instance <= limit if lower else instance >= limit:
            words = "more" if lower else "less"
            yield SchemaError(f"{instance!r} is not {words} than {limit!r}", validator=strict, validator_value=limit)

    return check


AcpValidator = jsonschema.validators.extend(
    jsonschema.Draft202012Validator,
    {
        "exclusiveMinimum": _flag_or_number("exclusiveMinimum", "minimum"),
        "exclusiveMaximum": _flag_or_number("exclusiveMaximum", "maximum"),
    },
)
"""Draft 2020-12, with a boolean exclusive bound read as the OpenAPI 3.0 flag it was written as."""


@cache
def validator(version: str, api: Api, name: str) -> Any:
    """A validator for one `$defs` entry of one ACP version, resolving every `$ref` inside the vendored tree."""
    relpath = FILES[api].format(v=version)
    refs = registry().refs("acp", version, tree_of(relpath))
    root = {"$schema": DRAFT_2020_12, "$ref": f"{refs.base_of(relpath)}#/$defs/{name}"}
    return AcpValidator(root, registry=refs.registry)


def check(version: str, api: Api, name: str, instance: Any) -> list[Problem]:
    """Every problem `instance` has against the definition, as RFC 9535 paths and words from the schema."""
    return problems(validator(version, api, name), instance)


def errors(version: str, api: Api, name: str, instance: Any) -> list[str]:
    """Every validation error as "<path>: <message>", sorted, or an empty list: what contract tests assert on."""
    out = []
    for err in validator(version, api, name).iter_errors(instance):
        where = "/" + "/".join(str(p) for p in err.absolute_path)
        out.append(f"{where}: {err.message}")
    return sorted(out)


def defines(version: str, api: Api, name: str) -> bool:
    """True when the version's schema bundle defines `name` (CancelSessionRequest arrived at 2025-12-12)."""
    doc = registry().document("acp", version, FILES[api].format(v=version))
    return isinstance(doc, Mapping) and name in (doc.get("$defs") or {})


@cache
def inlined(version: str, api: Api, name: str) -> dict[str, Any]:
    """A definition with every reference inlined, for a tool's input schema: an agent cannot follow a `$ref`."""
    relpath = FILES[api].format(v=version)
    reg = registry()
    refs = reg.refs("acp", version, tree_of(relpath))
    doc = reg.document("acp", version, relpath)
    base = refs.base_of(relpath)
    ident = doc.get("$id") if isinstance(doc, Mapping) else None
    if isinstance(ident, str):
        base = urljoin(base, ident)
    out: dict[str, Any] = bundle({"$ref": f"#/$defs/{name}"}, base, refs.registry)
    return out


PSP_ERROR_FIELDS: Final = ("type", "code", "message", "param", "supported_versions")
"""The members a delegate_payment Error may carry (additionalProperties false)."""
PSP_CODE_FOR: Final[Mapping[str, str]] = {
    "rate_limited": "too_many_requests",
    "idempotency_key_required": "idempotency_conflict",
    "idempotency_in_flight": "idempotency_conflict",
    "invalid_idempotency_key": "idempotency_conflict",
}
"""Our reasons that have a near code in the release's enum, before the fallback."""


def psp_error(body: Mapping[str, Any], version: str) -> dict[str, Any]:
    """A delegate_payment Error as the release's schema allows it: `type` and `code` from its enums, our own reason
    (an unauthorized key, a body that is not JSON, a shopper who said no) kept at the front of `message`, nothing
    the schema does not define. The v1 review's #25: the PSP and the wallet sent codes outside the enum."""
    props = inlined(version, "psp", "Error").get("properties", {})
    codes = list(props.get("code", {}).get("enum") or ())
    types = list(props.get("type", {}).get("enum") or ())
    kind = str(body.get("type") or "invalid_request")
    code = str(body.get("code") or "")
    message = str(body.get("message") or "")
    if types and kind not in types:
        kind = "invalid_request"
    if codes and code not in codes:
        near = PSP_CODE_FOR.get(code)
        fallback = "too_many_requests" if kind in ("rate_limit_exceeded", "service_unavailable") else "invalid_card"
        mapped = near if near is not None and near in codes else fallback
        message = f"{code}: {message}" if code else message
        code = mapped
    out: dict[str, Any] = {"type": kind, "code": code, "message": message}
    out.update({k: body[k] for k in ("param", "supported_versions") if k in body})
    return out
