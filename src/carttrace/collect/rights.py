"""The data rights every bundle the collector writes carries in its header (decision D10).

Why decided here, before the first record. A bundle's rights travel with it into every judge run, export and report,
and the export and publication steps refuse what its flags do not allow. So the flags are set from what the episode
is, before it starts, and never default to allowed:
- a live Shopify store is under Shopify's API terms (2.3.24 bars training on anything derived from its APIs without
  consent), so training and publication are both false, and `carttrace trace purge` later removes its raw logs;
- another live platform gets its own terms name, and both flags false;
- a sandbox store gets what its catalog's licence allows, through carttrace.sandbox.catalogs, except that a restricted
  catalog (Lazada, decision D11: internal research only) is never trained on or published, whatever the licence data
  says, and a sandbox bundle never claims a platform's terms.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from carttrace.core.errors import ValidationError
from carttrace.trace.rights import live_rights, platform_terms
from carttrace.trace.schema import Rights

RESTRICTED_BASIS = "D11: restricted catalog, internal research only; never trained on or published"


def sandbox_rights(licence: Rights | Mapping[str, Any], *, restricted: bool) -> Rights:
    """Rights for a sandbox episode from its catalog's licence fields. Raises ValidationError for fields that are not
    Rights fields, because a guessed flag could allow what the licence does not."""
    fields = licence.model_dump() if isinstance(licence, Rights) else dict(licence)
    unknown = sorted(set(fields) - set(Rights.model_fields))
    if unknown:
        raise ValidationError(
            f"licence_rights returned fields that are not Rights fields: {unknown}",
            hint="return carttrace.trace.Rights fields only: store_source, platform_terms, merchant_consent, "
            "training_allowed, publication_allowed, basis",
        )
    fields["platform_terms"] = "sandbox"
    fields.setdefault("store_source", "sandbox")
    if restricted:
        fields["training_allowed"] = False
        fields["publication_allowed"] = False
        fields["basis"] = "; ".join(b for b in (str(fields.get("basis") or ""), RESTRICTED_BASIS) if b)
    return Rights.model_validate(fields)


__all__ = ["RESTRICTED_BASIS", "live_rights", "platform_terms", "sandbox_rights"]
