"""Licence records for catalog sources, and the trace `Rights` they grant.

Why a record per source, carried into every bundle. Decision D10: every bundle says what its data may be used for,
and nothing defaults to allowed. A sandbox store built from a catalog inherits the catalog's licence, so the licence
is data (the `licence` block of `data/catalogs/<source>.json`), checked here for combinations that would leak rights
the licence does not grant:

- a restricted source (Lazada, decision D11: no licence covers the data beyond personal shopping use) never allows
  commercial use, training or publication, whatever its record says;
- a source whose licence forbids commercial use must be marked restricted, because everything this lab does is
  commercial;
- training needs commercial use;
- a share-alike source (ODbL: Open Food Facts, Open Beauty Facts) may not allow training until a lawyer has read the
  share-alike clause against trained weights and the record cites that review in `legal_review`.

`licence_rights(source)` turns a record into the `carttrace.trace.schema.Rights` a bundle header carries: store_source is
the source name and platform_terms is "sandbox". This is not legal advice; the flags record decisions the team made.
"""

from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from carttrace.config.schema import Settings
from carttrace.sandbox.catalogs.settings import CatalogSettings
from carttrace.trace.schema import Rights


class Licence(BaseModel):
    """What a catalog source's licence lets us do with it, and why we believe so."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    spdx: str = Field(min_length=1)
    """SPDX identifier ("CC-BY-4.0", "MIT", "ODbL-1.0", "Apache-2.0"), or LicenseRef-<name> for terms that have none."""
    name: str = Field(min_length=1)
    url: str | None = None
    """Where the licence text lives; None for platform terms we cite in `basis` instead."""
    attribution: str = ""
    """Credit to give wherever the data or anything built from it is shown or redistributed."""
    commercial_use: bool
    training_allowed: bool
    """May a model be trained or fine-tuned on bundles built from this source."""
    publication_allowed: bool
    """May bundle content built from this source be published (not just scores computed from it)."""
    share_alike: bool = False
    """Derived databases must carry the same licence (ODbL 4.4, CC BY-SA)."""
    restricted: bool = False
    """Internal research only, behind settings.sandbox.allow_restricted_catalogs."""
    verified: str = Field(min_length=1)
    """When and how the terms were checked, e.g. "2026-10-11: licence file in the archive"."""
    basis: str = Field(min_length=1)
    """Why the flags are what they are, in a sentence a reviewer can check."""
    legal_review: str = ""
    """Reference to a lawyer's reading; required before a share-alike source may allow training."""

    @model_validator(mode="after")
    def _grants_only_what_the_terms_allow(self) -> Self:
        if self.restricted and (self.commercial_use or self.training_allowed or self.publication_allowed):
            raise ValueError("a restricted licence allows no commercial use, training or publication")
        if not self.commercial_use and not self.restricted:
            raise ValueError("a licence without commercial use must be marked restricted")
        if self.training_allowed and not self.commercial_use:
            raise ValueError("training needs commercial use")
        if self.share_alike and self.training_allowed and not self.legal_review:
            raise ValueError("a share-alike licence allows training only with a cited legal_review")
        return self


def rights_for(source_name: str, licence: Licence) -> Rights:
    """The data rights a sandbox bundle built from this source carries."""
    terms = f"{licence.spdx} ({licence.url})" if licence.url else licence.spdx
    return Rights(
        store_source=source_name,
        platform_terms="sandbox",
        merchant_consent="none",
        training_allowed=licence.training_allowed and not licence.restricted,
        publication_allowed=licence.publication_allowed and not licence.restricted,
        basis=f"catalog {source_name}: {terms}; {licence.basis}",
    )


def licence_rights(source_name: str, *, settings: Settings | CatalogSettings | None = None) -> Rights:
    """The Rights fields for bundles built from `source_name`, read from its checked-in source manifest."""
    from carttrace.sandbox.catalogs.manifest import load_manifest

    manifest = load_manifest(source_name, settings=settings)
    return rights_for(manifest.name, manifest.licence)


def strictest(*rights: Rights) -> Rights:
    """Rights that allow only what every one of `rights` allows. Used when a pack's recorded rights meet the
    source's current licence: a licence that has since become stricter wins."""
    if not rights:
        return Rights()
    first = rights[0]
    return first.model_copy(
        update={
            "training_allowed": all(r.training_allowed for r in rights),
            "publication_allowed": all(r.publication_allowed for r in rights),
            "basis": " | ".join(dict.fromkeys(r.basis for r in rights if r.basis)),
        }
    )
