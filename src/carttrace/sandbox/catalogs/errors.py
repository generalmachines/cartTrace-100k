"""Typed errors for catalog sources, each with the next action in its hint.

Why separate types. A caller treats a restricted catalog (a data-rights refusal, never retried), a file whose bytes do
not match its pin (upstream moved or the download is corrupt), a download over budget and a source that was never
fetched differently, so each is its own type under the matching base from `carttrace.core.errors`.
"""

from __future__ import annotations

from carttrace.core.errors import NotFoundError, SafetyError, ValidationError


class CatalogError(ValidationError):
    """A source manifest, category map or catalog record that is malformed or inconsistent."""


class RestrictedCatalogError(SafetyError):
    """A source whose licence forbids commercial use was asked to fetch, index or build a store without the explicit
    research allowance (settings.sandbox.allow_restricted_catalogs)."""


class IntegrityError(ValidationError):
    """A downloaded or local file whose sha256 or size is not the one its source manifest pins."""


class BudgetError(ValidationError):
    """A download larger than its source's budget or than the total catalog budget."""


class NotFetchedError(NotFoundError):
    """A source's raw files are not on disk yet."""


class NotCoveredError(NotFoundError):
    """A source has no products, or no store, in the category asked for."""
