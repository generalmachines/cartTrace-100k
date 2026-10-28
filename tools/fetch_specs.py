"""Download the payment-handler specifications this repository may not redistribute: Google Pay's and the
Shopify-hosted card and Shop Pay handlers' (no licence file; their publishers' own terms apply).

Each is a source in protocols.lock.toml, pinned by URL and sha256: this fetches exactly the locked files into vendor/
and refuses any whose bytes differ from the lock. They document the handlers' wire shapes; a replay runs without them.

    uv run python tools/fetch_specs.py          # fetch what is missing
    uv run python tools/fetch_specs.py --check  # offline: are they here and intact?
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from carttrace.protocols.lock import read_lock  # noqa: E402
from carttrace.protocols.paths import ProtocolPaths  # noqa: E402
from carttrace.protocols.vendor import check, fetch  # noqa: E402

HOSTED = ("com.google.pay", "dev.shopify.card", "dev.shopify.shop_pay")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0] if __doc__ else None)
    parser.add_argument("--check", action="store_true", help="offline: report whether they are fetched and intact")
    args = parser.parse_args(argv)
    paths = ProtocolPaths.at(ROOT)
    ids = [s.id for s in read_lock(paths.lock).source if s.protocol in HOSTED]
    if args.check:
        report = check(paths, only=ids)
        print("\n".join([*report.problems, *[f"warning: {w}" for w in report.warnings], report.summary()]))
        return 0 if report.ok and not report.warnings else 1
    result = fetch(paths, only=ids)
    drifted = [r.id for r in result.results if r.status == "failed" and "upstream bytes differ" in r.message]
    if drifted:
        # Shopify edits files at its dated URLs (2026-10-23: a title, a description and a `requires` block), so a
        # source can drift after its lock. Replays do not read these specs, so this is a note, not a failure.
        print("\n".join(f"{i}: upstream changed since the lock; not fetched (replays do not need it)" for i in drifted))
    others = [r for r in result.results if r.id not in drifted]
    print("\n".join(f"{r.id}: {r.status}" + (f" ({r.message})" if r.message else "") for r in others))
    return 0 if all(r.status != "failed" for r in others) else 1


if __name__ == "__main__":
    raise SystemExit(main())
