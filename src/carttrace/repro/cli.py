"""`carttrace replay` and `carttrace rerun`: one published CartTrace-100k trace, run again.

- `replay` runs the trace's own recorded agent choices against its world and says whether the store ends where the
  trace says (the events chain head, and the state hash where one is published). No model, no GPU, no network.
- `rerun` runs the same row (world, task, persona, surface, rail, seed) with your own model as the agent, through
  any OpenAI-compatible chat endpoint with tool calls, and writes a new trace of it.

A trace is a bundle folder (as the sandbox writes them, and as `--dataset` writes them from the dataset's tables).
Its world comes from the dataset's worlds/ folder (`--worlds`): envpacks/<world id>.json and manifests/<world id>.json.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

DEFAULT_WORLDS = Path("worlds")
DEFAULT_OUT = Path("var/replays")


def _settings(var: Path) -> Any:
    from carttrace.config import settings

    base = settings()
    return base.model_copy(update={"paths": base.paths.model_copy(update={"var": var})})


def _scratch_var(worlds: Path) -> Path:
    """A run folder whose envpacks/ is the worlds folder's, so the runner finds each world where it looks."""
    var = Path(tempfile.mkdtemp(prefix="carttrace-"))
    (var / "envpacks").symlink_to((worlds / "envpacks").resolve(), target_is_directory=True)
    return var


def _bundle(args: argparse.Namespace) -> Path:
    if args.dataset is not None:
        from carttrace.repro.dataset import write_bundle

        return write_bundle(args.dataset, args.trace, args.out / "from-dataset", worlds=args.worlds)
    return Path(args.trace)


def _manifest(worlds: Path, bundle: Path) -> Any:
    from carttrace.collect.manifest import seal
    from carttrace.repro.worlds import world_files

    header = json.loads((bundle / "episode.json").read_text())
    _, path = world_files(worlds, str(header["envpack"]))
    body = json.loads(path.read_text())
    body.pop("id", None)
    body.pop("schema", None)
    return seal(**body)


def cmd_replay(args: argparse.Namespace) -> int:
    from carttrace.repro.replay import replay

    bundle = _bundle(args)
    var = _scratch_var(args.worlds)
    result = asyncio.run(replay(_manifest(args.worlds, bundle), bundle, settings=_settings(var), out=args.out))
    print(f"episode   {result.episode_id}")
    print(f"ended     {result.end_reason} (trace: {result.published_end})")
    print(f"calls     {result.calls[1]} of {result.calls[0]} recorded model calls replayed")
    print(f"events    {'same chain head' if result.events[0] == result.events[1] else 'DIFFERENT'}")
    if result.state[0] is not None:
        print(f"state     {'same hash' if result.state[0] == result.state[1] else 'DIFFERENT'}")
    print(f"bundle    {result.bundle}")
    print("REPRODUCED" if result.same else "NOT REPRODUCED")
    return 0 if result.same else 1


def server_root(url: str) -> str:
    """The server root of an OpenAI-compatible endpoint. The chat client adds /v1/chat/completions itself, so the
    .../v1 that servers and hosted APIs print as their base URL is dropped instead of being sent as /v1/v1/..."""
    root = url.rstrip("/")
    return root.removesuffix("/v1")


def cmd_rerun(args: argparse.Namespace) -> int:
    from carttrace.collect.runner import run_manifest
    from carttrace.repro.replay import one_row, replay_settings
    from carttrace.storage.local import LocalBlobStore

    bundle = _bundle(args)
    header = json.loads((bundle / "episode.json").read_text())
    var = _scratch_var(args.worlds)
    base = replay_settings(_settings(var))
    agent = base.endpoints["agent"].model_copy(
        update={"base_url": server_root(args.base_url), "model": args.model, "api_key_env": args.api_key_env}
    )
    settings = base.model_copy(update={"endpoints": {**base.endpoints, "agent": agent}})
    from carttrace.repro.published import load_published, published_ids
    from carttrace.repro.worlds import world_files

    row = one_row(_manifest(args.worlds, bundle), str(header["episode_id"]))
    out = args.out / "reruns"
    with published_ids():  # the withheld world answers to its published id (carttrace.repro.published)
        load_published(world_files(args.worlds, str(header["envpack"]))[0])
        asyncio.run(
            run_manifest(row, settings=settings, live_flag=False, store=LocalBlobStore(out), handle_signals=False)
        )
    found = sorted(out.rglob(f"{header['episode_id']}/outcome.json"))
    if not found:
        print("the rerun wrote no outcome; see the run's status under", var, file=sys.stderr)
        return 1
    outcome = json.loads(found[-1].read_text())
    print(f"ended     {outcome.get('end_reason')}")
    print(f"cart      {json.dumps(outcome.get('final_cart'))[:300]}")
    print(f"bundle    {found[-1].parent}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="carttrace", description=__doc__.split("\n\n")[0] if __doc__ else None)
    sub = parser.add_subparsers(dest="command", required=True)

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("trace", help="a bundle folder, or an episode id with --dataset")
        p.add_argument("--dataset", type=Path, help="the dataset's folder (holding data/<table>/...parquet)")
        p.add_argument("--worlds", type=Path, default=DEFAULT_WORLDS, help="the worlds folder (default: worlds)")
        p.add_argument("--out", type=Path, default=DEFAULT_OUT, help="where new bundles go (default: var/replays)")

    replay = sub.add_parser("replay", help="replay a trace's recorded agent choices; exit 0 when it reproduces")
    common(replay)
    replay.set_defaults(func=cmd_replay)
    rerun = sub.add_parser("rerun", help="run a trace's row again with your own model as the agent")
    common(rerun)
    rerun.add_argument(
        "--base-url",
        required=True,
        help="an OpenAI-compatible endpoint, e.g. http://127.0.0.1:8000/v1 (a trailing /v1 is optional)",
    )
    rerun.add_argument("--model", required=True, help="the model name the endpoint serves")
    rerun.add_argument("--api-key-env", help="the environment variable holding the endpoint's API key, if any")
    rerun.set_defaults(func=cmd_rerun)
    args = parser.parse_args(argv)
    os.environ.setdefault("CARTTRACE_GPU_MAX_INFLIGHT", "4")
    from carttrace.core.errors import MCError

    try:
        return int(args.func(args))
    except MCError as exc:  # a missing folder or file: say what to do, not a traceback
        print(f"carttrace: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
