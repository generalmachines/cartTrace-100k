"""The replay kit's own behaviour, offline: model calls recovered from a trace, redacted values restored, worlds cut,
and dataset rows turned back into a bundle. A full replay runs when a dataset and its worlds are at hand."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from carttrace.repro.dataset import served_domain, turns_from_rows
from carttrace.repro.recorded import ANY_FIELD, live_values, phone_fields, recorded_calls, restore
from carttrace.repro.replay import mixed_spellings
from carttrace.repro.worlds import WITHHELD, withhold, world_manifest

Q = "MC" + "-"
"""A queue id's prefix, spelt so this file never holds one whole."""


def _turns() -> list[dict[str, object]]:
    return [
        {"actor": "shopper", "kind": "message", "text": "Find me boots", "seq": 1},
        {"actor": "agent", "kind": "note", "text": "thinking", "seed": 11, "usage": {"completion_tokens": 9}, "seq": 2},
        {
            "actor": "agent",
            "kind": "tool_call",
            "tool": "search",
            "arguments": {"q": "boots"},
            "call_id": "call-3",
            "seed": 11,
            "seq": 3,
        },
        {"actor": "tool", "kind": "tool_result", "tool": "search", "result": "{}", "seq": 4},
        {"actor": "agent", "kind": "message", "text": "Which size?", "seed": 12, "seq": 5},
    ]


def test_each_model_call_is_recovered_from_the_turns_that_share_its_seed() -> None:
    calls = recorded_calls(_turns())
    assert [len(c.tool_calls) for c in calls] == [1, 0]
    assert calls[0].reasoning == "thinking" and calls[0].tool_calls[0].arguments == {"q": "boots"}
    assert calls[1].text == "Which size?"


def test_the_personas_placeholders_are_put_back_and_nothing_else_changes() -> None:
    value = {"email": "email-0123456789ab@redacted.invalid", "phone": "phone-0123456789ab", "name": "Ana"}
    assert restore(value, email="ana@example.com", phone="+1 202-555-0101") == {
        "email": "ana@example.com",
        "phone": "+1 202-555-0101",
        "name": "Ana",
    }


def test_a_redacted_secret_takes_the_live_value_the_replayed_store_issued() -> None:
    seen = '{"id": "gid://shopify/Cart/abc123?key=s3cr3t"}'
    assert live_values({"cart_id": "gid://shopify/Cart/abc123?key=redacted"}, seen) == {
        "cart_id": "gid://shopify/Cart/abc123?key=s3cr3t"
    }
    assert live_values({"cart_id": "gid://shopify/Cart/other?key=redacted"}, seen) == {
        "cart_id": "gid://shopify/Cart/other?key=redacted"
    }


def test_withholding_leaves_a_model_number_that_contains_a_queue_id_alone() -> None:
    title = f"Hanabishi H{Q}1200XWHT Cooker"
    doc = {"title": title, "provenance": {"store": Q + "1200", "round": Q + "041~2"}}
    out = withhold(doc)
    assert out["title"] == title
    assert out["provenance"] == {"store": WITHHELD, "round": f"{WITHHELD}~2"}


def test_a_world_withholds_every_queue_store_and_capture_it_names() -> None:
    doc = {
        "provenance": {"twin_of": Q + "507", "pick": {"capture": "cv2-b3893bfea0f7"}},
        "policy": {"store": Q + "507"},
    }
    out = withhold(doc)
    assert out == {"provenance": {"twin_of": WITHHELD, "pick": {"capture": WITHHELD}}, "policy": {"store": WITHHELD}}


def test_a_world_withholds_the_recorded_store_its_agent_view_names_by_domain() -> None:
    doc = {"agent_view": {"store": "asl-shop.example.com", "ref": "shopify-ucp-mcp:asl-shop.example.com@0a1b2c3d"}}
    out = withhold(doc)
    assert out == {"agent_view": {"store": WITHHELD, "ref": f"shopify-ucp-mcp:{WITHHELD}@0a1b2c3d"}}
    assert withhold({"title": "Basl-ware.com mug"}) == {"title": "Basl-ware.com mug"}


def test_a_world_manifest_keeps_only_its_rows_and_names_its_store_by_pseudonym() -> None:
    manifest = {
        "plan": {"store_specs": {Q + "001": {"p": 1}, Q + "002": {"p": 2}}},
        "stores": {Q + "001": {"domain": "mc-001.open.sandbox.invalid"}, Q + "002": {}},
        "tasks": {Q + "001": {"t": {}}, Q + "002": {}},
        "profiles": {"pk-1": {}},
        "worlds": {"env-a": {}, "env-b": {}},
        "rows": [
            {"envpack": "env-a", "store_id": Q + "001", "profile": "pk-1"},
            {"envpack": "env-b", "store_id": Q + "002", "profile": "pk-1"},
        ],
    }
    names = {Q + "001": "ct-shoes-001", "mc-001.open.sandbox.invalid": "ct-001.example"}
    out = world_manifest(manifest, "env-a", names)
    assert [r["store_id"] for r in out["rows"]] == ["ct-shoes-001"]
    assert out["plan"]["store_specs"] == {"ct-shoes-001": {"p": 1}}
    assert out["stores"] == {"ct-shoes-001": {"domain": "ct-001.example"}}


def test_a_world_manifest_drops_the_other_top_up_stand_ins_of_its_round() -> None:
    manifest = {
        "plan": {"stores": [Q + "041~2", Q + "607~2"]},
        "stores": {Q + "607~2": {}},
        "tasks": {Q + "607~2": {}},
        "profiles": {},
        "worlds": {"env-a": {}},
        "rows": [{"envpack": "env-a", "store_id": Q + "607~2"}],
    }
    out = world_manifest(manifest, "env-a", {Q + "607~2": "ct-bags-007"})
    assert out["plan"]["stores"] == ["ct-bags-007"]


PHONE = "phone-3c2023163159"


def _two_field_call() -> dict[str, object]:
    payload = {"buyer": {"phone_number": PHONE}, "fulfillment_details": {"phone_number": PHONE}}
    return {
        "actor": "agent",
        "kind": "tool_call",
        "tool": "create",
        "call_id": "call-1",
        "seed": 1,
        "arguments": {"payload": payload},
    }


def test_a_phone_is_restored_per_field_when_the_agent_wrote_it_two_ways() -> None:
    spelled = {
        "buyer.phone_number": "+12025550126",
        "fulfillment_details.phone_number": "+1 202-555-0126",
        ANY_FIELD: "+1 202-555-0126",
    }
    out = restore(_two_field_call(), email=None, phone=spelled)
    assert out["arguments"]["payload"]["buyer"]["phone_number"] == "+12025550126"
    assert out["arguments"]["payload"]["fulfillment_details"]["phone_number"] == "+1 202-555-0126"
    assert restore("call me on " + PHONE, email=None, phone=spelled) == "call me on +1 202-555-0126"


def test_per_field_spellings_are_kept_only_when_the_request_is_as_long_as_recorded(tmp_path: Path) -> None:
    def wire(folder: Path, size: int) -> None:
        folder.mkdir(parents=True)
        record = {"source_call_id": "call-1", "request_headers": {"content-length": str(size)}}
        (folder / "wire.jsonl").write_text(json.dumps(record) + "\n")

    wire(tmp_path / "ep-1", 1030)  # the trace: one field three characters shorter than the first spelling
    wire(tmp_path / "try-0" / "ep-1", 1033)  # the replay that spelled both fields "+1 202-555-0126"
    fields = phone_fields([_two_field_call()])
    assert fields == {"call-1": {"buyer.phone_number": 1, "fulfillment_details.phone_number": 1}}
    found = mixed_spellings(
        ["+1 202-555-0126", "202-555-0126", "+12025550126"], fields, tmp_path / "ep-1", tmp_path / "try-0"
    )
    assert found and all(sorted(len(v) for k, v in f.items() if k != ANY_FIELD) == [12, 15] for f in found)


def test_the_engine_decides_by_rule_and_ships_no_decision_model() -> None:
    import importlib.util

    from carttrace.decide.router import Router

    assert not hasattr(Router, "prompt_for")
    for name in ("prompt", "verdict", "breaker", "fallback"):
        assert importlib.util.find_spec(f"carttrace.decide.{name}") is None


def test_dataset_rows_become_turns_that_name_their_calls_and_model_calls() -> None:
    turns = [
        {
            "seq": 3,
            "actor": "agent",
            "kind": "tool_call",
            "tool": "search",
            "arguments": '{"q": "boots"}',
            "model": "m",
            "seed": 7,
        },
        {"seq": 4, "actor": "tool", "kind": "tool_result", "tool": "search", "result": "{}", "tool_call_id": "call-3"},
        {"seq": 1, "actor": "shopper", "kind": "message", "text": "Find me boots"},
    ]
    out = turns_from_rows(turns)
    assert [t["seq"] for t in out] == [1, 3, 4]
    assert out[1]["call_id"] == "call-3" and out[1]["arguments"] == {"q": "boots"} and out[1]["seed"] == 7
    assert out[2]["tool_call_id"] == "call-3"


def test_dataset_bundle_spells_the_roster_domain_as_the_world_domain_and_nothing_else() -> None:
    records = [
        {"arguments": {"merchant_id": "ct-034.example"}, "result": '{"url": "https://ct-034.example/x"}'},
        {"text": "ct-1034.example and xct-034.example"},
    ]
    out = served_domain(records, "ct-034.example", "amber-square-store.example")
    assert out[0]["arguments"] == {"merchant_id": "amber-square-store.example"}
    assert json.loads(out[0]["result"]) == {"url": "https://amber-square-store.example/x"}
    assert out[1] == records[1]
    assert served_domain(records, "ct-034.example", None) is records


def test_a_rerun_endpoint_given_with_or_without_v1_is_called_once_at_v1() -> None:
    from carttrace.repro.cli import server_root

    for url in ("http://127.0.0.1:8000/v1", "http://127.0.0.1:8000/v1/", "http://127.0.0.1:8000"):
        assert server_root(url) == "http://127.0.0.1:8000"
    assert server_root("https://openrouter.ai/api/v1") == "https://openrouter.ai/api"
    assert server_root("https://example.test/v10") == "https://example.test/v10"


@pytest.mark.dataset
@pytest.mark.skipif(
    not os.environ.get("CARTTRACE_DATASET"),
    reason="set CARTTRACE_DATASET, CARTTRACE_WORLDS and CARTTRACE_EPISODE to replay a published trace",
)
def test_a_published_trace_replays_to_its_own_hashes(tmp_path: Path) -> None:
    argv = [
        sys.executable,
        "-m",
        "carttrace.repro.cli",
        "replay",
        os.environ["CARTTRACE_EPISODE"],
        "--dataset",
        os.environ["CARTTRACE_DATASET"],
        "--worlds",
        os.environ["CARTTRACE_WORLDS"],
        "--out",
        str(tmp_path),
    ]
    done = subprocess.run(argv, capture_output=True, text=True, check=False)  # noqa: S603
    assert done.returncode == 0, done.stdout + done.stderr
    assert json.dumps("REPRODUCED")[1:-1] in done.stdout
