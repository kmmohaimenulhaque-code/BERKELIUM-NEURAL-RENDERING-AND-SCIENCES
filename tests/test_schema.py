import json

import pytest
from pydantic import ValidationError

from berkelium.schema import DesignProposal, DesignRecord, sha256_of
from berkelium.schema.export import EXPORTED, export_all, json_schema
from berkelium.schema.migrations import MigrationError, migrate

MINIMAL = {"structure": {"components": [{"id": "pair", "kind": "cem", "cem": "gear.spur_pair@0.1",
                                         "requirements": {"ratio": 2.0}}]}}


def test_proposal_roundtrip_and_hash_stable():
    p = DesignProposal.model_validate(MINIMAL)
    again = DesignProposal.model_validate_json(p.model_dump_json())
    assert sha256_of(p) == sha256_of(again)


def test_proposal_cannot_carry_evaluation():
    bad = dict(MINIMAL, evaluation={"validation": {"summary": "pass"}})
    with pytest.raises(ValidationError):
        DesignProposal.model_validate(bad)


def test_unknown_fields_rejected_deep():
    bad = json.loads(json.dumps(MINIMAL))
    bad["structure"]["components"][0]["validated"] = True
    with pytest.raises(ValidationError):
        DesignProposal.model_validate(bad)


def test_bad_expression_in_constraint_rejected():
    bad = dict(MINIMAL, specification={"constraints": [{"id": "c1", "expr": "a <"}]})
    with pytest.raises(ValidationError):
        DesignProposal.model_validate(bad)


def test_bad_unit_rejected():
    bad = dict(MINIMAL, specification={"requirements": [
        {"id": "r", "quantity": "torque", "comparator": ">=", "target": {"value": 1, "unit": "furlongs"}}]})
    with pytest.raises(ValidationError):
        DesignProposal.model_validate(bad)


def test_relation_refs_checked():
    bad = json.loads(json.dumps(MINIMAL))
    bad["structure"]["relations"] = [{"id": "m", "kind": "mesh", "a": "pair.a", "b": "nope.b"}]
    with pytest.raises(ValidationError):
        DesignProposal.model_validate(bad)


def test_record_physically_validated_requires_l7_evidence():
    with pytest.raises(ValidationError):
        DesignRecord.model_validate({"id": "x", "intent": {}, "specification": {}, "structure": MINIMAL["structure"],
                                     "provenance": {"author": "system"}, "proposal_hash": "0",
                                     "evaluation": {"validation": {"physically_validated": True}}})


def test_json_schema_export(tmp_path):
    for name in EXPORTED:
        s = json_schema(name)
        assert s["$schema"].endswith("2020-12/schema")
    paths = export_all(tmp_path)
    assert len(paths) == len(EXPORTED)
    # deterministic output
    first = [p.read_text() for p in paths]
    assert first == [p.read_text() for p in export_all(tmp_path)]


def test_migration_foundation():
    assert migrate({"schema_version": "0.1.0"})["schema_version"] == "0.1.0"
    with pytest.raises(MigrationError):
        migrate({"schema_version": "0.0.1"})
