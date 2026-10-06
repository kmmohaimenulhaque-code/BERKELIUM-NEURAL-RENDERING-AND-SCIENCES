"""Schema migration foundation (risk R6). Migrations are pure functions on JSON documents."""

from __future__ import annotations

from collections.abc import Callable

from .common import SCHEMA_VERSION

Migration = Callable[[dict], dict]
# (from_version) -> (to_version, function)
MIGRATIONS: dict[str, tuple[str, Migration]] = {}


class MigrationError(ValueError):
    pass


def register(from_version: str, to_version: str) -> Callable[[Migration], Migration]:
    def deco(fn: Migration) -> Migration:
        if from_version in MIGRATIONS:
            raise MigrationError(f"duplicate migration from {from_version}")
        MIGRATIONS[from_version] = (to_version, fn)
        return fn
    return deco


def migrate(doc: dict, target: str = SCHEMA_VERSION) -> dict:
    """Upgrade a JSON document step by step to ``target``. Never downgrades."""
    version = doc.get("schema_version")
    if version is None:
        raise MigrationError("document has no schema_version")
    seen = set()
    while version != target:
        if version in seen or version not in MIGRATIONS:
            raise MigrationError(f"no migration path from {version} to {target}")
        seen.add(version)
        nxt, fn = MIGRATIONS[version]
        doc = fn(dict(doc))
        doc["schema_version"] = nxt
        version = nxt
    return doc
