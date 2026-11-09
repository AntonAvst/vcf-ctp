"""
identity_record.py — which id each temp_id of each session ended up with
(ToDo item 6). Replaces reid_registry.known_temp_ids.

Why
───
known_temp_ids was a JSON list per cow that reconcile only ever appended to.
A re-run that changed a temp_id's id added a second entry under the new cow
and left the old one in place, so readers (display_tracks.py, the Cosine
Resolver's temp-gallery step, assign_identity.py) could see a stale id, and
which one they got depended on row order. Every lookup also scanned and
parsed every cow's JSON.

session_assignments
───────────────────
One row per (session_id, raw temp_id, written_by):

    session_id     session the temp_id belongs to
    temp_id        RAW temp_id, as in raw_tracks — every fragment of a stitched
                   or merged track gets a row, so display_tracks.py can show it
    final_temp_id  the id reconcile used for it (stitching + Duplicate Resolver);
                   NULL on link rows
    real_id        AnimalId, or a negative synthetic id
    source         manual | kinetic | boundary_rendezvous | cosine | legacy
    written_by     session whose reconcile run wrote the row

A run REPLACES every row it wrote before (written_by = this session) — its
own temp_ids and its link rows — so nothing goes stale on re-runs.

Link rows: when the Cosine Resolver matches a track to another session's
unresolved track and mints a synthetic id for the pair, the OTHER session's
temp_id is recorded too: session_id = that session, written_by = this one.
A lookup prefers the session's own row (written_by = session_id) and falls
back to the newest link.

Existing DBs: the first ensure() copies known_temp_ids into the table as
source 'legacy' rows (written_by = the entry's own session), skipping any
(session, temp_id) that known_temp_ids gives two different ids — those are
exactly the stale entries; re-run those sessions. The known_temp_ids column
is left in place but no longer written.

match_method
────────────
reid_registry.match_method is the BEST source that has ever identified the
cow, by identity.source_priority (manual > kinetic > boundary_rendezvous >
cosine): upgraded, never downgraded. Older values 'cosine_day' /
'cosine_night' rank as cosine, 'synthetic' and anything unknown below all.
"""

from __future__ import annotations

import json
import sqlite3

import pandas as pd

import config

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS session_assignments (
    session_id     TEXT    NOT NULL,
    temp_id        INTEGER NOT NULL,   -- raw temp_id (raw_tracks)
    final_temp_id  INTEGER,            -- after stitching + Duplicate Resolver; NULL on link rows
    real_id        INTEGER NOT NULL,   -- AnimalId, or negative synthetic id
    source         TEXT    NOT NULL,   -- manual | kinetic | boundary_rendezvous | cosine | legacy
    written_by     TEXT    NOT NULL,   -- session whose reconcile run wrote this row
    created_dt     TEXT    NOT NULL,
    PRIMARY KEY (session_id, temp_id, written_by)
);
CREATE INDEX IF NOT EXISTS idx_session_assignments_real ON session_assignments(real_id);
CREATE INDEX IF NOT EXISTS idx_session_assignments_writer ON session_assignments(written_by);
"""


def _table_exists(conn: sqlite3.Connection, name: str) -> bool:
    return conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                        (name,)).fetchone() is not None


def _columns(conn: sqlite3.Connection, table: str) -> set:
    return {r[1] for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def ensure(conn: sqlite3.Connection, log=print) -> None:
    """Create session_assignments if missing; on creation, backfill it once
    from reid_registry.known_temp_ids. Safe to call every time."""
    existed = _table_exists(conn, "session_assignments")
    conn.executescript(SCHEMA_SQL)
    conn.commit()
    if existed or not _table_exists(conn, "reid_registry") \
            or "known_temp_ids" not in _columns(conn, "reid_registry"):
        return

    claims: dict = {}   # (session_id, temp_id) -> set of real_ids
    for real_id, kj in conn.execute(
            "SELECT real_id, known_temp_ids FROM reid_registry "
            "WHERE known_temp_ids IS NOT NULL").fetchall():
        try:
            entries = json.loads(kj or "[]")
        except (TypeError, ValueError):
            continue
        for e in entries:
            try:
                key = (str(e["session_id"]), int(e["temp_id"]))
            except (KeyError, TypeError, ValueError):
                continue
            claims.setdefault(key, set()).add(int(real_id))

    now = pd.Timestamp.now().isoformat()
    rows = [(sid, tid, int(next(iter(ids))), "legacy", sid, now)
            for (sid, tid), ids in claims.items() if len(ids) == 1]
    conflicted = sorted(k for k, ids in claims.items() if len(ids) > 1)
    conn.executemany(
        "INSERT OR IGNORE INTO session_assignments "
        "(session_id, temp_id, final_temp_id, real_id, source, written_by, created_dt) "
        "VALUES (?, ?, NULL, ?, ?, ?, ?)", rows)
    conn.commit()
    if rows or conflicted:
        log(f"[identity] session_assignments created: {len(rows)} entries copied from "
            f"known_temp_ids, {len(conflicted)} skipped (two ids on record)")
    if conflicted:
        sessions = sorted({sid for sid, _ in conflicted})
        log(f"[identity]   re-run reconcile for: {', '.join(sessions)}")


# ─────────────────────────────────────────────────────────────────────────────
# Write (reconcile.py only)
# ─────────────────────────────────────────────────────────────────────────────

def write_session(conn: sqlite3.Connection, session_id: str,
                  assignment: dict, provenance: dict, raw_to_final: dict,
                  links: list, dry_run: bool = False) -> int:
    """
    Replace everything this session's earlier runs wrote.

    assignment   : {final temp_id -> id} after every identity step
    provenance   : {final temp_id -> source}
    raw_to_final : {raw temp_id -> final temp_id} for renamed fragments
                   (stitching + Duplicate Resolver); unrenamed ids map to themselves
    links        : [(other_session_id, other_temp_id, id)] — other sessions'
                   tracks this run tied to a synthetic id
    Returns the number of rows written.
    """
    now = pd.Timestamp.now().isoformat()
    rows = []
    for final, rid in assignment.items():
        rows.append((session_id, int(final), int(final), int(rid),
                     provenance.get(final, "cosine"), session_id, now))
    for raw, final in raw_to_final.items():
        if final in assignment and raw not in assignment:
            rows.append((session_id, int(raw), int(final), int(assignment[final]),
                         provenance.get(final, "cosine"), session_id, now))
    for other_sid, other_tid, rid in links:
        if other_sid != session_id:
            rows.append((other_sid, int(other_tid), None, int(rid), "cosine", session_id, now))
    if dry_run:
        return len(rows)
    conn.execute("DELETE FROM session_assignments WHERE written_by = ?", (session_id,))
    conn.executemany(
        "INSERT OR REPLACE INTO session_assignments "
        "(session_id, temp_id, final_temp_id, real_id, source, written_by, created_dt) "
        "VALUES (?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()
    return len(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Read
# ─────────────────────────────────────────────────────────────────────────────

_ORDER = ("ORDER BY (written_by = session_id) DESC, created_dt DESC")


def lookup(conn: sqlite3.Connection, session_id: str, temp_id: int):
    """The id on record for one (session, temp_id) — the session's own row
    first, else the newest link from another session. None if none."""
    row = conn.execute(
        "SELECT real_id FROM session_assignments WHERE session_id = ? AND temp_id = ? "
        + _ORDER + " LIMIT 1", (session_id, int(temp_id))).fetchone()
    return None if row is None else row[0]


def session_map(conn: sqlite3.Connection, session_id: str,
                sources: tuple | None = None, own_only: bool = False) -> dict:
    """{temp_id -> (id, source)} for every temp_id of a session, same
    preference as lookup(). sources: only rows with these sources.
    own_only: ignore link rows written by other sessions."""
    q = "SELECT temp_id, real_id, source FROM session_assignments WHERE session_id = ?"
    params: list = [session_id]
    if own_only:
        q += " AND written_by = session_id"
    if sources:
        q += f" AND source IN ({','.join('?' * len(sources))})"
        params += list(sources)
    out: dict = {}
    for tid, rid, src in conn.execute(q + " " + _ORDER, params).fetchall():
        out.setdefault(int(tid), (int(rid), src))
    return out


# ─────────────────────────────────────────────────────────────────────────────
# match_method — best source ever seen
# ─────────────────────────────────────────────────────────────────────────────

_LEGACY_METHODS = {"cosine_day": "cosine", "cosine_night": "cosine"}


def method_rank(method: str | None) -> int:
    """Lower is better; unknown / 'synthetic' / None rank last."""
    pri = config.params().identity.source_priority
    m = _LEGACY_METHODS.get(method, method)
    return pri.index(m) if m in pri else len(pri)


def upgrade_match_methods(conn: sqlite3.Connection, best_source: dict,
                          dry_run: bool = False) -> list:
    """best_source: {real_id -> best source this run}. Raises each cow's
    match_method to that source if it ranks higher than what is stored;
    creates the reid_registry row if the cow has none yet. Synthetic ids
    are ignored. Returns [(real_id, old, new)] for the rows that changed."""
    changed = []
    now = pd.Timestamp.now().isoformat()
    for rid, src in best_source.items():
        if rid < 0:
            continue
        row = conn.execute("SELECT match_method FROM reid_registry WHERE real_id = ?",
                           (rid,)).fetchone()
        if row is None:
            changed.append((rid, None, src))
            if not dry_run:
                conn.execute("INSERT INTO reid_registry (real_id, match_method, first_seen_dt) "
                             "VALUES (?, ?, ?)", (rid, src, now))
        elif method_rank(src) < method_rank(row[0]):
            changed.append((rid, row[0], src))
            if not dry_run:
                conn.execute("UPDATE reid_registry SET match_method = ? WHERE real_id = ?",
                             (src, rid))
    if not dry_run:
        conn.commit()
    return changed