#!/usr/bin/env python3
"""
pipeline_check.py — debugging + calibration for the Stage 2 identity pipeline.

Read-only: never writes the DB, galleries or params. Every report is short
plain text, meant to be pasted into a chat or a notes file.

DEBUGGING (does the code do what it should, on real data?)
  summary     what reconcile did per session: tracks, stitches, merges, ids by
              source, links, flags, held gallery contributions, timeline fill
  selfcheck   invariants across every feature: galleries == ledger rebuild,
              reid_registry mirror, identity record consistent with raw_tracks
              / manual labels / timeline / stitches / merges, match_method never
              below its evidence, synthetic ids not dangling, flags and
              embed_mean well-formed. Prints PASS / WARN / FAIL with examples.
  snapshot    write the pipeline's whole state (DB tables + galleries, minus
              timestamps) to a JSON file
  diff        compare two snapshots. Use it for:
                re-run idempotency  snapshot A → reconcile S → snapshot B → diff A B
                                    (A from an earlier run of S: no differences expected)
                dry-run safety      snapshot → reconcile --dry_run → snapshot → diff (empty)
                param changes       see exactly what a new value changed

CALIBRATION (which threshold values are right for this data?)
  Ground truth = labeled tracks. --truth manual (default): your
  manual_assignments (assign_identity.py). --truth kinetic: kinetic ids from
  reconcile runs. --truth all: manual first, kinetic where unlabeled;
  disagreements are reported. Labels sit on RAW temp_ids; a stitched chain
  takes the label of its labeled fragments (a chain holding two different
  labels is a wrong stitch link — reported).

  labels      the ground-truth set: per session and cow, tracks and frames;
              conflicts; manual vs kinetic agreement
  quality     (no labels) appearance_quality.min_px / full_px / edge_weight:
              per box-size bin, how often a frame is closer to its own track
              than to any other track
  trim        (no labels) appearance_quality.robust_mean: how much tracks get
              trimmed; tracks hitting the cap
  cosine      cosine_resolver.threshold, gallery.cosine_gate margins and
              gallery.contribution.drift_gate_cosine: labeled tracks scored
              against per-cow references built from OTHER tracks (other
              sessions = the real cross-session case; any other track = more data)
  idswitch    appearance_quality.id_switch.*: false flags on labeled (clean)
              tracks, detection of artificial switches made by splicing two
              different cows' tracks, and false flags on same-cow splices
  window      timeline.embed_mean_min_frames: how many frames a window mean
              needs before it reliably looks like its own cow
  kinetic     (needs --kinetics CSV) kinetic_match.corr_threshold /
              min_active_bins scored against the manual labels — run it once
              the kinetics data arrives; works without running reconcile
  all         summary + selfcheck + labels + quality + trim + cosine +
              idswitch + window (+ kinetic with --kinetics)

REVIEW (things only a person watching the video can judge)
  review --export FILE   CSV queue: ID-switch flags (at switch_frame), held
                         gallery contributions, cosine / boundary / kinetic /
                         synthetic ids, merges, a sample of stitch links, and
                         chains with conflicting labels. Each row has the
                         display_tracks.py command that opens the video there
                         and a blank verdict column (y / n / ?).
  review --ingest FILE   read the verdicts back: precision per category and
                         the rows marked n

Data: by default the canonical DB is pulled from Drive (drive_manager, same
flag checks as the other scripts) and embeddings / galleries are read from
the local data root. --db / --outputs / --gallery_dir use local copies
instead (no Drive access at all).

Usage:
    python3 pipeline_check.py all [--session S ...] [--truth manual] [--out report.txt]
    python3 pipeline_check.py snapshot --out before.json
    python3 pipeline_check.py diff before.json after.json
    python3 pipeline_check.py kinetic --kinetics kinetic_data_X.csv [--session S]
    python3 pipeline_check.py review --export queue.csv   |   review --ingest queue.csv
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sqlite3
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

import config
import appearance_quality as aq
import gallery_ledger
import identity_record

MODALITIES = ("day", "night")
RNG_SEED = 0


# ─────────────────────────────────────────────────────────────────────────────
# Output
# ─────────────────────────────────────────────────────────────────────────────

class Out:
    """Prints to stdout and, with --out, to a file as well."""
    def __init__(self, path: str | None):
        self.f = open(path, "w", encoding="utf-8") if path else None

    def __call__(self, msg: str = "") -> None:
        print(msg, flush=True)
        if self.f:
            self.f.write(msg + "\n")

    def section(self, title: str) -> None:
        self("")
        self(f"══ {title} " + "═" * max(3, 74 - len(title)))

    def close(self) -> None:
        if self.f:
            self.f.close()


def pct(x: float) -> str:
    return f"{100 * x:.0f}%" if x == x else "—"


def q3(a) -> str:
    """p10 / p50 / p90 of a 1-D array, 3 decimals."""
    a = np.asarray(a, dtype=float)
    a = a[~np.isnan(a)]
    if len(a) == 0:
        return "—"
    p = np.percentile(a, [10, 50, 90])
    return f"{p[0]:.3f} / {p[1]:.3f} / {p[2]:.3f}"


# ─────────────────────────────────────────────────────────────────────────────
# Data access
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Ctx:
    conn: sqlite3.Connection
    outputs_dir: Path
    gallery_dir: Path
    sessions: list
    truth_mode: str
    out: Out
    _embeds: dict = field(default_factory=dict)
    _frames: dict = field(default_factory=dict)

    # ── schema helpers ────────────────────────────────────────────────────────
    def has_table(self, name: str) -> bool:
        return self.conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                                 (name,)).fetchone() is not None

    def cols(self, table: str) -> set:
        if not self.has_table(table):
            return set()
        return {r[1] for r in self.conn.execute(f"PRAGMA table_info({table})")}

    def q(self, sql: str, params=()) -> pd.DataFrame:
        return pd.read_sql_query(sql, self.conn, params=params)

    def session_info(self, sid: str) -> dict:
        if not self.has_table("video_sessions"):
            return {}
        df = self.q("SELECT * FROM video_sessions WHERE session_id = ?", (sid,))
        return df.iloc[0].to_dict() if len(df) else {}

    # ── per-session frames (boxes, time, camera mode) ─────────────────────────
    def frames(self, sid: str) -> pd.DataFrame:
        """raw_tracks rows for one session: frame_index, temp_id, frame_datetime,
        x1..y2, cx, cy, is_night (bool; NULL → clock rule, as reconcile does
        for legacy rows)."""
        if sid in self._frames:
            return self._frames[sid]
        has_night = "is_night" in self.cols("raw_tracks")
        df = self.q("SELECT frame_index, temp_id, frame_datetime, x1, y1, x2, y2, cx, cy"
                    + (", is_night" if has_night else "")
                    + " FROM raw_tracks WHERE session_id = ?", (sid,))
        df["frame_datetime"] = pd.to_datetime(df["frame_datetime"], format="mixed")
        if not has_night:
            df["is_night"] = np.nan
        if df["is_night"].isna().any():
            sp = config.params().session
            h = df["frame_datetime"].dt.hour
            clock = (h >= sp.night_start_hour) | (h < sp.night_end_hour)
            df["is_night"] = df["is_night"].fillna(clock.astype(float))
        df["is_night"] = df["is_night"].astype(bool)
        self._frames[sid] = df
        return df

    def frame_size(self, sid: str):
        info = self.session_info(sid)
        w, h = info.get("frame_w"), info.get("frame_h")
        return (w, h) if w and h and not (pd.isna(w) or pd.isna(h)) else (None, None)

    # ── per-session embeddings ────────────────────────────────────────────────
    def embeds(self, sid: str) -> pd.DataFrame:
        """Embedding rows joined to their raw_tracks row: frame_index, temp_id
        (raw), embed, is_night, frame_datetime, box, q (appearance quality).
        Weight-0 frames are dropped, as reconcile does. Empty if no parquet."""
        if sid in self._embeds:
            return self._embeds[sid]
        d = self.outputs_dir / sid
        parts = sorted(d.glob("embeds_part*.parquet")) if d.exists() else []
        if not parts:   # flat layout: parts directly in --outputs
            parts = sorted(self.outputs_dir.glob("embeds_part*.parquet"))
        frames = []
        for p in parts:
            e = pd.read_parquet(p)
            if "session_id" in e.columns:
                e = e[e["session_id"] == sid]
            if len(e):
                frames.append(e[["frame_index", "temp_id", "embed"]])
        if not frames:
            self._embeds[sid] = pd.DataFrame()
            return self._embeds[sid]
        e = pd.concat(frames, ignore_index=True)
        f = self.frames(sid).drop_duplicates(["frame_index", "temp_id"])
        e = e.merge(f, on=["frame_index", "temp_id"], how="inner")
        fw, fh = self.frame_size(sid)
        e[aq.QUALITY_COL] = aq.frame_weights(e, fw, fh)
        e["embed"] = [np.asarray(v, dtype=np.float32) for v in e["embed"]]
        self._embeds[("all", sid)] = e
        e = e[e[aq.QUALITY_COL] > 0].reset_index(drop=True)
        self._embeds[sid] = e
        return e

    def embeds_unfiltered(self, sid: str) -> pd.DataFrame:
        """Like embeds() but keeps weight-0 frames (box below min_px)."""
        self.embeds(sid)
        return self._embeds.get(("all", sid), pd.DataFrame())

    # ── stitch chains ─────────────────────────────────────────────────────────
    def chain_map(self, sid: str) -> dict:
        """{raw temp_id → chain root} from tracklet_stitches (empty if the
        session hasn't been reconciled)."""
        if not self.has_table("tracklet_stitches"):
            return {}
        rows = self.conn.execute("SELECT loser_tid, root_tid FROM tracklet_stitches "
                                 "WHERE session_id = ?", (sid,)).fetchall()
        return {int(a): int(b) for a, b in rows}


def all_sessions(conn: sqlite3.Connection) -> list:
    return [r[0] for r in conn.execute(
        "SELECT DISTINCT session_id FROM raw_tracks ORDER BY session_id").fetchall()]


# ─────────────────────────────────────────────────────────────────────────────
# Ground truth
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Truth:
    raw: dict            # {(sid, raw tid) -> real_id}
    chain: dict          # {(sid, root) -> real_id}, consistent chains only
    conflicts: list      # [(sid, root, {real_id: [raw tids]})] — a chain with 2+ labels
    disagreements: list  # [(sid, raw tid, manual id, kinetic id)]  (--truth all)
    source: dict         # {(sid, raw tid) -> 'manual' | 'kinetic'}


def load_truth(ctx: Ctx) -> Truth:
    manual, kinetic = {}, {}
    if ctx.has_table("manual_assignments"):
        for sid, tid, rid in ctx.conn.execute(
                "SELECT session_id, temp_id, real_id FROM manual_assignments").fetchall():
            if sid in ctx.sessions:
                manual[(sid, int(tid))] = int(rid)
    if ctx.has_table("session_assignments"):
        for sid, tid, rid in ctx.conn.execute(
                "SELECT session_id, temp_id, real_id FROM session_assignments "
                "WHERE source = 'kinetic' AND written_by = session_id").fetchall():
            if sid in ctx.sessions:
                kinetic[(sid, int(tid))] = int(rid)

    disagreements = [(k[0], k[1], manual[k], kinetic[k]) for k in manual.keys() & kinetic.keys()
                     if manual[k] != kinetic[k]]
    if ctx.truth_mode == "manual":
        raw, source = dict(manual), {k: "manual" for k in manual}
    elif ctx.truth_mode == "kinetic":
        raw, source = dict(kinetic), {k: "kinetic" for k in kinetic}
    else:
        raw = {**kinetic, **manual}
        source = {**{k: "kinetic" for k in kinetic}, **{k: "manual" for k in manual}}

    by_chain: dict = {}
    for (sid, tid), rid in raw.items():
        root = ctx.chain_map(sid).get(tid, tid)
        by_chain.setdefault((sid, root), {}).setdefault(rid, []).append(tid)
    chain, conflicts = {}, []
    for key, labs in by_chain.items():
        if len(labs) == 1:
            chain[key] = next(iter(labs))
        else:
            conflicts.append((key[0], key[1], labs))
    return Truth(raw, chain, conflicts, disagreements, source)


# ─────────────────────────────────────────────────────────────────────────────
# Labeled track vectors (shared by cosine / idswitch / window)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class TrackVec:
    sid: str
    root: int
    mode: str            # 'day' | 'night'
    label: int | None
    vec: np.ndarray
    n_eff: float
    rows: pd.DataFrame   # its embed rows (frame_index, embed, q, ...)


def track_vectors(ctx: Ctx, truth: Truth | None, min_eff: float,
                  labeled_only: bool = True) -> list:
    """One TrackVec per (session, stitch chain, camera mode) with ≥ min_eff
    effective frames: the robust, quality-weighted mean reconcile would use."""
    out = []
    for sid in ctx.sessions:
        e = ctx.embeds(sid)
        if e.empty:
            continue
        cmap = ctx.chain_map(sid)
        e = e.assign(root=e["temp_id"].map(lambda t: cmap.get(int(t), int(t))))
        for (root, night), g in e.groupby(["root", "is_night"]):
            label = truth.chain.get((sid, int(root))) if truth else None
            if labeled_only and label is None:
                continue
            if aq.effective_frames(g) < min_eff:
                continue
            rep = aq.mean_embed_report(g, check_split=False)
            out.append(TrackVec(sid, int(root), "night" if night else "day", label,
                                rep.vector, aq.effective_frames(g), g))
    return out


def cow_reference(tracks: list, cow: int, mode: str, exclude, cross_session: bool):
    """Weighted mean (n_eff ** frames_weight_power, as the ledger weights) of
    a cow's OTHER tracks in one mode; None if she has none."""
    pw = config.params().gallery.contribution.frames_weight_power
    acc, wsum = None, 0.0
    for t in tracks:
        if t.label != cow or t.mode != mode or t is exclude:
            continue
        if cross_session and t.sid == exclude.sid:
            continue
        w = t.n_eff ** pw
        acc = t.vec * w if acc is None else acc + t.vec * w
        wsum += w
    if acc is None:
        return None
    v = acc / wsum
    return v / max(np.linalg.norm(v), 1e-8)


# ═════════════════════════════════════════════════════════════════════════════
# DEBUGGING
# ═════════════════════════════════════════════════════════════════════════════

def cmd_summary(ctx: Ctx) -> None:
    o = ctx.out
    o.section("SUMMARY — what reconcile did, per session")
    for sid in ctx.sessions:
        info = ctx.session_info(sid)
        f = ctx.frames(sid)
        n_raw = f["temp_id"].nunique()
        o(f"{sid}  [{info.get('camera_id', '?')}]  {len(f)} detections, {n_raw} raw temp_ids, "
          f"night frames {pct(f['is_night'].mean())}")
        e = ctx.embeds(sid)
        o(f"  embeddings: {len(e)} usable rows" if len(e) else
          "  embeddings: NONE found (outputs/<session>/embeds_part*.parquet)")
        if ctx.has_table("tracklet_stitches"):
            n = ctx.conn.execute("SELECT COUNT(*), COUNT(DISTINCT root_tid) FROM tracklet_stitches "
                                 "WHERE session_id=?", (sid,)).fetchone()
            o(f"  stitching: {n[0]} fragments joined into {n[1]} chains")
        if ctx.has_table("temp_id_merges"):
            m = dict(ctx.conn.execute("SELECT action, COUNT(*) FROM temp_id_merges WHERE session_id=? "
                                      "GROUP BY action", (sid,)).fetchall())
            o(f"  duplicate resolver: {m.get('merged', 0)} merged, "
              f"{m.get('unassigned_cooccurring', 0)} unassigned (co-occurring)")
        if ctx.has_table("session_assignments"):
            sa = ctx.q("SELECT temp_id, final_temp_id, real_id, source, written_by FROM "
                       "session_assignments WHERE session_id=? OR written_by=?", (sid, sid))
            own = sa[(sa.written_by == sid) & sa.final_temp_id.notna()]
            final = own[own.temp_id == own.final_temp_id]
            by_src = final.groupby("source").size().to_dict()
            o(f"  ids: {len(final)} tracks → {final.real_id.nunique()} ids "
              f"({', '.join(f'{k} {v}' for k, v in sorted(by_src.items())) or 'none'}); "
              f"{len(own) - len(final)} more raw fragments; "
              f"{int((final.real_id < 0).sum())} synthetic")
            o(f"  links: {int(((sa.written_by == sid) & (sa.final_temp_id.isna())).sum())} to other "
              f"sessions, {int(((sa.written_by != sid)).sum())} from other sessions")
        else:
            o("  ids: no session_assignments table — reconcile hasn't run with the new code")
        if ctx.has_table("manual_assignments"):
            n = ctx.conn.execute("SELECT COUNT(*) FROM manual_assignments WHERE session_id=?",
                                 (sid,)).fetchone()[0]
            o(f"  manual labels: {n}")
        if ctx.has_table("gallery_contributions"):
            g = ctx.q("SELECT gallery, status, COUNT(*) n FROM gallery_contributions "
                      "WHERE session_id=? GROUP BY gallery, status", (sid,))
            o("  gallery contributions: " + (", ".join(
                f"{r.gallery}/{r.status} {r.n}" for r in g.itertuples()) or "none"))
        if ctx.has_table("id_switch_flags"):
            n = ctx.conn.execute("SELECT COUNT(*) FROM id_switch_flags WHERE session_id=?",
                                 (sid,)).fetchone()[0]
            o(f"  ID-switch flags: {n}")
        if ctx.has_table("resolved_cow_timeline"):
            has_em = "embed_mean" in ctx.cols("resolved_cow_timeline")
            t = ctx.q("SELECT real_id, modality_mask" + (", embed_mean" if has_em else "")
                      + " FROM resolved_cow_timeline WHERE session_id=?", (sid,))
            if len(t):
                bits = {1: "sensor", 2: "vision", 4: "reid"}
                mask = ", ".join(f"{nm} {pct(((t.modality_mask.fillna(0).astype(int) & b) > 0).mean())}"
                                 for b, nm in bits.items())
                em = f", embed_mean {pct(t.embed_mean.notna().mean())}" if has_em else ""
                o(f"  timeline: {len(t)} rows, {t.real_id.nunique()} ids; {mask}{em}")
            else:
                o("  timeline: no rows")


# ── selfcheck ────────────────────────────────────────────────────────────────

class Checks:
    def __init__(self, out: Out):
        self.out, self.n = out, {"PASS": 0, "WARN": 0, "FAIL": 0}

    def __call__(self, level: str, name: str, detail: str = "", examples=None) -> None:
        self.n[level] += 1
        self.out(f"  {level:4s}  {name}" + (f" — {detail}" if detail else ""))
        for ex in list(examples or [])[:3]:
            self.out(f"          e.g. {ex}")

    def result(self, name: str, bad: list, detail_ok: str = "", level: str = "FAIL",
               what: str = "") -> None:
        if bad:
            self(level, name, f"{len(bad)} {what or 'problem(s)'}", bad)
        else:
            self("PASS", name, detail_ok)


def _load_flat(gallery_dir: Path, m: str) -> dict:
    p = gallery_dir / f"gallery_{m}.npy"
    if not p.exists():
        return {}
    return {int(k): np.asarray(v, np.float32) for k, v in np.load(p, allow_pickle=True).item().items()}


def _load_pose(gallery_dir: Path, m: str):
    try:
        from vision_features.gallery import PoseGallery
    except Exception:
        return None
    return PoseGallery.load(str(gallery_dir), m)


def cmd_selfcheck(ctx: Ctx) -> None:
    o, c = ctx.out, Checks(ctx.out)
    P = config.params()
    o.section("SELFCHECK — invariants (PASS / WARN / FAIL)")
    if not ctx.has_table("gallery_contributions") or not ctx.has_table("session_assignments"):
        c("WARN", "tables", "gallery_contributions / session_assignments missing — run reconcile "
                            "with the new code first; checks below are partial")
    sess = set(ctx.sessions)

    # ── 1. galleries == ledger rebuild ────────────────────────────────────────
    o("[gallery ledger]")
    if ctx.has_table("gallery_contributions"):
        rows = ctx.q("SELECT * FROM gallery_contributions")
        c.result("no synthetic ids in the ledger",
                 [f"id={r.id} real_id={r.real_id}" for r in rows[rows.real_id < 0].itertuples()])
        c.result("valid status values", [f"id={r.id} {r.status}" for r in rows[
            ~rows.status.isin(["active", "held_for_review", "rejected"])].itertuples()])
        leg = rows[rows.session_id == gallery_ledger.LEGACY_SESSION].groupby(
            ["real_id", "modality", "gallery", "slot_idx"]).size()
        c.result("at most one legacy seed per key", [str(k) for k, v in leg.items() if v > 1])
        sw = P.gallery.source_weights
        stale = rows[rows.source.isin(["manual", "kinetic", "boundary_rendezvous", "cosine"])]
        stale = [f"id={r.id} {r.source} stored {r.source_weight} now {getattr(sw, r.source)}"
                 for r in stale.itertuples() if abs(r.source_weight - getattr(sw, r.source)) > 1e-9]
        c.result("stored source weights match params", stale, level="WARN",
                 what="rows written with other source_weights (params changed since — re-run "
                      "those sessions if that was intended)")
        held = rows[rows.status == "held_for_review"]
        c("WARN" if len(held) else "PASS", "held for review",
          f"{len(held)} row(s) — gallery_ledger.py list-pending" if len(held) else "none")

        for m in MODALITIES:
            flat = _load_flat(ctx.gallery_dir, m)
            keys = rows[(rows.gallery == "flat") & (rows.modality == m)].real_id.unique()
            bad, missing = [], []
            for rid in keys:
                vec, n, _ = gallery_ledger.rebuild_vector(ctx.conn, int(rid), m, "flat")
                g = flat.get(int(rid))
                if vec is None and g is not None:
                    bad.append(f"{rid}: gallery has a vector, ledger has no active rows")
                elif vec is not None and g is None:
                    missing.append(f"{rid}: ledger has {n} active rows, gallery has no vector")
                elif vec is not None and float(vec @ g) < 0.9999:
                    bad.append(f"{rid}: cos(gallery, rebuild) = {float(vec @ g):.5f}")
            c.result(f"flat {m} gallery == ledger rebuild", bad + missing,
                     f"{len(keys)} cows", what="cows differ")
            pre = sorted(set(flat) - set(int(k) for k in rows.real_id.unique()))
            if pre:
                c("WARN", f"flat {m}: vectors with no ledger rows",
                  f"{len(pre)} pre-ledger cows (seeded on their next commit)", pre)
            # reid_registry mirror
            if ctx.has_table("reid_registry") and f"gallery_embed_{m}" in ctx.cols("reid_registry"):
                bad = []
                for rid, ej in ctx.conn.execute(f"SELECT real_id, gallery_embed_{m} FROM reid_registry "
                                                f"WHERE gallery_embed_{m} IS NOT NULL").fetchall():
                    g = flat.get(int(rid))
                    v = np.asarray(json.loads(ej), np.float32)
                    if g is None or float(v @ g) < 0.9999:
                        bad.append(f"{rid}: reid_registry.gallery_embed_{m} ≠ gallery file")
                c.result(f"reid_registry mirrors flat {m} gallery", bad)
            pg = _load_pose(ctx.gallery_dir, m)
            if pg is None:
                c("WARN", f"pose {m} gallery", "vision_features not importable — skipped")
                continue
            pkeys = rows[(rows.gallery == "pose") & (rows.modality == m)][
                ["real_id", "slot_idx"]].drop_duplicates()
            bad = []
            for r in pkeys.itertuples():
                vec, n, _ = gallery_ledger.rebuild_vector(ctx.conn, int(r.real_id), m, "pose",
                                                          int(r.slot_idx))
                g = pg.get_slot_vector(int(r.real_id), int(r.slot_idx))
                if (vec is None) != (g is None) or (vec is not None and float(vec @ g) < 0.9999):
                    bad.append(f"{r.real_id}/slot{r.slot_idx}")
            c.result(f"pose {m} gallery == ledger rebuild", bad, f"{len(pkeys)} slots",
                     what="slots differ")

    # ── 2. identity record ────────────────────────────────────────────────────
    o("[identity record]")
    if ctx.has_table("session_assignments"):
        sa = ctx.q("SELECT * FROM session_assignments")
        own = sa[sa.written_by == sa.session_id]
        raw_ids = {sid: set(ctx.frames(sid).temp_id.astype(int)) for sid in sess}
        bad = [f"{r.session_id} t{r.temp_id}" for r in own.itertuples()
               if r.session_id in sess and int(r.temp_id) not in raw_ids[r.session_id]]
        c.result("every recorded temp_id exists in raw_tracks", bad)
        bad = []
        fin = own[own.final_temp_id.notna()]
        fin_id = {(r.session_id, int(r.temp_id)): int(r.real_id) for r in fin.itertuples()}
        for r in fin.itertuples():
            k = (r.session_id, int(r.final_temp_id))
            if k not in fin_id or fin_id[k] != int(r.real_id):
                bad.append(f"{r.session_id} t{r.temp_id} → final t{int(r.final_temp_id)}: "
                           f"{r.real_id} vs {fin_id.get(k)}")
        c.result("fragments carry their final track's id", bad)
        for sid in sess:
            cmap = ctx.chain_map(sid)
            if not cmap:
                continue
            f_of = {int(r.temp_id): int(r.final_temp_id) for r in fin[fin.session_id == sid].itertuples()}
            root_final = {}
            bad = []
            for loser, root in cmap.items():
                if loser in f_of and root in f_of and f_of[loser] != f_of[root]:
                    bad.append(f"t{loser} (chain root t{root}): final t{f_of[loser]} vs t{f_of[root]}")
            c.result(f"stitched fragments follow their chain ({sid[:24]}…)", bad)
        links = sa[sa.written_by != sa.session_id]
        c.result("link rows have no final_temp_id",
                 [f"{r.session_id} t{r.temp_id}" for r in links[links.final_temp_id.notna()].itertuples()])
        # synthetic ids must exist (else resolved but not rewritten)
        if ctx.has_table("reid_registry"):
            synth = {int(r[0]) for r in ctx.conn.execute(
                "SELECT real_id FROM reid_registry WHERE real_id < 0").fetchall()}
            dang = sorted({int(x) for x in sa.real_id if x < 0} - synth)
            c.result("synthetic ids in session_assignments exist in reid_registry",
                     [str(x) for x in dang], what="dangling synthetic ids")
            used = {int(x) for x in sa.real_id if x < 0}
            orphan = sorted(synth - used)
            if orphan:
                c("WARN", "synthetic ids nobody uses", f"{len(orphan)} reid_registry rows",
                  [str(x) for x in orphan])
            # match_method never below its evidence
            mm = dict(ctx.conn.execute("SELECT real_id, match_method FROM reid_registry").fetchall())
            bad = []
            ev = own[(own.real_id > 0) & (own.source != "legacy")]
            for rid, g in ev.groupby("real_id"):
                best = min(g.source, key=identity_record.method_rank)
                if int(rid) not in mm:
                    bad.append(f"{rid}: no reid_registry row")
                elif identity_record.method_rank(mm[int(rid)]) > identity_record.method_rank(best):
                    bad.append(f"{rid}: match_method {mm[int(rid)]} below evidence '{best}'")
            c.result("match_method ≥ best recorded source", bad)
        # manual labels honoured
        if ctx.has_table("manual_assignments"):
            ma = ctx.q("SELECT session_id, temp_id, real_id, assigned_dt FROM manual_assignments")
            bad, pending = [], []
            last_run = own.groupby("session_id").created_dt.max().to_dict()
            for r in ma[ma.session_id.isin(sess)].itertuples():
                got = fin_id.get((r.session_id, int(r.temp_id)))
                if got == int(r.real_id):
                    continue
                run = last_run.get(r.session_id)
                if run is None or (r.assigned_dt and str(r.assigned_dt) > str(run)):
                    pending.append(f"{r.session_id} t{r.temp_id}")
                else:
                    bad.append(f"{r.session_id} t{r.temp_id}: manual {r.real_id}, recorded {got}")
            c.result("manual assignments are what reconcile recorded", bad)
            if pending:
                c("WARN", "manual assignments newer than the last reconcile run",
                  f"{len(pending)} — re-run those sessions", pending)
        # merges
        if ctx.has_table("temp_id_merges"):
            mg = ctx.q("SELECT * FROM temp_id_merges")
            bad = []
            for r in mg[mg.session_id.isin(sess)].itertuples():
                lo = fin_id.get((r.session_id, int(r.loser_tid)))
                if r.action == "merged" and lo is not None and lo != int(r.real_id):
                    bad.append(f"{r.session_id} loser t{r.loser_tid}: {lo} ≠ merged id {r.real_id}")
                if r.action == "unassigned_cooccurring" and lo == int(r.real_id):
                    bad.append(f"{r.session_id} loser t{r.loser_tid} still holds {r.real_id}")
            c.result("duplicate-resolver outcomes match the identity record", bad)

    # ── 3. flags ──────────────────────────────────────────────────────────────
    o("[ID-switch flags]")
    if ctx.has_table("id_switch_flags"):
        fl = ctx.q("SELECT * FROM id_switch_flags")
        bad = []
        for r in fl[fl.session_id.isin(sess)].itertuples():
            f = ctx.frames(r.session_id)
            cmap = ctx.chain_map(r.session_id)
            members = [t for t, root in cmap.items() if root == r.temp_id] + [int(r.temp_id)]
            fr = f[f.temp_id.isin(members)].frame_index
            if fr.empty:
                bad.append(f"{r.session_id} t{r.temp_id}: no such track")
            elif "switch_frame" in fl.columns and pd.notna(getattr(r, "switch_frame", None)) \
                    and not fr.min() <= r.switch_frame <= fr.max():
                bad.append(f"{r.session_id} t{r.temp_id}: switch_frame {r.switch_frame} outside "
                           f"{fr.min()}–{fr.max()}")
        c.result("flags point at real tracks and frames", bad, f"{len(fl)} flags")

    # ── 4. timeline ───────────────────────────────────────────────────────────
    o("[timeline]")
    if ctx.has_table("resolved_cow_timeline"):
        cols = ctx.cols("resolved_cow_timeline")
        tl = ctx.q("SELECT * FROM resolved_cow_timeline")
        tl = tl[tl.session_id.isin(sess)]
        dup = tl.groupby(["session_id", "real_id", "window_start_dt"]).size()
        c.result("one row per (session, cow, window)", [str(k) for k, v in dup.items() if v > 1])
        bm = P.timeline.bin_minutes
        w = pd.to_datetime(tl.window_start_dt, format="mixed")
        c.result(f"windows aligned to {bm} min",
                 [str(x) for x in tl.window_start_dt[(w != w.dt.floor(f"{bm}min"))].head(3)])
        if ctx.has_table("session_assignments"):
            ids = {(r.session_id, int(r.real_id)) for r in own.itertuples()}
            bad = sorted({f"{r.session_id} {r.real_id}" for r in tl.itertuples()
                          if (r.session_id, int(r.real_id)) not in ids})
            c.result("timeline ids are in the session's identity record", bad,
                     level="WARN" if not ctx.has_table("session_assignments") else "FAIL")
        if "embed_mean" in cols:
            bad = []
            for r in tl[tl.embed_mean.notna()].itertuples():
                v = np.asarray(json.loads(r.embed_mean), float)
                if len(v) != 128 or abs(np.linalg.norm(v) - 1) > 1e-3:
                    bad.append(f"{r.session_id} {r.real_id} {r.window_start_dt}: dim {len(v)}, "
                               f"norm {np.linalg.norm(v):.3f}")
            if "embed_mean_is_night" in cols:
                mism = tl[tl.embed_mean.notna() != tl.embed_mean_is_night.notna()]
                bad += [f"{r.session_id} {r.real_id} {r.window_start_dt}: embed_mean / is_night "
                        f"set apart" for r in mism.itertuples()]
            c.result("embed_mean well-formed", bad, f"{int(tl.embed_mean.notna().sum())} filled")
        if ctx.has_table("reid_registry"):
            synth = {int(r[0]) for r in ctx.conn.execute(
                "SELECT real_id FROM reid_registry WHERE real_id < 0").fetchall()}
            c.result("no timeline rows on resolved (deleted) synthetic ids",
                     sorted({f"{r.session_id} {r.real_id}" for r in tl.itertuples()
                             if r.real_id < 0 and int(r.real_id) not in synth}))

    o(f"  ── {c.n['PASS']} pass, {c.n['WARN']} warn, {c.n['FAIL']} fail")


# ── snapshot / diff ──────────────────────────────────────────────────────────

SNAP_SKIP_TABLES = {"raw_tracks", "sqlite_sequence"}
SNAP_SKIP_COLS = {"id", "created_dt", "reviewed_dt", "merged_dt", "assigned_dt", "first_seen_dt"}
VECTOR_COLS = {"vector", "embed_mean", "gallery_embed_day", "gallery_embed_night", "known_temp_ids"}


def _canon(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return None
    if isinstance(v, float):
        return round(v, 5)
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating,)):
        return round(float(v), 5)
    return v


def _vec_digest(s):
    if s is None:
        return None
    try:
        v = np.round(np.asarray(json.loads(s), float), 4)
        return "vec:" + hashlib.sha1(v.tobytes()).hexdigest()[:12]
    except Exception:
        return "str:" + hashlib.sha1(str(s).encode()).hexdigest()[:12]


def cmd_snapshot(ctx: Ctx, path: str) -> None:
    snap = {"tables": {}, "galleries": {}}
    for (t,) in ctx.conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall():
        if t in SNAP_SKIP_TABLES:
            continue
        df = ctx.q(f"SELECT * FROM [{t}]")
        keep = [c for c in df.columns if c not in SNAP_SKIP_COLS and not c.startswith("last_updated")]
        rows = []
        for rec in df[keep].itertuples(index=False):
            rows.append([_vec_digest(v) if c in VECTOR_COLS else _canon(v)
                         for c, v in zip(keep, rec)])
        snap["tables"][t] = {"columns": keep, "rows": rows}
    for m in MODALITIES:
        snap["galleries"][f"flat_{m}"] = {str(k): np.round(v, 5).tolist()
                                          for k, v in _load_flat(ctx.gallery_dir, m).items()}
        pg = _load_pose(ctx.gallery_dir, m)
        if pg is not None:
            snap["galleries"][f"pose_{m}"] = {
                f"{k}/{s}": np.round(pg.embeds[k][s], 5).tolist()
                for k in pg.embeds for s in range(pg.embeds[k].shape[0])
                if not np.all(np.isnan(pg.embeds[k][s]))}
    Path(path).write_text(json.dumps(snap))
    n = sum(len(v["rows"]) for v in snap["tables"].values())
    ctx.out(f"snapshot: {len(snap['tables'])} tables, {n} rows, "
            f"{sum(len(g) for g in snap['galleries'].values())} gallery vectors → {path}")


def cmd_diff(out: Out, a_path: str, b_path: str) -> None:
    from collections import Counter
    A, B = json.loads(Path(a_path).read_text()), json.loads(Path(b_path).read_text())
    out.section(f"DIFF {Path(a_path).name} → {Path(b_path).name}")
    same = True
    for t in sorted(set(A["tables"]) | set(B["tables"])):
        ta, tb = A["tables"].get(t), B["tables"].get(t)
        if ta is None or tb is None:
            out(f"  {t}: table {'added' if ta is None else 'removed'}")
            same = False
            continue
        if ta["columns"] != tb["columns"]:
            out(f"  {t}: columns changed {ta['columns']} → {tb['columns']}")
        ca = Counter(json.dumps(r) for r in ta["rows"])
        cb = Counter(json.dumps(r) for r in tb["rows"])
        rem, add = ca - cb, cb - ca
        if rem or add:
            same = False
            out(f"  {t}: -{sum(rem.values())} +{sum(add.values())} rows  (cols {tb['columns']})")
            for r in list(rem)[:3]:
                out(f"      - {r}")
            for r in list(add)[:3]:
                out(f"      + {r}")
    for g in sorted(set(A["galleries"]) | set(B["galleries"])):
        ga, gb = A["galleries"].get(g, {}), B["galleries"].get(g, {})
        changed = [k for k in ga.keys() & gb.keys()
                   if float(np.dot(ga[k], gb[k])) < 0.99999]
        added, removed = sorted(gb.keys() - ga.keys()), sorted(ga.keys() - gb.keys())
        if changed or added or removed:
            same = False
            out(f"  gallery {g}: {len(changed)} changed {changed[:5]}, {len(added)} added "
                f"{added[:5]}, {len(removed)} removed {removed[:5]}")
    out("  identical (ignoring timestamps and row ids)" if same else "")


# ═════════════════════════════════════════════════════════════════════════════
# CALIBRATION
# ═════════════════════════════════════════════════════════════════════════════

def cmd_labels(ctx: Ctx, truth: Truth) -> None:
    o = ctx.out
    o.section(f"LABELS — ground truth (--truth {ctx.truth_mode})")
    if not truth.raw:
        o("  no labeled tracks. Label the long tracks of each session in display_tracks.py and")
        o("  record them with: assign_identity.py --session S --assign <temp_id>:<AnimalId> ...")
        o("  (no reconcile run needed for calibration; labels on one fragment cover its chain")
        o("  once the session has been reconciled).")
        return
    for sid in ctx.sessions:
        f = ctx.frames(sid)
        e = ctx.embeds(sid)
        cmap = ctx.chain_map(sid)
        labs = {root: rid for (s_, root), rid in truth.chain.items() if s_ == sid}
        if not labs:
            o(f"  {sid}: no labels")
            continue
        root_of = f.temp_id.map(lambda t: cmap.get(int(t), int(t)))
        det = f.assign(root=root_of, label=root_of.map(labs)).dropna(subset=["label"])
        tot_frames = f.frame_index.nunique()
        o(f"  {sid}: {len(labs)} labeled chains, {det.label.nunique()} cows, "
          f"{pct(len(det) / max(len(f), 1))} of detections labeled")
        for rid, g in det.groupby("label"):
            em = 0.0
            if len(e):
                ee = e[e.temp_id.map(lambda t: labs.get(cmap.get(int(t), int(t)))) == rid]
                em = aq.effective_frames(ee)
            o(f"      cow {int(rid)}: {g.root.nunique()} chain(s), {len(g)} detections "
              f"({pct(g.frame_index.nunique() / tot_frames)} of frames), {em:.0f} effective "
              f"embedding frames")
    if truth.conflicts:
        o(f"  CONFLICTS — chains whose fragments carry different labels "
          f"(wrong stitch link or wrong label): {len(truth.conflicts)}")
        for sid, root, labs in truth.conflicts[:10]:
            o(f"      {sid} chain t{root}: " + ", ".join(f"{k}←t{v}" for k, v in labs.items()))
    if truth.disagreements:
        o(f"  manual vs kinetic disagree on {len(truth.disagreements)} temp_id(s) (manual wins):")
        for d in truth.disagreements[:10]:
            o(f"      {d[0]} t{d[1]}: manual {d[2]}, kinetic {d[3]}")


# ── quality ──────────────────────────────────────────────────────────────────

SIZE_EDGES = [0, 16, 24, 32, 40, 48, 64, 80, 96, 128, 192, 10_000]


def cmd_quality(ctx: Ctx, max_frames: int = 40_000) -> None:
    """For each frame: is it closer to its own track than to every track on
    screen at the same time (= certainly a different cow)? The own-track
    reference is built from the OTHER half of the track in time, so a frame
    is never compared with itself or its near-identical neighbours."""
    o, P = ctx.out, config.params()
    o.section("QUALITY — appearance_quality.min_px / full_px / edge_weight (no labels)")
    aqp = P.appearance_quality
    rng = np.random.default_rng(RNG_SEED)
    by_cam: dict = {}
    for sid in ctx.sessions:
        e = ctx.embeds_unfiltered(sid)
        if e.empty:
            continue
        cam = ctx.session_info(sid).get("camera_id", sid)
        cmap = ctx.chain_map(sid)
        fw, fh = ctx.frame_size(sid)
        e = e.assign(root=e.temp_id.map(lambda t: cmap.get(int(t), int(t))),
                     size=np.sqrt(np.clip((e.x2 - e.x1) * (e.y2 - e.y1), 0, None)))
        if fw and fh:
            m = aqp.edge_margin_px
            e["edge"] = (e.x1 <= m) | (e.y1 <= m) | (e.x2 >= fw - m) | (e.y2 >= fh - m)
        else:
            e["edge"] = False
        def unit(v):
            return v / max(np.linalg.norm(v), 1e-8)

        for night, em in e.groupby("is_night"):
            spans, means, halves, half_of = {}, {}, {}, {}
            for root, g in em.groupby("root"):
                if len(g) < 40:
                    continue
                g = g.sort_values("frame_index")
                mid = g.frame_index.iloc[len(g) // 2]
                h0, h1 = g[g.frame_index < mid], g[g.frame_index >= mid]
                # reference per half from its larger boxes (cleanest appearance)
                ref = [unit(np.stack(h[h["size"] >= h["size"].median()].embed.values).mean(0))
                       for h in (h0, h1)]
                halves[root] = ref
                means[root] = unit(ref[0] + ref[1])
                spans[root] = (g.frame_index.min(), g.frame_index.max())
                half_of[root] = mid
            if len(means) < 2:
                continue
            roots = list(means)
            M = np.stack([means[r] for r in roots])
            idx = {r: i for i, r in enumerate(roots)}
            sub = em[em.root.isin(roots)]
            if len(sub) > max_frames:
                sub = sub.iloc[rng.choice(len(sub), max_frames, replace=False)]
            X = np.stack(sub.embed.values)
            S = X @ M.T
            own_i = sub.root.map(idx).to_numpy()
            # own reference = the OTHER half of the frame's own track
            own_ref = np.stack([halves[r][0 if fi >= half_of[r] else 1]
                                for r, fi in zip(sub.root, sub.frame_index)])
            own = (X * own_ref).sum(1)
            # rivals: tracks on screen during this track's span (certainly other cows)
            ov = np.array([[spans[a][0] <= spans[b][1] and spans[b][0] <= spans[a][1] and a != b
                            for b in roots] for a in roots])
            rival_mask = ov[own_i]
            has_rival = rival_mask.any(1)
            best_other = np.where(rival_mask, S, -np.inf).max(1)
            rec = pd.DataFrame({"size": sub["size"].to_numpy(), "edge": sub.edge.to_numpy(),
                                "track": [f"{sid}/{r}" for r in sub.root],
                                "win": own > best_other, "margin": own - best_other})[has_rival]
            by_cam.setdefault((cam, "night" if night else "day"), []).append(rec)

    if not by_cam:
        o("  no session has embeddings + 2 co-occurring tracks — nothing to measure")
        return
    for (cam, mode), recs in sorted(by_cam.items()):
        r = pd.concat(recs)
        ne = r[~r.edge]
        o(f"  {cam} / {mode}: {len(r)} frames ({pct(r.edge.mean())} at the frame edge)")
        o("     box px     frames  tracks   own-track wins (per track)   median margin")
        ne = ne.assign(bin=pd.cut(ne["size"], SIZE_EDGES, right=False))
        stats = ne.groupby("bin", observed=True).agg(n=("win", "size"), tracks=("track", "nunique"),
                                                     margin=("margin", "median"))
        # track-balanced: every track counts once per bin, however many frames it has
        stats["win"] = ne.groupby(["bin", "track"], observed=True).win.mean().groupby(
            level=0, observed=True).mean()
        for b, row in stats.iterrows():
            mark = "  ← min_px" if b.left <= aqp.min_px < b.right else (
                   "  ← full_px" if b.left <= aqp.full_px < b.right else "")
            few = "  (few tracks)" if row.tracks < 5 else ""
            o(f"     {int(b.left):>4}–{int(min(b.right, 9999)):<5} {int(row.n):>7} {int(row.tracks):>6}"
              f"   {pct(row.win):>16}             {row.margin:+.3f}{mark}{few}")
        st = stats[(stats.n >= 30) & (stats.tracks >= 5)]
        big = st.loc[[b.left >= 96 for b in st.index]] if len(st) else st
        if big.n.sum() < 50 or len(st) < 3:
            o("     too few tracks per size bin (≥5 needed in ≥3 bins, incl. ≥96 px) — no "
              "suggestion; label/process more sessions")
        else:
            plateau = float((big.win * big.n).sum() / big.n.sum())
            # full_px: lowest bin of the unbroken run of bins (from the top) within 5 pts
            sug_full = None
            for b, row in st.iloc[::-1].iterrows():
                if row.win < plateau - 0.05:
                    break
                sug_full = int(b.left)
            # min_px: top of the highest bin clearly worse (below 60% of plateau)
            low = [b for b, row in st.iterrows() if row.win < 0.6 * plateau]
            sug_min = int(max(b.right for b in low)) if low else None
            o(f"     plateau (≥96 px) {pct(plateau)} → full_px ≈ {sug_full} (lowest size within 5 pts "
              f"of it), min_px ≈ "
              + (f"{sug_min} (below 60% of it)" if sug_min is not None else
                 "— no size is that bad here; small boxes still carry identity, so a lower min_px "
                 "(and a gentler ramp) would keep more frames"))
        ed = r[r.edge & (r["size"] >= aqp.full_px)]
        ref = r[~r.edge & (r["size"] >= aqp.full_px)]
        if len(ed) >= 30 and len(ref) >= 30:
            o(f"     edge frames ≥ full_px: wins {pct(ed.win.mean())} vs {pct(ref.win.mean())} "
              f"away from the edge → edge_weight ≈ {ed.win.mean() / max(ref.win.mean(), 1e-6):.2f} "
              f"(now {aqp.edge_weight})")


# ── trim ─────────────────────────────────────────────────────────────────────

def cmd_trim(ctx: Ctx, truth: Truth) -> None:
    o, rm = ctx.out, config.params().appearance_quality.robust_mean
    o.section("TRIM — appearance_quality.robust_mean (no labels needed)")
    tracks = track_vectors(ctx, truth, rm.min_frames_for_trim, labeled_only=False)
    if not tracks:
        o("  no track with enough frames")
        return
    rec = []
    for t in tracks:
        stack = np.stack(t.rows.embed.values)
        w = t.rows[aq.QUALITY_COL].to_numpy(float)
        keep = aq._trim_outliers(stack, w)
        frac = float(w[~keep].sum() / w.sum())
        delta = None
        if t.label is not None:
            ref = cow_reference(tracks, t.label, t.mode, t, cross_session=False)
            if ref is not None:
                delta = float(aq._wmean(stack[keep], w[keep]) @ ref - aq._wmean(stack, w) @ ref)
        rec.append((t, frac, delta))
    fr = np.array([r[1] for r in rec])
    o(f"  {len(rec)} track×mode means (≥{rm.min_frames_for_trim} effective frames): "
      f"{pct((fr > 0).mean())} trimmed at all; trimmed share p10/p50/p90 {q3(fr)}")
    capped = [r for r in rec if r[1] >= rm.max_trim_frac - 0.01]
    o(f"  hit the {pct(rm.max_trim_frac)} cap: {len(capped)} — many frames disagree with the "
      f"rest: a likely ID switch or a bad track (see idswitch / review)")
    for t, f_, _ in sorted(capped, key=lambda r: -r[1])[:5]:
        o(f"      {t.sid} t{t.root} {t.mode}: {pct(f_)} trimmed")
    d = np.array([r[2] for r in rec if r[2] is not None])
    if len(d):
        o(f"  labeled tracks: trimming changes cos to the cow's reference by p10/p50/p90 {q3(d)} "
          f"(positive = trimming helps); worse in {pct((d < -0.005).mean())}")
        o(f"  → if worse often: raise trim_mad_k (now {rm.trim_mad_k}); if the cap is hit often on "
          f"clean labeled tracks: lower max_trim_frac")


# ── cosine (+ gate + drift) ─────────────────────────────────────────────────

THR_GRID = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]


def _score_queries(tracks: list, cross_session: bool) -> pd.DataFrame:
    rows = []
    cows = sorted({t.label for t in tracks})
    for t in tracks:
        refs = {c: cow_reference(tracks, c, t.mode, t, cross_session) for c in cows}
        refs = {c: v for c, v in refs.items() if v is not None}
        if t.label not in refs or len(refs) < 2:
            continue
        sc = {c: float(t.vec @ v) for c, v in refs.items()}
        wrong = {c: v for c, v in sc.items() if c != t.label}
        bw = max(wrong, key=wrong.get)
        best = max(sc, key=sc.get)
        second = sorted(sc.values())[-2]
        rows.append({"sid": t.sid, "root": t.root, "mode": t.mode, "cow": t.label,
                     "right": sc[t.label], "best_wrong": wrong[bw], "best_wrong_id": bw,
                     "best": sc[best], "best_id": best, "second": second,
                     "all_wrong": list(wrong.values()), "n_eff": t.n_eff})
    return pd.DataFrame(rows)


def cmd_cosine(ctx: Ctx, truth: Truth, max_error: float) -> None:
    o, P = ctx.out, config.params()
    cr, cg, ct = P.cosine_resolver, P.gallery.cosine_gate, P.gallery.contribution
    o.section("COSINE — cosine_resolver.threshold, gallery.cosine_gate, drift_gate_cosine")
    tracks = track_vectors(ctx, truth, cr.min_embeds)
    cows = {t.label for t in tracks}
    o(f"  {len(tracks)} labeled track×mode means, {len(cows)} cows")
    if len(cows) < 2:
        o("  need ≥2 labeled cows (a wrong cow to compare against)")
        return
    for scope, cross in (("cross-session (other sessions only — the real case)", True),
                         ("any other track (more data, easier)", False)):
        q = _score_queries(tracks, cross)
        o(f"  ── {scope}: {len(q)} queries")
        if q.empty:
            o("     none — needs the same cow labeled in ≥2 sessions" if cross else "     none")
            continue
        o(f"     cos to right cow  p10/p50/p90  {q3(q.right)}")
        o(f"     best wrong cow                 {q3(q.best_wrong)}")
        o(f"     right − best wrong             {q3(q.right - q.best_wrong)}  "
          f"(<0 in {pct((q.right < q.best_wrong).mean())})")
        o("     threshold   accepted   wrong among accepted   right ids found")
        best_thr = None
        for thr in sorted(set(THR_GRID + [cr.threshold])):
            acc = q[q.best >= thr]
            wrong = int((acc.best_id != acc.cow).sum())
            err = wrong / max(len(acc), 1)
            found = int((acc.best_id == acc.cow).sum())
            if best_thr is None and len(acc) and err <= max_error:
                best_thr = thr
            o(f"     {thr:.2f}{'*' if abs(thr - cr.threshold) < 1e-9 else ' '}      "
              f"{pct(len(acc) / len(q)):>7}   {wrong:>4} ({pct(err):>4})            "
              f"{pct(found / len(q))}")
        o(f"     → lowest threshold with ≤{pct(max_error)} wrong: {best_thr}   (* = current)")

        acc = q[q.best >= cr.threshold]
        good, bad = acc[acc.best_id == acc.cow], acc[acc.best_id != acc.cow]
        if len(acc):
            o(f"     GATE at threshold {cr.threshold}: {len(good)} right / {len(bad)} wrong accepted "
              f"ids — share that would still update the gallery (right kept | wrong let through)")
            sms = [0.0, 0.05, 0.10, 0.15, 0.20]
            o("       thr+margin \\ 2nd-best margin  " + "  ".join(f"{s:>11.2f}" for s in sms))
            for tm in [0.0, 0.025, 0.05, 0.10]:
                cells = []
                for sm in sms:
                    def passes(d):
                        return ((d.best >= cr.threshold + tm) & (d.best - d.second >= sm))
                    cells.append(f"{pct(passes(good).mean() if len(good) else np.nan):>4} | "
                                 f"{int(passes(bad).sum()) if len(bad) else 0:<3}")
                cur = "*" if abs(tm - cg.threshold_margin) < 1e-9 else " "
                o(f"       {tm:<5.3f}{cur}                       " + "  ".join(f"{c:>11}" for c in cells))
            o(f"       (* row = current threshold_margin; current second_best_margin "
              f"{cg.second_best_margin}) — pick the cell keeping most right ids with 0 wrong")

        wrong_all = np.concatenate([np.asarray(x) for x in q.all_wrong]) if len(q) else np.array([])
        o(f"     DRIFT gate — session mean vs. the cow's vector: right cow p1/p5 "
          f"{np.percentile(q.right, 1):.3f} / {np.percentile(q.right, 5):.3f}; any wrong cow "
          f"p95/p99 {np.percentile(wrong_all, 95):.3f} / {np.percentile(wrong_all, 99):.3f}")
        o("       gate    right held (false alarm)   wrong held (caught)")
        for g in sorted(set([0.3, 0.4, 0.5, 0.6, 0.7, 0.8] + [ct.drift_gate_cosine])):
            o(f"       {g:.2f}{'*' if abs(g - ct.drift_gate_cosine) < 1e-9 else ' '}   "
              f"{pct((q.right < g).mean()):>10}               {pct((wrong_all < g).mean()):>6}")


# ── idswitch ─────────────────────────────────────────────────────────────────

def _split_stats(rows: pd.DataFrame):
    """(gap, smaller-group frac, interleave frac, eff) as mean_embed_report
    computes them, or None if the frames don't form two groups."""
    stack = np.stack(rows.embed.values)
    w = rows[aq.QUALITY_COL].to_numpy(float)
    keep = aq._trim_outliers(stack, w)
    stack, w, fr = stack[keep], w[keep], rows.frame_index.to_numpy()[keep]
    if len(stack) < 4:
        return None
    g = aq._two_groups(stack, w)
    if g is None:
        return None
    labels, gap, frac = g
    inter, _ = aq._best_switch_point(labels, w, fr)
    return gap, frac, inter, float(w.sum())


def cmd_idswitch(ctx: Ctx, truth: Truth, max_pairs: int) -> None:
    o, isc = ctx.out, config.params().appearance_quality.id_switch
    o.section("IDSWITCH — appearance_quality.id_switch (min_gap, max_interleave_frac)")
    tracks = track_vectors(ctx, truth, isc.min_frames)
    if not tracks:
        o("  no labeled track with enough frames")
        return
    rng = np.random.default_rng(RNG_SEED)

    def splice(a: TrackVec, b: TrackVec):
        ra, rb = a.rows.sort_values("frame_index"), b.rows.sort_values("frame_index")
        T = min(len(ra), len(rb), 400)
        na = int(T * rng.uniform(0.3, 0.7))
        nb = T - na
        sa = rng.integers(0, len(ra) - na + 1)
        sb = rng.integers(0, len(rb) - nb + 1)
        pa, pb = ra.iloc[sa:sa + na], rb.iloc[sb:sb + nb].copy()
        pb["frame_index"] = pb.frame_index - pb.frame_index.min() + pa.frame_index.max() + 1
        return pd.concat([pa, pb], ignore_index=True)

    clean = [(t, _split_stats(t.rows)) for t in tracks]
    diff, same = [], []
    by_mode: dict = {}
    for t in tracks:
        by_mode.setdefault(t.mode, []).append(t)
    for mode, ts in by_mode.items():
        pairs_d = [(a, b) for a in ts for b in ts if a.label != b.label]
        pairs_s = [(a, b) for a in ts for b in ts if a.label == b.label and a is not b]
        for pool, dst in ((pairs_d, diff), (pairs_s, same)):
            if not pool:
                continue
            for i in rng.choice(len(pool), min(len(pool), max_pairs), replace=False):
                a, b = pool[i]
                dst.append(_split_stats(splice(a, b)))

    def flagged(st, gap, inter):
        return (st is not None and st[3] >= isc.min_frames and st[1] >= isc.min_cluster_frac
                and st[0] >= gap and st[2] <= inter)

    o(f"  samples: {len(clean)} labeled tracks (should NOT flag), {len(diff)} two-cow splices "
      f"(should flag), {len(same)} same-cow splices (should NOT flag)")
    if not diff:
        o("  need ≥2 labeled cows in the same camera mode for two-cow splices")
    gaps, inters = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30], [0.05, 0.10, 0.15, 0.20]
    o("  min_gap  max_inter   detects switch   false: clean | same-cow splice")
    for g in gaps:
        for it in inters:
            cur = "*" if abs(g - isc.min_gap) < 1e-9 and abs(it - isc.max_interleave_frac) < 1e-9 else " "
            tp = np.mean([flagged(s, g, it) for s in diff]) if diff else np.nan
            fc = np.mean([flagged(s, g, it) for _, s in clean]) if clean else np.nan
            fs = np.mean([flagged(s, g, it) for s in same]) if same else np.nan
            o(f"  {g:.2f}{cur}    {it:.2f}       {pct(tp):>6}          {pct(fc):>5} | {pct(fs)}")
    bad = [t for t, s in clean if flagged(s, isc.min_gap, isc.max_interleave_frac)]
    if bad:
        o(f"  labeled tracks flagged at the current values ({len(bad)}) — a real switch inside a "
          f"labeled track, or a false flag; check in the video:")
        for t in bad[:10]:
            o(f"      {t.sid} t{t.root} {t.mode} (cow {t.label})")
    o(f"  (min_cluster_frac fixed at {isc.min_cluster_frac}, min_frames {isc.min_frames}; "
      f"same-cow splices across sessions can differ by lighting — weigh the 'clean' column more)")


# ── window ───────────────────────────────────────────────────────────────────

def cmd_window(ctx: Ctx, truth: Truth, runs_per_track: int = 30) -> None:
    o, tp = ctx.out, config.params().timeline
    o.section("WINDOW — timeline.embed_mean_min_frames")
    tracks = track_vectors(ctx, truth, 40)
    if not tracks:
        o("  no labeled track with ≥40 effective frames")
        return
    rng = np.random.default_rng(RNG_SEED)
    cows = sorted({t.label for t in tracks})
    res = {}
    for t in tracks:
        refs = {c: cow_reference(tracks, c, t.mode, t, cross_session=False) for c in cows}
        refs = {c: v for c, v in refs.items() if v is not None}
        if t.label not in refs:
            continue
        rows = t.rows.sort_values("frame_index")
        stack = np.stack(rows.embed.values)
        w = rows[aq.QUALITY_COL].to_numpy(float)
        for n in (1, 2, 3, 5, 8, 12, 20, 30):
            if len(rows) < n:
                continue
            for s in rng.integers(0, len(rows) - n + 1, runs_per_track):
                v = aq._wmean(stack[s:s + n], w[s:s + n])
                sc = {c: float(v @ r) for c, r in refs.items()}
                res.setdefault(n, []).append((sc[t.label], max(sc, key=sc.get) == t.label))
    if not res:
        o("  needs a cow labeled in ≥2 tracks (a reference from her other tracks)")
        return
    o("  frames   cos to own cow p10/p50/p90    closest cow is right")
    sug = None
    for n in sorted(res):
        a = np.array([x[0] for x in res[n]])
        right = np.mean([x[1] for x in res[n]])
        if sug is None and right >= 0.9:
            sug = n
        o(f"  {n:>5}{'*' if n == tp.embed_mean_min_frames else ' '}   {q3(a)}       {pct(right)}")
    o(f"  → smallest run that is the right cow ≥90% of the time: {sug} "
      f"(now {tp.embed_mean_min_frames}; consecutive frames, so this is pessimistic for windows "
      f"that collect frames across 15 min)")


# ── kinetic ──────────────────────────────────────────────────────────────────

def cmd_kinetic(ctx: Ctx, truth: Truth, kinetics_paths: list) -> None:
    import kinetic_match as km
    o, P = ctx.out, config.params()
    o.section("KINETIC — kinetic_match.corr_threshold / min_active_bins vs manual labels")
    kin = pd.concat([pd.read_csv(p, parse_dates=["datetime"]) for p in kinetics_paths])
    o(f"  kinetics: {kin.AnimalId.nunique()} animals, {kin.datetime.min()} – {kin.datetime.max()}")
    allrec = []
    for sid in ctx.sessions:
        f = ctx.frames(sid)
        t0, t1 = f.frame_datetime.min(), f.frame_datetime.max()
        k = kin[(kin.datetime >= t0 - pd.Timedelta(hours=1)) & (kin.datetime <= t1 + pd.Timedelta(hours=1))]
        if k.empty:
            o(f"  {sid}: kinetics don't cover {t0} – {t1} — skipped")
            continue
        cmap = ctx.chain_map(sid)
        tr = f.assign(temp_id=f.temp_id.map(lambda t: cmap.get(int(t), int(t))))
        stable = km.stable_temp_ids(tr, P.kinetic_match.min_temp_id_frames)
        tr = tr[tr.temp_id.isin(stable)]
        bins = km.make_bins(tr.frame_datetime, P.timeline.bin_minutes)
        cam = km.camera_displacement_per_bin(tr, bins)
        kd = km.kinetics_delta_per_bin(kin, bins)
        labs = {root: rid for (s_, root), rid in truth.chain.items() if s_ == sid}
        missing = sorted({r for r in labs.values()} - set(kin.AnimalId))
        o(f"  {sid}: {len(stable)} stable tracks ({sum(1 for s in stable if s in labs)} labeled), "
          f"{len(bins) - 1} bins" + (f"; labeled cows without kinetics: {missing}" if missing else ""))
        for mab in sorted({1, 2, 3, 4, 6, P.kinetic_match.min_active_bins}):
            sc = km.compute_scores(cam, kd, P.kinetic_match.activity_pct, mab)
            sc["sid"], sc["mab"] = sid, mab
            sc["label"] = sc.temp_id.map(labs)
            allrec.append(sc)
    if not allrec:
        return
    S = pd.concat(allrec)
    o("  min_active_bins  corr_thr   assigned   wrong   labeled tracks found")
    for mab, g in S.groupby("mab"):
        for thr in (0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9):
            n_ass = n_wrong = n_found = 0
            n_lab = 0
            for sid, gs in g.groupby("sid"):
                a = km.greedy_assign(gs, thr)
                labs = dict(zip(gs.temp_id, gs.label))
                judged = {t: r for t, r in a.items() if pd.notna(labs.get(t))}
                n_ass += len(judged)
                n_wrong += sum(1 for t, r in judged.items() if r != labs[t])
                n_found += sum(1 for t, r in judged.items() if r == labs[t])
                n_lab += gs[gs.label.notna()].temp_id.nunique()
            cur = "*" if (thr == P.kinetic_match.corr_threshold and
                          mab == P.kinetic_match.min_active_bins) else " "
            o(f"  {mab:>8}        {thr:.1f}{cur}     {n_ass:>5}    {n_wrong:>4}    "
              f"{n_found}/{n_lab}")
    o("  (only labeled tracks are judged; '*' = current values)")


# ═════════════════════════════════════════════════════════════════════════════
# REVIEW
# ═════════════════════════════════════════════════════════════════════════════

REVIEW_COLS = ["category", "session_id", "temp_ids", "real_id", "frame", "time", "question",
               "detail", "command", "verdict", "note"]


def _first_frame(ctx: Ctx, sid: str, tids) -> int | None:
    f = ctx.frames(sid)
    cmap = ctx.chain_map(sid)
    tids = set(int(t) for t in tids)
    members = tids | {t for t, root in cmap.items() if root in tids}
    fr = f[f.temp_id.isin(members)].frame_index
    return int(fr.min()) if len(fr) else None


def cmd_review_export(ctx: Ctx, truth: Truth, path: str, per_category: int) -> None:
    rng = np.random.default_rng(RNG_SEED)
    rows = []

    def add(cat, sid, tids, rid, frame, question, detail=""):
        info = ctx.session_info(sid)
        f = ctx.frames(sid)
        t = f[f.frame_index == frame].frame_datetime
        rows.append({"category": cat, "session_id": sid, "temp_ids": " ".join(f"t{x}" for x in tids),
                     "real_id": rid, "frame": frame,
                     "time": str(t.iloc[0]) if len(t) else "", "question": question,
                     "detail": detail,
                     "command": f'python3 display_tracks.py --video "{info.get("video_path", "?")}" '
                                f'--start {max(0, (frame or 0) - 50)}',
                     "verdict": "", "note": ""})

    def sample(df):
        return df if len(df) <= per_category else df.iloc[rng.choice(len(df), per_category, replace=False)]

    sess = ctx.sessions
    if ctx.has_table("id_switch_flags"):
        fl = ctx.q("SELECT * FROM id_switch_flags")
        for r in sample(fl[fl.session_id.isin(sess)]).itertuples():
            sf = getattr(r, "switch_frame", None)
            frame = int(sf) if sf is not None and pd.notna(sf) else _first_frame(ctx, r.session_id, [r.temp_id])
            add("id_switch", r.session_id, [r.temp_id], r.real_id, frame,
                "y = it IS a switch: a different cow carries this id after this frame",
                f"{r.modality}, gap {r.split_gap:.2f}")
    if ctx.has_table("session_assignments"):
        sa = ctx.q("SELECT * FROM session_assignments WHERE written_by = session_id "
                   "AND temp_id = final_temp_id")
        sa = sa[sa.session_id.isin(sess)]
        for src, cat in (("cosine", "cosine_id"), ("boundary_rendezvous", "boundary_id"),
                         ("kinetic", "kinetic_id")):
            for r in sample(sa[(sa.source == src) & (sa.real_id > 0)]).itertuples():
                add(cat, r.session_id, [r.temp_id], r.real_id,
                    _first_frame(ctx, r.session_id, [r.temp_id]), f"y = this track is cow {r.real_id}")
        allsa = ctx.q("SELECT * FROM session_assignments")
        for r in sample(sa[sa.real_id < 0]).itertuples():
            other = allsa[(allsa.real_id == r.real_id) & (allsa.session_id != r.session_id)]
            add("synthetic_id", r.session_id, [r.temp_id], r.real_id,
                _first_frame(ctx, r.session_id, [r.temp_id]),
                "y = the same cow as the linked track(s) in the detail column",
                "; ".join(f"{o_.session_id} t{o_.temp_id}" for o_ in other.itertuples()))
        if ctx.has_table("gallery_contributions"):
            held = ctx.q("SELECT * FROM gallery_contributions WHERE status = 'held_for_review'")
            for r in sample(held[held.session_id.isin(sess)]).itertuples():
                tids = sa[(sa.session_id == r.session_id) & (sa.real_id == r.real_id)].temp_id.tolist()
                add("held_contribution", r.session_id, tids, r.real_id,
                    _first_frame(ctx, r.session_id, tids) if tids else None,
                    f"y = this track is cow {r.real_id} (then: gallery_ledger.py approve {r.id})",
                    f"{r.modality}/{r.gallery}" + (f"/slot{r.slot_idx}" if r.gallery == "pose" else "")
                    + f", drift {r.drift:.2f}, source {r.source}")
    if ctx.has_table("temp_id_merges"):
        mg = ctx.q("SELECT * FROM temp_id_merges")
        for r in sample(mg[mg.session_id.isin(sess)]).itertuples():
            same = r.action == "merged"
            add("merge" if same else "cooccurring", r.session_id, [r.winner_tid, r.loser_tid],
                r.real_id, _first_frame(ctx, r.session_id, [r.loser_tid]),
                "y = the two tracks are the SAME cow" if same else "y = the two tracks are DIFFERENT cows",
                f"{r.winner_reason} vs {r.loser_reason}")
    if ctx.has_table("tracklet_stitches"):
        st = ctx.q("SELECT * FROM tracklet_stitches")
        for r in sample(st[st.session_id.isin(sess)]).itertuples():
            add("stitch", r.session_id, [r.prev_tid, r.loser_tid], None,
                _first_frame(ctx, r.session_id, [r.loser_tid]),
                f"y = t{r.loser_tid} is the same cow as t{r.prev_tid} just before it",
                f"gap {r.gap_s:.1f}s, dist {r.dist:.2f}")
    for sid, root, labs in truth.conflicts:
        tids = [t for v in labs.values() for t in v]
        add("label_conflict", sid, tids, None, _first_frame(ctx, sid, tids),
            "which label is right? (or the stitch link is wrong) — fix with assign_identity.py",
            ", ".join(f"{k}←t{v}" for k, v in labs.items()))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=REVIEW_COLS)
        w.writeheader()
        w.writerows(rows)
    by = pd.Series([r["category"] for r in rows]).value_counts().to_dict() if rows else {}
    ctx.out(f"review queue: {len(rows)} rows → {path}  " + str(by))
    ctx.out("fill the 'verdict' column with y / n / ? and run: pipeline_check.py review --ingest " + path)


def cmd_review_ingest(out: Out, path: str) -> None:
    df = pd.read_csv(path, dtype=str).fillna("")
    out.section(f"REVIEW RESULTS — {path}")
    v = df.verdict.str.strip().str.lower().str[:1]
    for cat, g in df.assign(v=v).groupby("category"):
        y, n = int((g.v == "y").sum()), int((g.v == "n").sum())
        u = len(g) - y - n
        out(f"  {cat:18s} {len(g):>3} rows: y {y}, n {n}, unsure/blank {u}"
            + (f"  → pipeline right in {pct(y / (y + n))}" if y + n else ""))
        for r in g[g.v == "n"].head(10).itertuples():
            out(f"      n: {r.session_id} {r.temp_ids}" + (f" id {r.real_id}" if r.real_id else "")
                + f" @frame {r.frame}"
                + (f" — {r.note}" if r.note else ""))


# ═════════════════════════════════════════════════════════════════════════════
# main
# ═════════════════════════════════════════════════════════════════════════════

def open_data(args, out: Out):
    """(conn read-only, outputs_dir, gallery_dir)."""
    if args.db:
        db = Path(args.db).expanduser().resolve()
    else:
        from drive_manager import DriveManager, DriveNotSyncedError, DriveUnavailableError
        dm = DriveManager(bypass=args.bypass_upload_check, caller=__file__)
        try:
            dm.check_flag("db")
            db = dm.pull_db(allow_stale=args.bypass_upload_check)
            dm.pull_gallery()
        except (DriveNotSyncedError, DriveUnavailableError) as e:
            sys.exit(f"ERROR: {e}\nUse --bypass_upload_check, or --db for a local copy.")
    if args.outputs and args.gallery_dir:
        outputs, gallery = Path(args.outputs), Path(args.gallery_dir)
    else:
        import drive_manager
        outputs = Path(args.outputs) if args.outputs else drive_manager.LOCAL_OUTPUTS_DIR
        gallery = Path(args.gallery_dir) if args.gallery_dir else drive_manager.LOCAL_GALLERY_DIR
    from urllib.parse import quote
    conn = sqlite3.connect(f"file:{quote(str(db))}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    out(f"[pipeline_check] db {db}  outputs {outputs}  galleries {gallery}")
    return conn, outputs.expanduser(), gallery.expanduser()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["summary", "selfcheck", "snapshot", "diff", "labels",
                                        "quality", "trim", "cosine", "idswitch", "window",
                                        "kinetic", "review", "all"])
    ap.add_argument("files", nargs="*", help="diff: two snapshot files")
    ap.add_argument("--session", action="append", default=None,
                    help="limit to these session_ids (repeatable; default all)")
    ap.add_argument("--truth", choices=["manual", "kinetic", "all"], default="manual")
    ap.add_argument("--kinetics", action="append", default=None, help="kinetics CSV (repeatable)")
    ap.add_argument("--out", default=None, help="also write the report (or snapshot) to this file")
    ap.add_argument("--export", default=None, help="review: write the queue CSV here")
    ap.add_argument("--ingest", default=None, help="review: read verdicts from this CSV")
    ap.add_argument("--per_category", type=int, default=20, help="review: max rows per category")
    ap.add_argument("--max_error", type=float, default=0.02,
                    help="cosine: wrong-id rate allowed when suggesting a threshold")
    ap.add_argument("--max_pairs", type=int, default=300, help="idswitch: splices per kind")
    ap.add_argument("--db", default=None, help="local DB copy (no Drive access)")
    ap.add_argument("--outputs", default=None, help="dir with <session_id>/embeds_part*.parquet")
    ap.add_argument("--gallery_dir", default=None)
    ap.add_argument("--bypass_upload_check", action="store_true")
    config.add_params_arg(ap)
    args = ap.parse_args()
    config.init(args.params)

    if args.command == "diff":
        if len(args.files) != 2:
            ap.error("diff needs two snapshot files")
        out = Out(args.out)
        cmd_diff(out, *args.files)
        out.close()
        return
    if args.command == "review" and args.ingest:
        out = Out(args.out)
        cmd_review_ingest(out, args.ingest)
        out.close()
        return

    out = Out(None if args.command == "snapshot" else args.out)
    conn, outputs, gallery = open_data(args, out)
    sessions = args.session or all_sessions(conn)
    ctx = Ctx(conn, outputs, gallery, sessions, args.truth, out)
    truth = load_truth(ctx)
    cmd = args.command
    if cmd == "snapshot":
        if not args.out:
            ap.error("snapshot needs --out FILE")
        cmd_snapshot(ctx, args.out)
    elif cmd == "review":
        if not args.export:
            ap.error("review needs --export FILE or --ingest FILE")
        cmd_review_export(ctx, truth, args.export, args.per_category)
    elif cmd == "kinetic":
        if not args.kinetics:
            ap.error("kinetic needs --kinetics CSV")
        cmd_kinetic(ctx, truth, args.kinetics)
    else:
        steps = {"summary": [lambda: cmd_summary(ctx)],
                 "selfcheck": [lambda: cmd_selfcheck(ctx)],
                 "labels": [lambda: cmd_labels(ctx, truth)],
                 "quality": [lambda: cmd_quality(ctx)],
                 "trim": [lambda: cmd_trim(ctx, truth)],
                 "cosine": [lambda: cmd_cosine(ctx, truth, args.max_error)],
                 "idswitch": [lambda: cmd_idswitch(ctx, truth, args.max_pairs)],
                 "window": [lambda: cmd_window(ctx, truth)]}
        if cmd == "all":
            todo = [f for k in ("summary", "selfcheck", "labels", "quality", "trim", "cosine",
                                "idswitch", "window") for f in steps[k]]
            if args.kinetics:
                todo.append(lambda: cmd_kinetic(ctx, truth, args.kinetics))
        else:
            todo = steps[cmd]
        for f in todo:
            try:
                f()
            except Exception as e:      # one broken section must not hide the others
                import traceback
                out(f"  !! section crashed: {type(e).__name__}: {e}")
                out("     " + traceback.format_exc().strip().splitlines()[-3])
    out.close()
    conn.close()


if __name__ == "__main__":
    main()
