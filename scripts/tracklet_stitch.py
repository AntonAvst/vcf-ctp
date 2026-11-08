"""
tracklet_stitch.py — join a cow's track fragments across short tracker dropouts.

The tracker gives a cow a new temp_id whenever it loses her for longer than
its buffer (occlusion, a missed detection on a small far-away box, a lying cow
flickering in and out). On the June 16 sessions one 2-hour video had 3,032
temp_ids for ~7 cows on screen, and only one track covered the 10% of frames
the Kinetic Matcher needs.

Purely geometric (positions, box sizes, timing), so it works the same for any
detector. Runs in reconcile.py right after tracks are loaded — before the
noise filter, so fragments get the chance to be joined before they are judged
too short — and before every identity step.

A link A → B (A ends, B starts later) is a candidate when:
  - B's first frame comes after A's last frame (the two are never on screen
    together), within stitching.max_gap_s
  - A's end point and B's start point (median box centre over the last / first
    end_window_s, which absorbs box jitter) are within
    max_dist + dist_per_s × gap, measured in box sizes (sqrt(w·h))
  - box sizes at the two ends differ by at most max_size_ratio
Links are accepted greedily by cost (distance, gap as tie-break), one-to-one:
each track gets at most one successor and one predecessor, so fragments form
chains A → B → C. A link is skipped when it is ambiguous (another candidate
for the same end or start costs within ambiguity_margin) or when it would
join two different manual assignments into one chain.

No velocity model: extrapolating each side's motion across the gap was tested
and gave no gain — over a few seconds the cows barely move relative to their
box size, and the fitted velocity of a jittery small box is mostly noise.

Calibration (June 16 data, cam26 + cam33_6): ~700 real long tracks per run
were split by cutting out a gap, with every other real track left in place,
over 3 random cut sets. With the defaults (max_gap_s 8.5): 90% of ≤3 s cuts
rejoined, 83% of 5–8 s cuts; 1.4–3.4% of cuts got a wrong link.

Every chain is renamed to its first (earliest) temp_id.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config


@dataclass
class StitchLink:
    prev_tid: int          # track that ends
    next_tid: int          # track that starts after it
    gap_s: float
    dist: float            # box sizes, at the gap midpoint
    size_ratio: float
    cost: float


@dataclass
class StitchResult:
    tid_map: dict = field(default_factory=dict)   # {loser temp_id -> chain root temp_id}
    links: list = field(default_factory=list)     # accepted StitchLink, in acceptance order
    n_candidates: int = 0
    n_ambiguous: int = 0
    n_manual_veto: int = 0

    @property
    def n_chains(self) -> int:
        return len(set(self.tid_map.values()))


def _end_point(t: np.ndarray, x: np.ndarray, y: np.ndarray, s: np.ndarray,
               at_end: bool, window_s: float) -> tuple:
    """(time, x, y, size) at one end of a track: the median over the rows in
    the last (or first) window_s seconds, which absorbs box jitter."""
    m = (t >= t[-1] - window_s) if at_end else (t <= t[0] + window_s)
    return (float(t[-1] if at_end else t[0]),
            float(np.median(x[m])), float(np.median(y[m])), float(np.median(s[m])))


def summarize_tracks(tracks: pd.DataFrame, end_window_s: float) -> pd.DataFrame:
    """One row per temp_id: frame range plus position and size at its start
    and end. tracks needs: temp_id, frame_index, t (seconds), x1, y1, x2, y2."""
    rows = []
    for tid, g in tracks.sort_values("t").groupby("temp_id", sort=False):
        t = g["t"].to_numpy(float)
        x = ((g["x1"] + g["x2"]) / 2).to_numpy(float)
        y = ((g["y1"] + g["y2"]) / 2).to_numpy(float)
        s = np.sqrt(np.clip((g["x2"] - g["x1"]) * (g["y2"] - g["y1"]), 1.0, None)).to_numpy(float)
        st = _end_point(t, x, y, s, at_end=False, window_s=end_window_s)
        en = _end_point(t, x, y, s, at_end=True,  window_s=end_window_s)
        rows.append({
            "temp_id": tid,
            "f0": int(g["frame_index"].min()), "f1": int(g["frame_index"].max()),
            "t0": st[0], "x0": st[1], "y0": st[2], "s0": st[3],
            "t1": en[0], "x1": en[1], "y1": en[2], "s1": en[3],
        })
    return pd.DataFrame(rows)


def find_candidates(summ: pd.DataFrame, sp) -> pd.DataFrame:
    """All (prev → next) pairs passing the gap / distance / size gates."""
    if summ.empty:
        return pd.DataFrame(columns=["prev", "next", "gap", "dist", "size_ratio", "cost"])
    starts = summ.sort_values("t0").reset_index(drop=True)
    t0s = starts["t0"].to_numpy()
    out = []
    for a in summ.itertuples(index=False):
        lo = np.searchsorted(t0s, a.t1, side="right")
        hi = np.searchsorted(t0s, a.t1 + sp.max_gap_s, side="right")
        for b in starts.iloc[lo:hi].itertuples(index=False):
            if b.temp_id == a.temp_id or b.f0 <= a.f1:
                continue
            gap = b.t0 - a.t1
            size = (a.s1 + b.s0) / 2.0
            dist = float(np.hypot(a.x1 - b.x0, a.y1 - b.y0) / max(size, 1.0))
            if dist > sp.max_dist + sp.dist_per_s * gap:
                continue
            ratio = max(a.s1, b.s0) / max(min(a.s1, b.s0), 1.0)
            if ratio > sp.max_size_ratio:
                continue
            out.append((a.temp_id, b.temp_id, gap, dist, ratio,
                        dist + 0.01 * gap / max(sp.max_gap_s, 1e-6)))
    return pd.DataFrame(out, columns=["prev", "next", "gap", "dist", "size_ratio", "cost"])


def stitch(tracks: pd.DataFrame, manual: dict | None = None, sp=None) -> StitchResult:
    """
    tracks : temp_id, frame_index, x1, y1, x2, y2, and frame_time_sec (seconds)
             or frame_datetime (used when frame_time_sec is missing/NULL)
    manual : {temp_id -> real_id} operator assignments; a chain never joins two
             different real_ids.
    """
    sp = sp or config.params().stitching      # sp override: parameter sweeps / tests
    res = StitchResult()
    if not sp.enabled or tracks.empty:
        return res

    df = tracks[["temp_id", "frame_index", "x1", "y1", "x2", "y2"]].copy()
    if "frame_time_sec" in tracks.columns and tracks["frame_time_sec"].notna().all():
        df["t"] = tracks["frame_time_sec"].astype(float).to_numpy()
    else:
        dt = pd.to_datetime(tracks["frame_datetime"])
        df["t"] = (dt - dt.min()).dt.total_seconds().to_numpy()

    summ = summarize_tracks(df, sp.end_window_s)
    cand = find_candidates(summ, sp)
    res.n_candidates = len(cand)
    if cand.empty:
        return res

    # second-best cost per prev and per next, for the ambiguity check
    def _second(col):
        srt = cand.sort_values("cost")
        return srt.groupby(col)["cost"].apply(lambda c: c.iloc[1] if len(c) > 1 else np.inf)
    second_for_prev = _second("prev")
    second_for_next = _second("next")

    manual = {int(k): v for k, v in (manual or {}).items()}
    root = {}                       # union-find over temp_ids
    chain_manual = {}               # root -> manual real_id in that chain (if any)

    def find(t):
        while root.get(t, t) != t:
            t = root[t]
        return t

    has_succ, has_pred = set(), set()
    for c in cand.sort_values("cost").itertuples(index=False):
        a, b = int(c.prev), int(c.next)
        if a in has_succ or b in has_pred:
            continue
        if (second_for_prev[c.prev] - c.cost < sp.ambiguity_margin or
                second_for_next[c.next] - c.cost < sp.ambiguity_margin):
            res.n_ambiguous += 1
            continue
        ra, rb = find(a), find(b)
        ma = chain_manual.get(ra, manual.get(a))
        mb = chain_manual.get(rb, manual.get(b))
        if ma is not None and mb is not None and ma != mb:
            res.n_manual_veto += 1
            continue
        has_succ.add(a)
        has_pred.add(b)
        root[rb] = ra                  # b's chain hangs under a's
        if ma is not None or mb is not None:
            chain_manual[ra] = ma if ma is not None else mb
        res.links.append(StitchLink(a, b, float(c.gap), float(c.dist),
                                    float(c.size_ratio), float(c.cost)))

    # chain root = earliest temp_id in the chain (a's side always starts first)
    members = {}
    for t in set(root) | set(root.values()):
        members.setdefault(find(t), []).append(t)
    t0 = summ.set_index("temp_id")["t0"]
    for r, mem in members.items():
        first = min(mem, key=lambda t: (t0[t], t))
        for t in mem:
            if t != first:
                res.tid_map[t] = first
    return res
