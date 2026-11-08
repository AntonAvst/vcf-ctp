"""
Calibrate tracklet stitching on real data. Real long tracks (>= 30 s) get a gap
cut out of them, splitting each into two temp_ids, with every other real track
left in place; the stitcher then runs and we count how many cuts are rejoined
and how many get a wrong link. Prints recall/wrong for the current params.yaml
values and a few variations.

Usage (from scripts/):  python3 stitch_calibration.py path/to/calving_project.db
Cut sessions are listed in `data` below — change them to your own sessions.
"""
import sys, itertools, numpy as np, pandas as pd
sys.path.insert(0,'.')  # run from scripts/
import config; config.init()
from types import SimpleNamespace as NS
import tracklet_stitch as T
import sqlite3
rows=pd.read_sql('select session_id,frame_index,frame_time_sec t,temp_id,x1,y1,x2,y2 from raw_tracks',
                 sqlite3.connect(sys.argv[1] if len(sys.argv)>1 else 'calving_project.db'))
GAPS=[0.5,1.0,2.0,3.0,5.0,8.0,12.0]
def make_cuts(sid, seed):
    d=rows[rows.session_id==sid].rename(columns={'t':'frame_time_sec'}).copy()
    rng=np.random.default_rng(seed)
    g=d.groupby('temp_id').frame_time_sec.agg(['min','max']); g['dur']=g['max']-g['min']
    long=g[g.dur>=30].index.tolist(); cuts=[]
    for tid in long:
        lo,hi=g.loc[tid,'min'],g.loc[tid,'max']
        tc=rng.uniform(lo+0.1*(hi-lo),hi-0.1*(hi-lo)); gap=rng.choice(GAPS)
        m=d.temp_id==tid
        d=d[~(m&(d.frame_time_sec>tc)&(d.frame_time_sec<=tc+gap))]
        tail=(d.temp_id==tid)&(d.frame_time_sec>tc+gap)
        if tail.sum()<3 or ((d.temp_id==tid)&(d.frame_time_sec<=tc)).sum()<3: continue
        d.loc[tail,'temp_id']=tid+1_000_000; cuts.append((tid,tid+1_000_000,gap))
    return d,cuts
def score(res,cuts):
    root=lambda t: res.tid_map.get(t,t)
    succ={l.prev_tid:l.next_tid for l in res.links}; pred={l.next_tid:l.prev_tid for l in res.links}
    ok=[root(h)==root(t) for h,t,_ in cuts]
    wrong=[(h in succ and succ[h]!=t) or (t in pred and pred[t]!=h) for h,t,_ in cuts]
    return np.array(ok),np.array(wrong)
base=vars(config.params().stitching).copy()   # current params.yaml values
data={sid:make_cuts(sid,7) for sid in ['cam26_27_S20260616180000_E20260616200000','cam33_6_S20260616100001_E20260616110000']}
for sid,(d,c) in data.items(): print(sid[:8],'cuts:',len(c))
def run(**kw):
    sp=NS(**{**base,**kw}); out=[]
    for sid,(d,cuts) in data.items():
        r=T.stitch(d,sp=sp); ok,wr=score(r,cuts); out.append((ok,wr,cuts,len(r.links)))
    ok=np.concatenate([o[0] for o in out]); wr=np.concatenate([o[1] for o in out]); gaps=np.array([g for o in out for _,_,g in o[2]])
    per={g:f"{ok[gaps==g].mean()*100:3.0f}" for g in GAPS}
    return ok.mean(),wr.mean(),per,sum(o[3] for o in out)
if __name__=='__main__':
    print(f"{'setting':55s} recall  wrong  recall-by-gap{GAPS}  links")
    grid=[{}]+[{'max_gap_s':v} for v in (5.5,12.5)]+[{'max_dist':v} for v in (0.3,1.0)]+[{'ambiguity_margin':v} for v in (0.0,0.3)]+[{'max_size_ratio':v} for v in (1.3,2.0)]
    for kw in grid:
        r,w,per,nl=run(**kw); print(f"{str(kw or 'base'):55s} {r*100:5.1f}% {w*100:5.1f}%  {per}  {nl}")
