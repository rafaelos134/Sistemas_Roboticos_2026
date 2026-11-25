import numpy as np, sys; sys.path.insert(0, sys.argv[1]); from carregar import *
A = [carregar(m, l) for m, l, _ in partidas()]
R = []  # por amostra do atacante do Nyx
for d in A:
    P, H, C, F, t, j = d["P"], d["H"], d["C"], d["F"], d["t"], d["jogo"]
    for k in range(1, len(t) - 1):
        dt = t[k+1] - t[k]
        if not j[k] or H[k, 0] or not (0.04 < dt < 0.3): continue
        me = P[k, 0]; nx = P[k+1, 0]
        d_ = nx[:2] - me[:2]
        v = (d_[0]*np.cos(me[2]) + d_[1]*np.sin(me[2])) / dt
        w = wrap(nx[2] - me[2]) / dt
        if np.hypot(*d_) > 0.08: continue
        od = P[k, 3]; oa = P[k, 2]
        R.append(dict(x=me[0], y=me[1], v=v, w=w, carry=C[k, 0], d_def=np.hypot(*(od[:2]-me[:2])),
                      def_x=od[0], def_held=H[k, 3], ef_dist=np.hypot(*(F[k, 1]-me[:2])), mid=d["mid"],
                      ang_flag=np.degrees(abs(wrap(np.arctan2(F[k,1,1]-me[1], F[k,1,0]-me[0]) - me[2]))),
                      ang_home=np.degrees(abs(wrap(np.arctan2(-me[1], -0.62-me[0]) - me[2]))),
                      immune=False))
import pandas as pd
df = pd.DataFrame(R); df["parado"] = (df.v.abs() < 0.02) & (df.w.abs() < 0.3)
print("ATACANTE Nyx: n", len(df), " parado", f"{df.parado.mean():.0%}", " v p50/p90", df.v.abs().quantile([.5,.9]).round(3).tolist(), " |w| p90", df.w.abs().quantile(.9).round(2))
df["zona"] = pd.cut(df.x, [-1, -0.3, 0, 0.3, 0.45, 1])
print(df.groupby(["carry", "zona"], observed=True).agg(n=("v", "size"), parado=("parado", "mean"), v=("v", "median"),
      d_def=("d_def", "median"), ang_flag=("ang_flag", "median"), ang_home=("ang_home", "median")).round(2))
nc = df[~df.carry]
print("\nsem bandeira, no campo inimigo (x>0): parado vs distância ao defensor inimigo")
nc2 = nc[nc.x > 0].copy(); nc2["dd"] = pd.cut(nc2.d_def, [0, .1, .15, .2, .25, .3, .4, .6, 2])
print(nc2.groupby("dd", observed=True).agg(n=("v", "size"), parado=("parado", "mean"), v=("v", "median"), ef=("ef_dist", "median")).round(3))
print("\nonde fica parado (sem bandeira): x,y p10/50/90")
p = nc[nc.parado]; print(p[["x", "y", "d_def", "ef_dist"]].quantile([.1, .5, .9]).round(2))
