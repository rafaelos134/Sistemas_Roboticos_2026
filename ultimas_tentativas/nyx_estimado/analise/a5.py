import numpy as np, sys; sys.path.insert(0, sys.argv[1]); from carregar import *
ms = {m: (l, o) for m, l, o in partidas()}
for mid, t0, t1 in [(672 if 672 in ms else list(ms)[3], 0, 16), (list(ms)[10], 30, 50)]:
    l, opp = ms[mid]; d = carregar(mid, l); t = d["t"] - d["t"][d["jogo"]][0]
    print(f"== partida {mid} (Nyx x {opp}): NYX_ATT | nyx_def | inim_att | inim_DEF | band_inimiga")
    last = -9
    for k in np.nonzero((t >= t0) & (t <= t1))[0]:
        if t[k] - last < 0.4: continue
        last = t[k]
        f = lambda i: f"({d['P'][k,i,0]:+.2f},{d['P'][k,i,1]:+.2f},{np.degrees(d['P'][k,i,2]):+4.0f}){'H' if d['H'][k,i] else ' '}{'C' if d['C'][k,i] else ' '}"
        print(f"{t[k]:6.1f} {f(0)} {f(1)} {f(2)} {f(3)} ({d['F'][k,1,0]:+.2f},{d['F'][k,1,1]:+.2f}) placar {d['score'][k].tolist()}")
