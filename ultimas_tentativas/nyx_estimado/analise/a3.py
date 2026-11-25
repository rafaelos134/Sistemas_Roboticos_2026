import numpy as np, sys; sys.path.insert(0, sys.argv[1]); from carregar import *
ms = {m: l for m, l, _ in partidas()}
for mid, t0, t1 in [(650, 92, 108), (639, 8, 25)]:
    d = carregar(mid, ms[mid]); t = d["t"] - d["t"][d["jogo"]][0]
    print(f"== partida {mid}: nyx_att | nyx_DEF | inim_att | inim_def | band_nyx | (H=preso C=carrega)")
    last = -9
    for k in np.nonzero((t >= t0) & (t <= t1))[0]:
        if t[k] - last < 0.5: continue
        last = t[k]
        f = lambda i: f"({d['P'][k,i,0]:+.2f},{d['P'][k,i,1]:+.2f},{np.degrees(d['P'][k,i,2]):+4.0f}){'H' if d['H'][k,i] else ' '}{'C' if d['C'][k,i] else ' '}"
        print(f"{t[k]:6.1f} {f(0)} {f(1)} {f(2)} {f(3)} ({d['F'][k,0,0]:+.2f},{d['F'][k,0,1]:+.2f})")
